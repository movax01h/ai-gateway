"""Tests for the deterministic bl_dedup_findings tool.

Every test drives the tool through its public ``ainvoke`` (the path the flow's
deterministic step takes). The tool does no I/O, so nothing is faked.
"""

import asyncio
import json

import pytest
from langchain_core.tools import ToolException
from structlog.testing import capture_logs

from duo_workflow_service.bl_security.findings import TRIAGE_CLAUSE_MAX
from duo_workflow_service.tools.bl_report import BlDedupFindings, BlDedupFindingsInput
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD


@pytest.fixture(name="tool")
def tool_fixture():
    return BlDedupFindings(metadata={"outbox": object()})


def _run(tool, batches, **args):
    return asyncio.run(tool.ainvoke({"batches": batches, **args}))


def _events(logs, event):
    return [e for e in logs if e["event"] == event]


def test_the_bl_tools_are_hidden_from_list_tools():
    # ListTools publishes only tools at or above STABLE_VERSION_THRESHOLD.
    assert BlDedupFindings.tool_version < STABLE_VERSION_THRESHOLD


class TestInlineInput:
    def test_an_inline_json_string_is_read_without_io(self, tool):
        # Single source, passed as a JSON *string* rather than a list.
        inline = json.dumps([[{"cwe": "CWE-284", "file": "a.py", "line": 1}]])
        out = _run(tool, inline)

        assert [f["cwe"] for f in out] == ["CWE-284"]


# --------------------------------------------------------------------------- #
# BlDedupFindings: scope-drop + facet collapse
# --------------------------------------------------------------------------- #
class TestDedupLogic:
    def test_a_per_batch_element_that_is_a_json_string_is_flattened(self, tool):
        out = _run(tool, [['{"cwe": "CWE-1", "file": "a.py"}']])
        assert out == [{"cwe": "CWE-1", "file": "a.py"}]

    def test_drop_out_of_scope_removes_a_foreign_cwe(self, tool):
        findings = [
            {"cwe": "CWE-918", "file": "a.py", "line": 1},  # SSRF is SAST's
            {"cwe": "CWE-639", "file": "c.py", "line": 3},
        ]
        out = _run(tool, findings, drop_out_of_scope=True)
        assert [f["file"] for f in out] == ["c.py"]

    def test_drop_out_of_scope_removes_a_noise_keyword_in_scope_cwe(self, tool):
        findings = [
            {
                "cwe": "CWE-639",
                "file": "b.py",
                "line": 2,
                "body": "susceptible to rate limiting bypass",
            },
            {"cwe": "CWE-639", "file": "c.py", "line": 3},
        ]
        out = _run(tool, findings, drop_out_of_scope=True)
        assert [f["file"] for f in out] == ["c.py"]

    def test_semantic_classes_survive_the_scope_drop(self, tool):
        # CWE-362/367/200 are core (race/TOCTOU) and qualified-in (data
        # exposure) BL classes; the scope drop must keep them.
        cwes = ("362", "367", "200", "863", "285")
        findings = [
            {"cwe": f"CWE-{cwe}", "file": f"{cwe}.py", "line": 1} for cwe in cwes
        ]
        out = _run(tool, findings, drop_out_of_scope=True)
        assert {f["file"] for f in out} == {f"{cwe}.py" for cwe in cwes}

    @pytest.mark.parametrize(
        ("cwe", "kept"),
        [
            # Conservative by design: only a POSITIVELY identified foreign class
            # is dropped, so a finding with no parseable CWE survives for triage.
            (None, True),
            ("CWE-327", False),
            ("CWE-862", True),
        ],
    )
    def test_only_a_known_foreign_cwe_is_dropped_as_out_of_scope(self, tool, cwe, kept):
        finding = {"file": "a.py", "line": 1}
        if cwe is not None:
            finding["cwe"] = cwe
        out = _run(tool, [finding], drop_out_of_scope=True)
        assert (out == [finding]) is kept

    def test_an_idor_enumeration_finding_is_kept_by_the_scope_drop(self, tool):
        finding = {
            "cwe": "CWE-639",
            "file": "a.py",
            "line": 1,
            "body": "Sequential IDs allow enumeration of other users' invoices",
        }
        out = _run(tool, [finding], drop_out_of_scope=True)
        assert out == [finding]

    def test_an_exact_duplicate_in_one_line_bucket_collapses(self, tool):
        # Same CWE, file, line//4 (10 // 4 == 11 // 4) and excerpt.
        excerpt = "user = User.find(params[:id])"
        findings = [
            {"cwe": "CWE-639", "file": "a.py", "line": 10, "code_excerpt": excerpt},
            {"cwe": "CWE-639", "file": "a.py", "line": 11, "code_excerpt": excerpt},
        ]
        out = _run(tool, findings)
        assert [f["line"] for f in out] == [10]

    def test_a_relabelled_statement_collapses_despite_whitespace_and_case(self, tool):
        # Different CWE and a distant line, but the SAME statement once the
        # excerpt is whitespace/case-normalized -> second-pass collapse.
        findings = [
            {
                "cwe": "CWE-639",
                "file": "a.py",
                "line": 10,
                "code_excerpt": "user = User.find(params[:id])",
            },
            {
                "cwe": "CWE-284",
                "file": "a.py",
                "line": 40,
                "code_excerpt": "   USER = User.find(params[:id])\n",
            },
        ]
        out = _run(tool, findings)
        assert [f["cwe"] for f in out] == ["CWE-639"]


class TestAFailedUpstreamStepFailsTheStep:
    """A failed flow step publishes ``None``, and the next step still runs.

    These go through ``ainvoke``, the path the flow's step takes, so a ``handle_tool_error`` that turned the raise into
    returned text would fail them.
    """

    @pytest.mark.parametrize(
        "args",
        [
            {"batches": None},
            {"batches": "Error: x"},
            {"batches": "[]", "batches2": "Error: x"},
        ],
    )
    def test_dedup_refuses(self, tool, args):
        with pytest.raises(ToolException, match="earlier step most likely failed"):
            asyncio.run(tool.ainvoke(args))

    @pytest.mark.parametrize(
        "batches",
        [
            {"error": "step failed"},
            {},
            {"findings_files": ["bl-collect-0.jsonl"], "count": 1},
        ],
    )
    def test_dedup_refuses_a_dict(self, tool, batches):
        with pytest.raises(ToolException, match="earlier step most likely failed"):
            _run(tool, batches)

    @pytest.mark.parametrize("batches", [[], "[]"])
    def test_dedup_of_an_empty_list_is_empty(self, tool, batches):
        assert _run(tool, batches) == []


class TestMultiSampleUnionDedup:
    """N reviews of the SAME authz unit are fed to bl_dedup_findings as N batches.

    A subtle vuln caught in only ONE of the N samples must SURVIVE (union), while a vuln caught in every sample must
    collapse to a single finding (no N-fold double-report).
    """

    def test_finding_in_one_of_n_samples_survives_union(self, tool):
        # Subtle authz bug the model catches only INTERMITTENTLY: present in
        # sample 1, MISSED in samples 2 and 3.
        rare = {
            "cwe": "CWE-639",
            "file": "routers/api/v1/repo/pull.go",
            "line": 42,
            "code_excerpt": "if !ctx.IsSigned { return }",
        }
        # A bug EVERY sample catches -> must not be reported three times.
        common = {
            "cwe": "CWE-284",
            "file": "routers/api/v1/org/team.go",
            "line": 10,
            "code_excerpt": "AddTeamRepository(org, repo)",
        }
        # Three samples of the SAME authz unit (bl_dedup_findings batches shape).
        batches = [[rare, common], [common], [common]]

        out = _run(tool, batches)
        files = [f["file"] for f in out]
        # UNION: the intermittently-caught vuln survives despite 1/3 coverage.
        assert "routers/api/v1/repo/pull.go" in files, (
            "intermittent finding lost by dedup"
        )
        # the always-caught vuln collapses to exactly one (identical-collapse).
        assert files.count("routers/api/v1/org/team.go") == 1
        assert len(out) == 2


class TestSiblingScanSecondSource:
    """``batches2`` merges a second detection pass with the first before dedup.

    Both sources are flattened independently and concatenated. A finding only one pass caught has to survive the merge,
    and one both passes caught must collapse to a single report.
    """

    def test_second_source_is_flattened_and_merged(self, tool):
        shared = {
            "cwe": "CWE-284",
            "file": "routers/api/v1/org/team.go",
            "line": 10,
            "code_excerpt": "AddTeamRepository(org, repo)",
        }
        map_blob = json.dumps(
            [
                [
                    {
                        "cwe": "CWE-639",
                        "file": "routers/api/v1/repo/pull.go",
                        "line": 42,
                    },
                    shared,
                ]
            ]
        )
        sibling_blob = json.dumps(
            [
                [
                    shared,
                    {"cwe": "CWE-285", "file": "routers/api/v1/user/key.go", "line": 7},
                ]
            ]
        )
        out = _run(tool, map_blob, batches2=sibling_blob)

        files = [f["file"] for f in out]
        assert "routers/api/v1/repo/pull.go" in files  # map_reviews only
        assert "routers/api/v1/user/key.go" in files  # sibling_scan only
        # caught by BOTH passes -> reported once
        assert files.count("routers/api/v1/org/team.go") == 1
        assert len(out) == 3

    def test_an_inline_second_source_is_merged_without_io(self, tool):
        # Both sources inline Python lists: `batches2` is flattened and
        # concatenated without any read.
        out = _run(
            tool,
            [[{"cwe": "CWE-284", "file": "a.go", "line": 1}]],
            batches2=[[{"cwe": "CWE-639", "file": "b.go", "line": 2}]],
        )
        assert [f["file"] for f in out] == ["a.go", "b.go"]

    def test_none_second_source_leaves_findings_untouched(self, tool):
        out = _run(
            tool, [[{"cwe": "CWE-284", "file": "a.go", "line": 1}]], batches2=None
        )
        assert [f["file"] for f in out] == ["a.go"]


# --------------------------------------------------------------------------- #
# ONE FINDING PER VULNERABLE LOCATION
#
# A single file can hold N separate sites that each miss the SAME control — N
# near-identical resolvers a few lines apart, each independently reachable and
# each independently exploitable. Fixing N-1 of them leaves the app vulnerable, so
# each site is its own finding. Dedup must still collapse a true duplicate of
# ONE site, and one statement re-labelled under a second CWE.
#
# Fixtures below are synthetic by construction (generic `svc_*` / `handler_*`
# names, no real repository, symbol, or advisory content).
# --------------------------------------------------------------------------- #


def _site(idx, line, cwe="CWE-639", excerpt=None, body=None):
    """One vulnerable site in the shared fixture file."""
    return {
        "cwe": cwe,
        "file": "svc/handlers.py",
        "new_line": line,
        "code_excerpt": (
            excerpt if excerpt is not None else f"return Widget{idx}.objects.all()"
        ),
        "body": body if body is not None else f"handler_{idx} misses its owner check",
    }


def _finding(**over):
    base = {
        "cwe": "CWE-862",
        "file": "svc/hooks.py",
        "new_line": 42,
        "severity": "high",
        "code_excerpt": "func Handle(w, r) {",
        "body": "webhook secret returned to a non-admin caller",
    }
    base.update(over)
    return base


class TestPerLocationGranularity:
    def test_forty_distinct_sites_in_one_file_all_survive(self, tool):
        # The shape that motivated this: 40 near-identical handlers on a 5-line
        # stride, all missing the same control, so all carrying the same CWE.
        out = _run(tool, [_site(i, 100 + i * 5) for i in range(40)])
        assert len(out) == 40
        assert len({f["new_line"] for f in out}) == 40

    @pytest.mark.parametrize("stride", [1, 2, 3, 4, 5, 8])
    def test_distinct_sites_survive_every_stride(self, tool, stride):
        # The line bucket is line//4, so strides 1-3 put adjacent sites in ONE
        # bucket. They must still survive on the strength of their excerpts.
        out = _run(tool, [_site(i, 100 + i * stride) for i in range(40)])
        assert len(out) == 40

    def test_adjacent_sites_in_one_line_bucket_both_survive(self, tool):
        # Same CWE, same line bucket (100//4 == 101//4 == 25), different code.
        out = _run(tool, [_site(0, 100), _site(1, 101)])
        assert len(out) == 2

    def test_repeated_identical_statement_same_cwe_survives(self, tool):
        # Byte-identical excerpts far apart under ONE CWE: N copy-pasted sites,
        # not a re-report; an excerpt-only collapse would keep just one.
        shared = "return self.queryset"
        out = _run(tool, [_site(i, 100 + i * 50, excerpt=shared) for i in range(5)])
        assert len(out) == 5

    def test_true_duplicate_of_one_site_still_collapses(self, tool):
        # Same CWE, same site, same statement, emitted three times.
        out = _run(tool, [_site(0, 100), _site(0, 100), _site(0, 101)])
        assert len(out) == 1

    def test_one_statement_relabelled_under_a_second_cwe_collapses(self, tool):
        # Facet explosion: ONE statement re-reported under another CWE at a
        # scattered (LLM-guessed) anchor. Still one defect, still collapses.
        out = _run(tool, [_site(0, 100), _site(0, 260, cwe="CWE-284")])
        assert len(out) == 1
        assert out[0]["cwe"] == "CWE-639"

    def test_a_stray_relabel_sorting_first_cannot_evict_the_real_sites(self, tool):
        # The regression a keep-first collapse would cause: ONE finding carrying a
        # minority CWE arrives ahead of the N sites that share the statement. The
        # surviving CWE is decided by frequency, so the N sites win and the stray
        # re-label is the one that goes.
        shared = "return self.queryset"
        findings = [_site(0, 100, cwe="CWE-284", excerpt=shared)] + [
            _site(i, 200 + i * 50, excerpt=shared) for i in range(5)
        ]
        out = _run(tool, findings)
        assert len(out) == 5
        assert {f["cwe"] for f in out} == {"CWE-639"}

    def test_tied_cwes_on_one_statement_keep_the_highest_ranked(self, tool):
        # 1-vs-1 is the classic facet case: no majority, so arrival (ranking)
        # order decides — unchanged behaviour.
        shared = "return self.queryset"
        out = _run(
            tool,
            [
                _site(0, 100, cwe="CWE-863", excerpt=shared),
                _site(1, 300, cwe="CWE-284", excerpt=shared),
            ],
        )
        assert len(out) == 1
        assert out[0]["cwe"] == "CWE-863"

    def test_each_statement_resolves_its_own_dominant_cwe(self, tool):
        # The dominant CWE is per STATEMENT, not per file. Two unrelated
        # statements in one file each pick up their own stray re-label, and the
        # winner differs between them (CWE-639 wins the first group and LOSES
        # the second). Resolving one group must leave the other's distinct
        # sites untouched — a per-file winner would evict half of them.
        stmt_a, stmt_b = "return self.queryset", "obj = Model.objects.get(pk=pk)"
        out = _run(
            tool,
            [
                _site(0, 100, cwe="CWE-284", excerpt=stmt_a),
                _site(1, 200, excerpt=stmt_a),
                _site(2, 250, excerpt=stmt_a),
                _site(3, 400, cwe="CWE-639", excerpt=stmt_b),
                _site(4, 500, cwe="CWE-862", excerpt=stmt_b),
                _site(5, 550, cwe="CWE-862", excerpt=stmt_b),
            ],
        )
        assert {(f["new_line"], f["cwe"]) for f in out} == {
            (200, "CWE-639"),
            (250, "CWE-639"),
            (500, "CWE-862"),
            (550, "CWE-862"),
        }

    def test_excerptless_findings_are_never_collapsed_by_the_second_pass(self, tool):
        # Findings with no code_excerpt keep the exact pass
        # as their only dedup and must not be merged on empty-excerpt equality.
        out = _run(
            tool,
            [
                {"cwe": "CWE-862", "file": "svc/handlers.py", "new_line": 10},
                {"cwe": "CWE-862", "file": "svc/handlers.py", "new_line": 90},
            ],
        )
        assert len(out) == 2

    def test_cweless_findings_neither_vote_nor_evict_a_tagged_finding(self, tool):
        # Same statement, one copy with no CWE: it casts no vote for the
        # dominant CWE and is not itself evicted by the tagged one.
        untagged = _finding(cwe="", new_line=10)
        tagged = _finding(cwe="CWE-89", new_line=50)
        assert _run(tool, [untagged, tagged]) == [untagged, tagged]


class TestTriageVerdictAudit:
    """Every adjudication decision must be recorded, and a DROP must still drop.

    The adjudicator returns the finding annotated with verdict/clause/evidence for BOTH verdicts; this tool logs every
    decision and removes only the DROPs.
    """

    @staticmethod
    def _audit(tool, findings):
        with capture_logs() as logs:
            out = _run(tool, [findings])
        return out, logs

    def test_a_drop_verdict_removes_the_finding_and_records_why(self, tool):
        out, logs = self._audit(
            tool,
            [
                _finding(
                    verdict="DROP",
                    clause="DROP-4",
                    evidence="svc/hooks.py:42 consumer is an admin-configured webhook",
                )
            ],
        )
        assert out == []  # behaviour unchanged: a DROP still reaches no report
        (audit,) = _events(logs, "bl_triage_verdict")
        assert audit["verdict"] == "DROP"
        assert audit["clause"] == "DROP-4"
        assert "admin-configured webhook" in audit["evidence"]
        assert audit["file"] == "svc/hooks.py"
        assert audit["line"] == 42
        assert audit["cwe"] == "862"

    def test_secrets_in_audited_text_are_redacted(self, tool):
        token = "glpat-abcdefghijklmnopqrst12"
        _, logs = self._audit(
            tool,
            [
                _finding(
                    verdict="KEEP",
                    clause=f"KEEP {token}",
                    evidence=f"svc/hooks.py:42 key = '{token}'",
                    body=f"leaks {token}",
                )
            ],
        )
        (audit,) = _events(logs, "bl_triage_verdict")
        for field in ("clause", "evidence", "body"):
            assert token not in audit[field]
            assert "[REDACTED]" in audit[field]

    def test_a_keep_verdict_survives_and_is_recorded_too(self, tool):
        out, logs = self._audit(
            tool,
            [
                _finding(
                    verdict="KEEP",
                    clause="KEEP-cross-principal",
                    evidence="svc/hooks.py:42 no owner check",
                )
            ],
        )
        assert len(out) == 1
        # The audit fields ride along on the kept finding, so the intermediate
        # artifact is auditable and not just the log.
        assert out[0]["verdict"] == "KEEP"
        assert out[0]["clause"] == "KEEP-cross-principal"
        (audit,) = _events(logs, "bl_triage_verdict")
        assert audit["verdict"] == "KEEP"

    def test_the_summary_accounts_for_every_candidate(self, tool):
        _, logs = self._audit(
            tool,
            [
                _finding(verdict="KEEP", clause="KEEP-sibling-asymmetry", file="a.py"),
                _finding(verdict="DROP", clause="DROP-1", file="b.py"),
                _finding(verdict="DROP", clause="DROP-5", file="c.py"),
                _finding(file="d.py"),  # model omitted the verdict entirely
            ],
        )
        (summary,) = _events(logs, "bl_triage_verdicts summary")
        assert summary["candidates"] == 4
        assert summary["adjudicated"] == 3
        assert summary["unlabelled"] == 1
        assert summary["dropped"] == 2
        assert summary["kept"] == 2

    def test_a_finding_with_no_verdict_is_kept_not_crashed(self, tool):
        # The model may ignore the contract. The old contract was "an object
        # means KEEP", so an unlabelled finding must still be reported -- never
        # dropped, and never an exception.
        out, logs = self._audit(
            tool, [_finding(), _finding(verdict="DROP", file="b.py")]
        )
        assert [f["file"] for f in out] == ["svc/hooks.py"]
        assert [a["verdict"] for a in _events(logs, "bl_triage_verdict")] == ["DROP"]

    def test_missing_clause_and_evidence_are_recorded_as_unspecified(self, tool):
        out, logs = self._audit(tool, [_finding(verdict="DROP")])
        assert out == []
        (audit,) = _events(logs, "bl_triage_verdict")
        assert audit["clause"] == "UNSPECIFIED"
        assert audit["evidence"] == "UNSPECIFIED"

    def test_legacy_triage_evidence_is_used_when_evidence_is_absent(self, tool):
        _, logs = self._audit(
            tool,
            [_finding(verdict="KEEP", triage_evidence="svc/hooks.py:42 legacy field")],
        )
        (audit,) = _events(logs, "bl_triage_verdict")
        assert audit["evidence"] == "svc/hooks.py:42 legacy field"

    @pytest.mark.parametrize(
        ("field", "bound"),
        [("clause", TRIAGE_CLAUSE_MAX), ("evidence", 600), ("body", 240)],
    )
    def test_model_written_text_is_bounded_in_the_audit_record(
        self, tool, field, bound
    ):
        # A 500-finding scan must not be able to flood the log through the
        # model-authored strings, so each is cut to its bound.
        _, logs = self._audit(
            tool, [_finding(verdict="DROP", **{field: "x" * (bound + 100)})]
        )
        (audit,) = _events(logs, "bl_triage_verdict")
        assert audit[field] == "x" * bound

    def test_an_unrecognised_verdict_keeps_the_finding(self, tool):
        # Only an explicit DROP removes anything; anything else falls back to the
        # old "object present means KEEP" behaviour rather than silently deleting.
        out, _ = self._audit(tool, [_finding(verdict="MAYBE")])
        assert len(out) == 1

    def test_verdict_matching_is_case_and_whitespace_insensitive(self, tool):
        out, _ = self._audit(tool, [_finding(verdict=" drop ")])
        assert out == []

    def test_an_empty_per_batch_array_flattens_away(self, tool):
        # A per-unit array of [] must flatten to nothing rather than raise.
        out = _run(tool, [[], [_finding(verdict="KEEP")]])
        assert [f["verdict"] for f in out] == ["KEEP"]

    def test_findings_with_no_verdicts_at_all_are_untouched(self, tool):
        # The PRE-triage `adjudicate` call runs this same tool over detection
        # output, which carries no verdicts. It must be a strict no-op there:
        # nothing dropped, and no audit log emitted.
        out, logs = self._audit(tool, [_finding(file="a.py"), _finding(file="b.py")])
        assert [f["file"] for f in out] == ["a.py", "b.py"]
        assert _events(logs, "bl_triage_verdict") == []
        assert _events(logs, "bl_triage_verdicts summary") == []


class TestFormatDisplayMessage:
    """The default DuoBaseTool.format_display_message dumps every arg's str() into the chat-log `content` field with no
    size cap -- unlike tool_info.tool_response, which IS capped at TOOL_RESPONSE_MAX_DISPLAY_MSG.

    `batches`/`batches2`/`findings` here can carry the full inline findings
    blob; this override must never let
    that reach `content`.
    """

    def test_dedup_reports_response_count_not_content(self, tool):
        args = BlDedupFindingsInput(batches="x" * 10_000)
        message = tool.format_display_message(
            args, [{"cwe": "CWE-284"}, {"cwe": "CWE-639"}]
        )
        assert message == "Deduplicated to 2 findings"
        assert "x" * 100 not in message

    def test_dedup_falls_back_without_a_list_response(self, tool):
        args = BlDedupFindingsInput(batches="x" * 10_000)
        message = tool.format_display_message(args, None)
        assert message == "Deduplicated to ? findings"
