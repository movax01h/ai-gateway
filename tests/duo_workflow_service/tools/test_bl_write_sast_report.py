"""Tests for the deterministic bl_write_sast_report tool."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import ToolException
from structlog.testing import capture_logs

import duo_workflow_service.tools.bl_report as dedup_mod
import duo_workflow_service.tools.bl_write_sast_report as bl
from contract import contract_pb2
from duo_workflow_service.bl_security.findings import TRIAGE_CLAUSE_MAX
from duo_workflow_service.tools.bl_report import BlDedupFindings
from duo_workflow_service.tools.bl_write_sast_report import (
    BlWriteSastReport,
    BlWriteSastReportInput,
)
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD


def _dedup():
    return BlDedupFindings(metadata={"outbox": object()})


def _writer():
    return BlWriteSastReport(metadata={"outbox": object()})


def test_the_bl_tools_are_hidden_from_list_tools():
    # ListTools publishes only tools at or above STABLE_VERSION_THRESHOLD.
    assert BlWriteSastReport.tool_version < STABLE_VERSION_THRESHOLD


def test_the_report_writer_takes_no_project_id():
    """Nothing in the report comes from the project id, so the tool does not accept one."""
    assert "project_id" not in BlWriteSastReportInput.model_fields


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_find_excerpt_line_empty_and_no_match(self):
        assert bl._find_excerpt_line("", "x") == 0
        assert bl._find_excerpt_line("some content", "") == 0
        assert bl._find_excerpt_line("a\nb\nc", "zzz") == 0

    def test_find_excerpt_line_hint_picks_closest(self):
        content = "x\nfoo(bar)\ny\nfoo(bar)\nz"
        # two matches on lines 2 and 4; hint biases toward the closer one
        assert bl._find_excerpt_line(content, "foo(bar)", hint=4) == 4
        assert bl._find_excerpt_line(content, "foo(bar)") == 2

    def test_find_excerpt_line_first_line_fallback(self):
        # whole multi-line excerpt does not match, but its first line does
        content = "def handler():\n    do_thing()\n    return 1"
        assert bl._find_excerpt_line(content, "def handler():\n    REFORMATTED") == 1

    def test_find_excerpt_line_prefers_the_line_that_pins_one_site(self):
        """A quote opening on a repeated annotation must anchor on its unique line.

        The excerpt opens with a decorator that recurs at an unrelated site far away. Anchoring on the FIRST excerpt
        line makes the finding land on that unrelated site; the signature line inside the same excerpt occurs once and
        identifies the real one.
        """
        content = "\n".join(
            ["@guard_post"]  # line 1 - the decoy occurrence
            + ["    def unrelated_sibling(self):"]
            + ["filler"] * 20
            + ["@guard_post"]  # line 23 - the real site
            + ["    def target_handler(self):"]  # line 24
            + ["        return do_work()"]
        )
        excerpt = "@guard_post\n    def target_handler(self):\n        return do_work()"
        # hint is absent/wrong, exactly the condition the anchor pass exists for
        assert bl._find_excerpt_line(content, excerpt) == 24
        assert bl._find_excerpt_line(content, excerpt, hint=1) == 24
        assert bl._find_excerpt_line(content, excerpt, hint=999) == 24

    def test_find_excerpt_line_ambiguous_everywhere_still_uses_hint(self):
        """When no excerpt line is unique, fall back to fewest-matches + hint."""
        content = "\n".join(["a()", "b()", "filler", "a()", "b()"])
        excerpt = "a()\nb()"
        assert bl._find_excerpt_line(content, excerpt, hint=5) == 4
        assert bl._find_excerpt_line(content, excerpt) == 1

    def test_find_excerpt_line_ignores_structural_only_excerpt_lines(self):
        """A bare `}` in the excerpt must never become the search key."""
        content = "\n".join(["}", "}", "}", "int unique_marker = 1;", "}"])
        assert bl._find_excerpt_line(content, "int unique_marker = 1;\n}") == 4


# --------------------------------------------------------------------------- #
# BlWriteSastReport._build_report (sync)
# --------------------------------------------------------------------------- #
class TestBuildReport:
    def test_schema_identity_and_cwe_identifier(self):
        findings = [
            {
                "cwe": "CWE-639",
                "file": "app/x.rb",
                "new_line": 12,
                "severity": "high",
                "body": "IDOR on record lookup",
                "code_excerpt": "Record.find(params[:id])",
            }
        ]
        report = _writer()._build_report(findings)
        # 15.2.2 is the first schema version defining `scan.partial_scan`.
        assert report["version"] == bl._REPORT_SCHEMA_VERSION == "15.2.2"
        # analyzer + scanner identity
        assert report["scan"]["analyzer"]["id"] == "gitlab_bl_security_analyzer"
        assert report["scan"]["scanner"]["vendor"]["name"] == "GitLab"
        assert report["scan"]["status"] == "success"
        assert "partial_scan" not in report["scan"]
        v = report["vulnerabilities"][0]
        assert v["severity"] == "High"
        assert v["location"] == {"file": "app/x.rb", "start_line": 12}
        ident = v["identifiers"][0]
        assert ident["type"] == "cwe"
        assert ident["name"] == "CWE-639"
        assert ident["url"].endswith("/639.html")
        assert v["name"].startswith("CWE-639:")
        # id is a stable sha256 hex digest
        assert len(v["id"]) == 64

    def test_no_cwe_fallback_identifier_and_differential(self):
        findings = [{"file": "a.py", "line": 3, "body": "some finding"}]
        report = _writer()._build_report(findings, partial=True)
        assert report["scan"]["partial_scan"] == {"mode": "differential"}
        v = report["vulnerabilities"][0]
        # no cwe -> synthetic bl_finding identifier, name without CWE prefix
        assert v["identifiers"][0]["type"] == "bl_finding"
        assert not v["name"].startswith("CWE-")

    def test_empty_body_defaults_to_finding_name(self):
        report = _writer()._build_report([{"file": "a.py"}])
        assert report["vulnerabilities"][0]["name"] == "Business-logic finding"

    def test_name_is_the_cwe_and_first_sentence(self):
        findings = [
            {
                "cwe": "CWE-639",
                "file": "a.py",
                "body": "Any user can refund any order. Details.",
            },
            {"cwe": "CWE-79", "file": "b.py", "body": "Unescaped name"},
        ]
        names = [
            v["name"] for v in _writer()._build_report(findings)["vulnerabilities"]
        ]
        assert names == [
            "CWE-639: Any user can refund any order.",
            "CWE-79: Unescaped name",
        ]

    def test_name_is_one_plain_line_without_markup(self):
        body = "Line\none\x1b[31m with `code` and [a](http://x) <b>tag</b>"
        name = bl._title_of("639", body)
        assert name == "CWE-639: Line one 31m with code and a(http://x) btag/b"

    def test_an_abbreviation_does_not_end_the_sentence(self):
        body = "Missing check, e.g. on refund, vs. cancel. Details."
        assert bl._title_of("639", body) == (
            "CWE-639: Missing check, e.g. on refund, vs. cancel."
        )

    def test_a_long_name_is_cut_on_a_word_boundary(self):
        name = bl._title_of("639", "word " * 60)
        assert len(name) <= bl._TITLE_MAX
        assert name.endswith(" word\u2026")

    def test_a_first_sentence_that_fits_has_no_ellipsis(self):
        body = "x" * (bl._TITLE_MAX - len("CWE-639: "))
        assert bl._title_of("639", body) == "CWE-639: " + body


# --------------------------------------------------------------------------- #
# BlWriteSastReport._reanchor (async)
# --------------------------------------------------------------------------- #
class TestReanchor:
    def test_relocates_to_excerpt_line(self, monkeypatch):
        content = "class Foo\n  def show\n    Record.find(params[:id])\n  end\nend"

        async def _fake(metadata, action):
            return content

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = [
            {"file": "a.rb", "new_line": 99, "code_excerpt": "Record.find(params[:id])"}
        ]
        counts = asyncio.run(_writer()._verify_anchors(findings))
        assert counts[bl.ANCHOR_CORRECTED] == 1
        assert findings[0]["new_line"] == 3  # relocated to the real line

    def test_skips_findings_without_excerpt_or_file(self):
        findings = [{"file": "a.rb"}, {"code_excerpt": "x"}]
        with patch.object(bl, "_execute_action", new_callable=AsyncMock) as read:
            counts = asyncio.run(_writer()._verify_anchors(findings))
        read.assert_not_called()
        assert counts == {
            bl.ANCHOR_VERIFIED: 0,
            bl.ANCHOR_CORRECTED: 0,
            bl.ANCHOR_UNVERIFIED: 2,
        }
        # Nothing to check against => the anchor is a CLAIM, and says so.
        assert [f["anchor_status"] for f in findings] == [bl.ANCHOR_UNVERIFIED] * 2

    def test_an_excluded_file_is_not_read_and_stays_unverified(self, monkeypatch):
        """The same exclusion rules ``_repair_path`` applies to a listing apply to the claimed file itself."""
        read: list = []

        async def _fake(metadata, action):
            read.append(action)
            return "Record.find(params[:id])"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        writer = BlWriteSastReport(
            metadata={"outbox": object(), "project": {"exclusion_rules": ["secret/**"]}}
        )
        findings = [
            {"file": "secret/a.rb", "new_line": 1, "code_excerpt": "Record.find"}
        ]

        counts = asyncio.run(writer._verify_anchors(findings))

        assert counts[bl.ANCHOR_UNVERIFIED] == 1
        assert findings[0]["anchor_reason"] == bl._ANCHOR_EXCLUDED
        assert read == []

    @pytest.mark.parametrize(
        "path", [".env", ".ssh/id_rsa", "../etc/passwd", "app/../../x.rb"]
    )
    def test_a_denylisted_or_traversal_path_is_not_read(self, monkeypatch, path):
        """Model-written paths go through the always-on denylist and traversal guard, as ReadFile does."""
        read: list = []

        async def _fake(metadata, action):
            read.append(action)
            return "SECRET=1"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = [{"file": path, "new_line": 1, "code_excerpt": "SECRET"}]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts[bl.ANCHOR_UNVERIFIED] == 1
        assert findings[0]["anchor_reason"] == bl._ANCHOR_EXCLUDED
        assert read == []

    def test_a_repaired_path_on_the_denylist_is_not_read(self, monkeypatch):
        """A path repair that lands on a denylisted file does not read it."""
        reads: list = []

        async def _fake(metadata, action):
            if action.HasField("findFiles"):
                return ".ssh/id_rsa\n"
            reads.append(action.runReadFile.filepath)
            return ""

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = [{"file": "id_rsa", "new_line": 1, "code_excerpt": "KEY"}]

        asyncio.run(_writer()._verify_anchors(findings))

        assert reads == ["id_rsa"]
        assert findings[0]["file"] == "id_rsa"
        assert findings[0]["anchor_reason"] == bl._ANCHOR_UNREADABLE

    def test_read_failure_leaves_finding_and_caches_none(self, monkeypatch):
        async def _boom(metadata, action):
            raise RuntimeError("read blew up")

        monkeypatch.setattr(bl, "_execute_action", _boom)
        findings = [
            {"file": "a.rb", "new_line": 5, "code_excerpt": "x"},
            {
                "file": "a.rb",
                "new_line": 6,
                "code_excerpt": "x",
            },  # second: cache hit -> None
        ]
        counts = asyncio.run(_writer()._verify_anchors(findings))
        assert counts[bl.ANCHOR_UNVERIFIED] == 2
        # The line is KEPT (a correct finding is never discarded over a bad
        # coordinate), but it is marked as unchecked.
        assert findings[0]["new_line"] == 5
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED

    def test_loop_level_exception_is_swallowed(self, monkeypatch):
        async def _fake(metadata, action):
            return "irrelevant"

        monkeypatch.setattr(bl, "_execute_action", _fake)

        class _BadDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom in .get")

        # the per-finding try/except must swallow this and keep going
        findings = [_BadDict()]
        counts = asyncio.run(_writer()._verify_anchors(findings))
        assert counts[bl.ANCHOR_VERIFIED] == 0
        # A finding whose verification BLEW UP is unverified, not fine.
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED


# --------------------------------------------------------------------------- #
# BlWriteSastReport._execute (async end-to-end write path)
# --------------------------------------------------------------------------- #
class TestWriteExecute:
    def test_writes_report_and_publishes(self, monkeypatch):
        content = "line1\nRecord.find(params[:id])\nline3"
        actions = {"read": 0, "write": None, "cmd": 0}

        async def _fake(metadata, action):
            if action.HasField("runReadFile"):
                actions["read"] += 1
                return content
            if action.HasField("runWriteFile"):
                actions["write"] = action.runWriteFile
                return "ok"
            actions["cmd"] += 1
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = json.dumps(
            [
                {
                    "cwe": "CWE-639",
                    "file": "a.rb",
                    "new_line": 1,
                    "severity": "critical",
                    "body": "IDOR",
                    "code_excerpt": "Record.find(params[:id])",
                }
            ]
        )
        out = asyncio.run(_writer()._execute(findings))
        assert "Wrote 1 vulnerabilities" in out
        # One file write straight to the pinned path; no command is executed.
        assert actions["write"].filepath == "gl-sast-report.json"
        assert actions["cmd"] == 0
        report = json.loads(actions["write"].contents)
        assert report["vulnerabilities"][0]["severity"] == "Critical"
        # reanchor relocated to line 2
        assert report["vulnerabilities"][0]["location"]["start_line"] == 2

    def test_an_executor_write_error_is_not_reported_as_success(self, monkeypatch):
        async def _fail(metadata, action):
            raise ToolException("Action error: unable to create file")

        monkeypatch.setattr(bl, "_execute_action", _fail)
        with pytest.raises(ToolException, match="unable to create file"):
            asyncio.run(_writer()._execute([{"file": "a.rb", "body": "x"}]))

    def test_an_error_only_in_the_response_text_is_not_success(self, monkeypatch):
        # An executor that puts the failure in the response text, not the
        # error field, must not produce "Wrote N".
        async def _legacy(metadata, action):
            return "Error running tool: file is gitignored: gl-sast-report.json"

        monkeypatch.setattr(bl, "_execute_action", _legacy)
        with pytest.raises(ToolException, match="gl-sast-report.json failed"):
            asyncio.run(_writer()._execute([{"file": "a.rb", "body": "x"}]))

    def test_re_reports_converging_on_one_line_collapse_in_the_report(
        self, monkeypatch
    ):
        # Dedup keyed on the claimed lines (1 and 40: different buckets), so
        # both survive it. Anchor correction moves both to line 2, and only
        # the second pass in _execute can then see them as one.
        content = "line1\nRecord.find(params[:id])\nline3"
        writes = {}

        async def _fake(metadata, action):
            if action.HasField("runReadFile"):
                return content
            writes["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        finding = {"cwe": "CWE-639", "file": "a.rb", "body": "IDOR"}
        excerpt = "Record.find(params[:id])"
        findings = [
            {**finding, "new_line": 1, "code_excerpt": excerpt},
            {**finding, "new_line": 40, "code_excerpt": excerpt},
        ]
        assert len(asyncio.run(_dedup()._execute([findings]))) == 2

        out = asyncio.run(_writer()._execute(findings))

        assert "Wrote 1 vulnerabilities" in out
        vulns = json.loads(writes["contents"])["vulnerabilities"]
        assert [v["location"]["start_line"] for v in vulns] == [2]

    def test_reanchor_exception_does_not_abort_write(self, monkeypatch):
        writes = {}

        async def _fake(metadata, action):
            if action.HasField("runWriteFile"):
                writes["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)

        async def _boom(self, findings):
            raise RuntimeError("reanchor exploded")

        monkeypatch.setattr(BlWriteSastReport, "_verify_anchors", _boom)

        out = asyncio.run(
            _writer()._execute([{"file": "a.rb", "body": "x", "line": 5}])
        )
        assert "Wrote 1 vulnerabilities" in out
        assert "contents" in writes  # report still written despite reanchor failure


class TestAFailedUpstreamStepFailsTheStep:
    """A failed flow step publishes ``None``, and the next step still runs.

    These go through ``ainvoke``, the path the flow's step takes, so a
    ``handle_tool_error`` that turned the raise into returned text would fail
    them.
    """

    @pytest.fixture
    def writes(self, monkeypatch):
        writes = []

        async def _fake(metadata, action):
            if action.HasField("runWriteFile"):
                writes.append(json.loads(action.runWriteFile.contents))
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        return writes

    @pytest.mark.parametrize(
        "findings",
        [None, "Error: x", "null", "42", '[{"cwe": "CWE-284"}] -- then it failed'],
    )
    def test_the_report_refuses_and_writes_nothing(self, writes, findings):
        with pytest.raises(ToolException, match="earlier step most likely failed"):
            asyncio.run(_writer().ainvoke({"findings": findings}))
        assert writes == []

    @pytest.mark.parametrize("findings", [{}, '{"error":"boom"}'])
    def test_the_report_refuses_a_dict(self, writes, findings):
        with pytest.raises(ToolException, match="earlier step most likely failed"):
            asyncio.run(_writer().ainvoke({"findings": findings}))
        assert writes == []

    @pytest.mark.parametrize("findings", [[], "[]"])
    def test_an_empty_list_still_writes_a_zero_finding_report(self, writes, findings):
        out = asyncio.run(_writer().ainvoke({"findings": findings}))
        assert "Wrote 0 vulnerabilities" in out
        assert writes[0]["vulnerabilities"] == []
        assert writes[0]["scan"]["status"] == "success"

    @pytest.mark.parametrize(
        "text", ["Error: tool failed on item [0]", 'Error: {"detail": "timeout"}']
    )
    def test_a_bracket_inside_an_error_string_is_not_parsed(self, writes, text):
        with pytest.raises(ToolException, match="is not a findings list"):
            asyncio.run(_dedup().ainvoke({"batches": text}))
        with pytest.raises(ToolException, match="is not a findings list"):
            asyncio.run(_writer().ainvoke({"findings": text}))
        assert writes == []

    def test_a_failed_write_raises_through_ainvoke(self, monkeypatch):
        async def _fail(metadata, action):
            if action.HasField("runWriteFile"):
                raise ToolException("Action error: unable to create file")
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fail)
        with pytest.raises(ToolException, match="unable to create file"):
            asyncio.run(_writer().ainvoke({"findings": []}))


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


class TestVulnerabilityIdDistinctness:
    def _ids(self, findings):
        report = _writer()._build_report(findings)
        return [v["id"] for v in report["vulnerabilities"]]

    def test_sites_sharing_a_statement_get_distinct_ids(self):
        # Same CWE, same file, identical excerpt -> the fingerprint alone would
        # give all three ONE id and the ingesting side would see a single vuln.
        shared = "return self.queryset"
        ids = self._ids([_site(i, 100 + i * 50, excerpt=shared) for i in range(3)])
        assert len(set(ids)) == 3

    def test_excerptless_sites_sharing_a_body_get_distinct_ids(self):
        # No excerpt -> fingerprint falls back to the body prefix, which N
        # sibling-pass findings about one missing control also share.
        ids = self._ids(
            [
                {
                    "cwe": "CWE-862",
                    "file": "svc/handlers.py",
                    "new_line": ln,
                    "body": "no role check on this action",
                }
                for ln in (10, 90, 300)
            ]
        )
        assert len(set(ids)) == 3

    def test_unique_finding_id_stays_line_independent(self):
        # The anti-churn property: a lone finding's id must not move when the
        # (frequently wrong) line number does.
        first = self._ids([_site(0, 100)])
        moved = self._ids([_site(0, 512)])
        assert first == moved

    def test_same_site_reported_twice_keeps_one_id(self):
        # Identical statement AND identical line -> genuinely one vuln.
        ids = self._ids([_site(0, 100), _site(0, 100)])
        assert len(set(ids)) == 1

    def test_a_relabelled_finding_keeps_its_id(self):
        # The customer-visible property the CWE-free key exists for: the same site,
        # quoted identically, re-labelled between runs must keep ONE id; with the
        # CWE in the key it would read downstream as one vuln resolved plus a new
        # one found.
        run1 = self._ids([_site(0, 100, cwe="CWE-639")])
        run2 = self._ids([_site(0, 100, cwe="CWE-862")])
        assert run1 == run2

    def test_distinct_sites_are_told_apart_even_when_relabelled(self):
        # The guard the CWE-free id makes load-bearing. Two genuinely different
        # sites quoting the same statement collide on (file, fingerprint), so only
        # the line can tell them apart. If _shared_fingerprints still keyed on the
        # CWE, their differing CWEs would split them into two singleton keys ->
        # NEITHER flagged as shared, NEITHER given a line, and both hashing to the
        # identical id.
        shared = "return self.queryset"
        ids = self._ids(
            [
                _site(0, 100, cwe="CWE-639", excerpt=shared),
                _site(1, 400, cwe="CWE-862", excerpt=shared),
            ]
        )
        assert len(set(ids)) == 2

    def test_cwe_is_still_reported_though_it_is_not_identity(self):
        # Out of the id is not out of the report: the CWE stays in `identifiers`,
        # which is what the Vulnerability Report displays and filters on.
        report = _writer()._build_report([_site(0, 100, cwe="CWE-862")])
        ident = report["vulnerabilities"][0]["identifiers"][0]
        assert ident["type"] == "cwe"
        assert ident["name"] == "CWE-862"
        assert ident["value"] == "862"
        assert ident["url"].endswith("/862.html")


class _LogRecorder:
    """Captures the structlog calls bl_report makes, so the audit trail can be asserted on."""

    def __init__(self):
        self.records = []

    def info(self, event, **kw):
        self.records.append((event, kw))

    def of(self, event):
        return [kw for name, kw in self.records if name == event]


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


class TestTriageVerdictAudit:
    """Every adjudication decision must be recorded, and a DROP must still drop.

    The adjudicator returns the finding annotated with verdict/clause/evidence for BOTH verdicts; this tool logs every
    decision and removes only the DROPs.
    """

    def test_audit_fields_do_not_change_dedup_or_the_report(self, monkeypatch):
        # Two reports of the SAME site that the adjudicator annotated
        # differently must still collapse to one: the audit fields are not part
        # of the dedup key, so instrumentation cannot inflate the finding count.
        rec = _LogRecorder()
        monkeypatch.setattr(dedup_mod, "_log", rec)
        out = asyncio.run(
            _dedup()._execute(
                [
                    [
                        _finding(verdict="KEEP", clause="KEEP-cross-principal"),
                        _finding(verdict="KEEP", clause="KEEP-incorrect-guard"),
                    ]
                ]
            )
        )
        assert len(out) == 1
        report = _writer()._build_report(out)
        vuln = report["vulnerabilities"][0]
        # The vulnerability object must not grow audit fields: it carries only
        # the fixed set below. `details` and `raw_source_code_extract` are
        # schema-defined, and they are where the audit travels:
        # the anchor state, the quote it was checked against, and the triage
        # clause + verdict, all inside `details`. What stays out is the RAW
        # model-authored keys: `verdict` / `clause` /
        # `evidence` / `triage_evidence` are never emitted under their own
        # names, at the top level or as detail keys, so the schema does not
        # grow and an ingesting side sees only defined properties.
        assert set(vuln) == {
            "id",
            "name",
            "description",
            "severity",
            "location",
            "identifiers",
            "details",
            "raw_source_code_extract",
        }
        assert not {"verdict", "clause", "evidence", "triage_evidence"} & set(vuln)
        assert not {"verdict", "clause", "evidence", "triage_evidence"} & set(
            vuln["details"]
        )


# --------------------------------------------------------------------------- #
# SCAN COVERAGE DISCLOSURE — the report says what the scan actually opened
# --------------------------------------------------------------------------- #
# Caps truncate a run (discovery dispatches only some review units, triage only
# some candidates). The report must say so, or a partial scan is
# indistinguishable from a complete one to whoever reads it.
_REVIEW_COVERAGE = {
    "emitted": 1339,
    "dispatched": 150,
    "completed": 150,
    "unit_noun": "review units",
    "summary": (
        "map_reviews reviewed 150 of 1339 review units (no scan_effort set, "
        "fan-out cap 150); the 150 reviewed review units name 519 distinct "
        "files, out of 1341 distinct files named across all 1339 emitted "
        "review units (the repository's total file count is not visible at "
        "this stage)."
    ),
    "truncation": (
        "1189 of 1339 review units were not reviewed because the fan-out cap "
        "(150) was reached."
    ),
}

_TRIAGE_COVERAGE = {
    "emitted": 432,
    "dispatched": 150,
    "completed": 150,
    "unit_noun": "findings",
    "summary": "triage reviewed 150 of 432 findings (fan-out cap 150).",
    "truncation": (
        "282 of 432 findings were not reviewed because the fan-out cap (150) "
        "was reached."
    ),
}

_COMPLETE_COVERAGE = {
    "emitted": 4,
    "dispatched": 4,
    "completed": 4,
    "unit_noun": "review units",
    "summary": "sibling_scan reviewed all 4 review units it was given.",
    "truncation": None,
}

# The schema's own vocabulary for scan.messages[].level, quoted so a widened
# enum here has to be a deliberate edit rather than a typo that ships.
_SCHEMA_LEVELS = {"info", "warn", "fatal"}


# --------------------------------------------------------------------------- #
# The triage clause travels WITH the surviving finding
#
# "Did this KEEP arm ever fire?" must be answerable with a grep over the report.
# --------------------------------------------------------------------------- #
class TestTriageClauseReachesTheReport:
    def test_the_clause_and_verdict_land_in_the_vulnerability_details(self):
        report = _writer()._build_report(
            [_finding(verdict="KEEP", clause="KEEP-incorrect-guard")]
        )
        details = report["vulnerabilities"][0]["details"]

        # THE assertion: the arm that let this finding survive is greppable in
        # the artifact a reader is handed.
        assert details[bl.TRIAGE_CLAUSE_DETAIL_KEY]["value"] == "KEEP-incorrect-guard"
        assert details[bl.TRIAGE_VERDICT_DETAIL_KEY]["value"] == "KEEP"
        # ...and the anchor disclosure it shares `details` with is untouched.
        assert bl.ANCHOR_DETAIL_KEY in details

    def test_the_clause_survives_the_real_triage_path_end_to_end(self):
        # Not a hand-built dict: the same list the adjudicator returns, through
        # the audit + dedup layer that removes the DROPs, into the report. This
        # is what proves the field is actually available where the report is
        # assembled rather than only in the log.
        out = asyncio.run(
            _dedup()._execute(
                [
                    [
                        _finding(
                            file="keep.py",
                            verdict="KEEP",
                            clause="KEEP-stale-authorization-state",
                        ),
                        _finding(file="drop.py", verdict="DROP", clause="DROP-2"),
                    ]
                ]
            )
        )
        report = _writer()._build_report(out)

        assert [v["location"]["file"] for v in report["vulnerabilities"]] == ["keep.py"]
        assert (
            report["vulnerabilities"][0]["details"][bl.TRIAGE_CLAUSE_DETAIL_KEY][
                "value"
            ]
            == "KEEP-stale-authorization-state"
        )

    def test_an_unadjudicated_finding_says_so_rather_than_going_silent(self):
        # A finding that never went through triage carries no annotation. The
        # key is still emitted: absence would read as "adjudicated and fine".
        report = _writer()._build_report([_finding()])
        details = report["vulnerabilities"][0]["details"]

        assert bl.TRIAGE_VERDICT_DETAIL_KEY not in details
        assert "no triage annotation" in details[bl.TRIAGE_CLAUSE_DETAIL_KEY]["value"]

    def test_a_verdict_with_no_clause_is_not_dressed_up_as_one(self):
        # Three distinct states, three distinct strings: annotated, never
        # annotated, and annotated-without-a-reason. Collapsing the third into
        # either of the others invents a criterion the adjudicator never named.
        report = _writer()._build_report([_finding(verdict="KEEP")])
        details = report["vulnerabilities"][0]["details"]

        assert details[bl.TRIAGE_VERDICT_DETAIL_KEY]["value"] == "KEEP"
        assert "named no clause" in details[bl.TRIAGE_CLAUSE_DETAIL_KEY]["value"]

    def test_the_clause_is_bounded_and_schema_shaped(self):
        # The clause is LLM output. A 500-finding report must not be able to
        # grow without bound through it, and every detail entry still has to
        # satisfy the schema's named_field + text detail type.
        report = _writer()._build_report([_finding(verdict="KEEP", clause="K" * 500)])
        details = report["vulnerabilities"][0]["details"]

        assert len(details[bl.TRIAGE_CLAUSE_DETAIL_KEY]["value"]) == TRIAGE_CLAUSE_MAX
        for detail in details.values():
            assert set(detail) == {"name", "type", "value"}
            assert isinstance(detail["name"], str) and detail["name"]
            assert detail["type"] == "text"
            assert isinstance(detail["value"], str) and detail["value"]


def _messages(report: dict) -> list[dict]:
    """The COVERAGE messages only.

    ``scan.messages`` also carries the anchor-verification disclosure (see
    ``TestAnchorVerificationIsDisclosedInTheReport``); the coverage tests below
    are about coverage, so that one is filtered out here rather than woven into
    nine exact-list assertions that are not about it. It is not hidden: it has
    its own tests, and ``test_every_message_is_shaped_the_way_the_schema_defines``
    reads the unfiltered array.
    """
    return [
        m
        for m in report["scan"]["messages"]
        if not m["value"].startswith(bl.ANCHOR_MESSAGE_PREFIX)
    ]


def _anchor_message(report: dict) -> dict:
    """The single anchor-verification entry in ``scan.messages``."""
    found = [
        m
        for m in report["scan"]["messages"]
        if m["value"].startswith(bl.ANCHOR_MESSAGE_PREFIX)
    ]
    assert len(found) == 1, f"expected exactly one anchor message, got {found}"
    return found[0]


class TestScanCoverageIsDisclosedInTheReport:
    """The report must not read as complete when it is not.

    Where it goes is decided by the SAST report schema this tool declares:
    ``scan.messages`` is an array of ``{level, value}`` with ``level``
    in info/warn/fatal, described by the schema as "Communication intended for
    the initiator of a scan." That is the one field in the format built to
    carry this, which is why nothing new is invented alongside it.
    """

    def test_every_message_is_shaped_the_way_the_schema_defines(self):
        report = _writer()._build_report(
            [{"file": "a.py"}],
            [_REVIEW_COVERAGE, _COMPLETE_COVERAGE, _TRIAGE_COVERAGE],
        )
        # Deliberately the UNFILTERED array: every entry the tool emits must be
        # schema-shaped, the anchor-verification one included.
        messages = report["scan"]["messages"]

        assert messages, "the report disclosed no coverage at all"
        for message in messages:
            assert set(message) == {"level", "value"}
            assert message["level"] in _SCHEMA_LEVELS
            # minLength: 1 in the schema.
            assert isinstance(message["value"], str) and message["value"]

    def test_a_capped_stage_states_the_files_and_the_truncation(self):
        report = _writer()._build_report([{"file": "a.py"}], [_REVIEW_COVERAGE])
        levels = [(m["level"], m["value"]) for m in _messages(report)]

        # What was covered is INFO; a cap that actually bound is a WARNing,
        # because it means findings were discarded, not merely counted.
        assert levels == [
            ("info", _REVIEW_COVERAGE["summary"]),
            ("warn", _REVIEW_COVERAGE["truncation"]),
        ]
        # The numbers a reader needs: reviewed vs eligible, both halves.
        assert "150 of 1339 review units" in levels[0][1]
        assert "519 distinct files, out of 1341 distinct files" in levels[0][1]

    def test_a_complete_stage_gets_no_truncation_warning(self):
        report = _writer()._build_report([{"file": "a.py"}], [_COMPLETE_COVERAGE])

        assert _messages(report) == [
            {"level": "info", "value": _COMPLETE_COVERAGE["summary"]}
        ]

    def test_findings_truncated_by_triage_are_disclosed_too(self):
        # The SECOND cap. Discovery coverage alone would still let a report of
        # 150 adjudicated findings out of 432 candidates read as the whole set.
        report = _writer()._build_report([{"file": "a.py"}], [_TRIAGE_COVERAGE])
        warnings = [m["value"] for m in _messages(report) if m["level"] == "warn"]

        assert warnings == [_TRIAGE_COVERAGE["truncation"]]
        assert "282 of 432 findings were not reviewed" in warnings[0]

    def test_a_run_that_reports_no_coverage_says_so(self):
        # THE failure mode this exists to close: silence is what made a partial
        # scan look complete, so an absent record must be stated, not skipped.
        report = _writer()._build_report([{"file": "a.py"}], [None, None])
        (message,) = _messages(report)

        assert message["level"] == "warn"
        assert "not reported" in message["value"]
        assert "must not be read as covering the whole repository" in message["value"]

    def test_a_stage_record_that_arrives_as_json_text_still_counts(self):
        # Context values reach a tool as whatever the graph put there; a record
        # that survived a JSON round-trip must not silently disappear.
        report = _writer()._build_report(
            [{"file": "a.py"}], [json.dumps(_COMPLETE_COVERAGE)]
        )

        assert _messages(report) == [
            {"level": "info", "value": _COMPLETE_COVERAGE["summary"]}
        ]

    def test_unreadable_stage_records_are_ignored_not_fatal(self):
        # A malformed record must never break report generation -- it just
        # leaves that stage undisclosed, and with NO stage readable the report
        # falls back to saying coverage is unknown.
        assert bl._coverage_record("{not json") is None
        assert bl._coverage_record(7) is None
        report = _writer()._build_report([{"file": "a.py"}], ["{not json", 7])

        assert _messages(report) == [
            {"level": "warn", "value": bl._COVERAGE_UNAVAILABLE}
        ]

    def test_a_record_with_no_sentences_contributes_nothing(self):
        report = _writer()._build_report(
            [{"file": "a.py"}], [{"unit_noun": "review units", "emitted": 3}]
        )

        assert _messages(report) == [
            {"level": "warn", "value": bl._COVERAGE_UNAVAILABLE}
        ]

    def test_execute_carries_all_three_stages_into_the_written_file(self, monkeypatch):
        # The end-to-end shape: the three coverage inputs the flow wires in
        # must land in the file that is actually published.
        written = {}

        async def _fake(metadata, action):
            if action.HasField("runWriteFile"):
                written["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)

        asyncio.run(
            _writer()._execute(
                [{"file": "a.rb", "body": "x", "line": 5}],
                review_coverage=_REVIEW_COVERAGE,
                sibling_coverage=_COMPLETE_COVERAGE,
                triage_coverage=_TRIAGE_COVERAGE,
            )
        )
        messages = _messages(json.loads(written["contents"]))

        assert [m["value"] for m in messages] == [
            _REVIEW_COVERAGE["summary"],
            _REVIEW_COVERAGE["truncation"],
            _COMPLETE_COVERAGE["summary"],
            _TRIAGE_COVERAGE["summary"],
            _TRIAGE_COVERAGE["truncation"],
        ]

    def test_the_tool_declares_an_argument_for_every_stage(self):
        fields = bl.BlWriteSastReportInput.model_fields

        for name in ("review_coverage", "sibling_coverage", "triage_coverage"):
            assert name in fields, name
            # Optional, so a run missing a stage still writes a report.
            assert fields[name].default is None


# A stage's coverage record as `bl_collect_and_flatten` emits it: counts plus
# the `summary` and `loss` sentences (no `truncation` key when no cap bound).
_LOSSY_COVERAGE = {
    "emitted": 40,
    "dispatched": 40,
    "completed": 30,
    "errored": 10,
    "unit_noun": "review units",
    "summary": "Reviewed 26 of 40 review units.",
    "loss": (
        "14 of 40 review units were DISCARDED without their findings being "
        "counted (4 completed but their findings could not be read; 10 errored)."
    ),
}


class TestDiscardedUnitsAreDisclosedInTheReport:
    """A stage that produced findings and then threw them away because it could not read them must say so in the
    artifact the reader is handed.

    Units can all be recorded ``completed`` while some are discarded at the parser, so without this a report identical
    to a clean one comes out of a scan missing part of its work.
    """

    def test_a_lossy_stage_warns_even_though_no_cap_bound(self):
        report = _writer()._build_report([{"file": "a.py"}], [_LOSSY_COVERAGE])
        levels = [(m["level"], m["value"]) for m in _messages(report)]

        assert levels == [
            ("info", _LOSSY_COVERAGE["summary"]),
            ("warn", _LOSSY_COVERAGE["loss"]),
        ]
        assert "14 of 40 review units" in levels[1][1]

    def test_a_stage_that_lost_nothing_adds_no_warning(self):
        clean = {**_LOSSY_COVERAGE, "errored": 0, "loss": None}
        report = _writer()._build_report([{"file": "a.py"}], [clean])

        assert _messages(report) == [{"level": "info", "value": clean["summary"]}]

    def test_a_capped_and_lossy_stage_reports_both_losses_apart(self):
        # A cap that bound is a CHOICE; a discarded unit is a DEFECT. Collapsing
        # them into one message would let the second hide behind the first.
        both = {**_LOSSY_COVERAGE, "truncation": _REVIEW_COVERAGE["truncation"]}
        report = _writer()._build_report([{"file": "a.py"}], [both])
        warnings = [m["value"] for m in _messages(report) if m["level"] == "warn"]

        assert warnings == [_REVIEW_COVERAGE["truncation"], _LOSSY_COVERAGE["loss"]]

    def test_every_loss_message_still_matches_the_schema(self):
        report = _writer()._build_report([{"file": "a.py"}], [_LOSSY_COVERAGE])

        for message in _messages(report):
            assert set(message) == {"level", "value"}
            assert message["level"] in _SCHEMA_LEVELS
            assert isinstance(message["value"], str) and message["value"]


# --------------------------------------------------------------------------- #
# ANCHOR VERIFICATION
#
# These tests pin the three outcomes as DISTINCT, VISIBLE states. The collapse in
# either direction is the defect: an unverifiable anchor read as "fine" hides a
# real mislocation behind a bad coordinate, and an unverifiable anchor dropped
# discards a correct security finding.
# --------------------------------------------------------------------------- #

# A file with two functions, so a wrong anchor lands in the WRONG ONE.
_TWO_FUNCS = "\n".join(
    [
        "package container",  # 1
        "",  # 2
        "func PostBlobsUploads(ctx *Context) {",  # 3
        "\tuploader.Create(ctx)",  # 4
        "\treturn",  # 5
        "}",  # 6
        "",  # 7
        "func PutBlobsUpload(ctx *Context) {",  # 8
        '\tupload := uploader.Get(ctx.Params["uuid"])',  # 9
        "\tupload.Commit()",  # 10
        "}",  # 11
    ]
)


def _reader(content):
    async def _fake(metadata, action):
        if action.HasField("runReadFile"):
            return content
        return "ok"

    return _fake


class TestAnchorVerificationOutcomes:
    def test_a_quote_found_at_the_claimed_line_is_verified(self, monkeypatch):
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [
            {
                "file": "container.go",
                "new_line": 10,
                "code_excerpt": "upload.Commit()",
            }
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts == {
            bl.ANCHOR_VERIFIED: 1,
            bl.ANCHOR_CORRECTED: 0,
            bl.ANCHOR_UNVERIFIED: 0,
        }
        assert findings[0]["anchor_status"] == bl.ANCHOR_VERIFIED
        assert findings[0]["new_line"] == 10  # emitted as-is
        assert "anchor_claimed_line" not in findings[0]

    def test_a_quote_found_elsewhere_moves_the_line_and_keeps_the_claim(
        self, monkeypatch
    ):
        """THE CASE THAT PAYS.

        The claim (line 4) is in ``PostBlobsUploads``; the quoted statement is
        in ``PutBlobsUpload`` at line 9. The finding is CORRECT and its anchor
        is RECOVERABLE, so the line is moved to where the quote actually is -
        and the line the model claimed is kept, because a correction that erases
        what it corrected cannot be audited.
        """
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [
            {
                "file": "container.go",
                "new_line": 4,
                "code_excerpt": 'upload := uploader.Get(ctx.Params["uuid"])',
            }
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts == {
            bl.ANCHOR_VERIFIED: 0,
            bl.ANCHOR_CORRECTED: 1,
            bl.ANCHOR_UNVERIFIED: 0,
        }
        assert findings[0]["anchor_status"] == bl.ANCHOR_CORRECTED
        assert findings[0]["new_line"] == 9
        assert findings[0]["anchor_claimed_line"] == 4

    def test_a_quote_that_is_nowhere_in_the_file_is_unverified_not_dropped(
        self, monkeypatch
    ):
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [
            {
                "file": "container.go",
                "new_line": 7,
                "code_excerpt": "if !ctx.User.IsAdmin() { return }",
            }
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts == {
            bl.ANCHOR_VERIFIED: 0,
            bl.ANCHOR_CORRECTED: 0,
            bl.ANCHOR_UNVERIFIED: 1,
        }
        # Not dropped - a sound security claim must survive a bad coordinate.
        assert len(findings) == 1
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED
        assert findings[0]["new_line"] == 7  # the model's claim, kept as such
        assert "not appear" in findings[0]["anchor_reason"]

    def test_a_finding_with_no_quote_is_unverified(self, monkeypatch):
        """A pass whose output carries NO excerpt produces structurally uncheckable anchors.

        That is a fact about the run and belongs in the data, not an absence that reads as agreement.
        """
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [{"file": "container.go", "new_line": 5, "cwe": "CWE-639"}]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts[bl.ANCHOR_UNVERIFIED] == 1
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED
        assert findings[0]["anchor_reason"] == bl._ANCHOR_NO_EXCERPT

    def test_a_finding_with_no_file_is_unverified(self, monkeypatch):
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [{"new_line": 5, "code_excerpt": "upload.Commit()"}]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts[bl.ANCHOR_UNVERIFIED] == 1
        assert "no file path" in findings[0]["anchor_reason"]

    def test_a_finding_that_claimed_no_line_is_not_reported_as_a_corrected_claim(
        self, monkeypatch
    ):
        """0 and "never stated" are different facts.

        ``_line_of`` returns 0 for a finding that carried no line at all. The
        anchor still moves (to where the quote is), but recording
        ``anchor_claimed_line: 0`` would assert the model claimed line 0, which
        it did not.
        """
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [{"file": "container.go", "code_excerpt": "upload.Commit()"}]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts[bl.ANCHOR_CORRECTED] == 1
        assert findings[0]["new_line"] == 10
        assert "anchor_claimed_line" not in findings[0]
        assert "no line number" in findings[0]["anchor_reason"]

    def test_all_three_outcomes_are_counted_separately_in_one_pass(self, monkeypatch):
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [
            {"file": "c.go", "new_line": 10, "code_excerpt": "upload.Commit()"},
            {"file": "c.go", "new_line": 4, "code_excerpt": "upload.Commit()"},
            {"file": "c.go", "new_line": 7, "code_excerpt": "nothing::like_this()"},
            {"file": "c.go", "new_line": 5},
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts == {
            bl.ANCHOR_VERIFIED: 1,
            bl.ANCHOR_CORRECTED: 1,
            bl.ANCHOR_UNVERIFIED: 2,
        }
        # Every finding carries a state; none is left unlabelled.
        assert all("anchor_status" in f for f in findings)


class TestAnchorVerificationIsDisclosedInTheReport:
    """A state the report does not carry is a state the scorer cannot read.

    The SAST schema this tool declares already has the homes for this:
    ``vulnerability.details`` (a named-list of typed fields) and
    ``vulnerability.raw_source_code_extract`` ("an unsanitized excerpt of the
    affected source code"). Nothing new is invented and no version is bumped.
    """

    def test_every_vulnerability_carries_an_explicit_anchor_state(self):
        findings = [
            {"file": "a.go", "new_line": 3, "anchor_status": bl.ANCHOR_VERIFIED},
            {"file": "b.go", "new_line": 4, "anchor_status": bl.ANCHOR_UNVERIFIED},
        ]
        report = _writer()._build_report(findings)

        states = [
            v["details"][bl.ANCHOR_DETAIL_KEY]["value"]
            for v in report["vulnerabilities"]
        ]
        assert states == [bl.ANCHOR_VERIFIED, bl.ANCHOR_UNVERIFIED]

    def test_an_unverified_anchor_is_distinguishable_from_a_verified_one(self):
        verified = _writer()._build_report(
            [{"file": "a.go", "new_line": 3, "anchor_status": bl.ANCHOR_VERIFIED}],
        )["vulnerabilities"][0]
        unverified = _writer()._build_report(
            [
                {
                    "file": "a.go",
                    "new_line": 3,
                    "anchor_status": bl.ANCHOR_UNVERIFIED,
                    "anchor_reason": "the quoted code does not appear in the file",
                }
            ],
        )["vulnerabilities"][0]

        # Same file, same line, same everything the scorer binds on...
        assert verified["location"] == unverified["location"]
        # ...and yet they are not the same claim, and the report says which.
        assert (
            verified["details"][bl.ANCHOR_DETAIL_KEY]["value"]
            != unverified["details"][bl.ANCHOR_DETAIL_KEY]["value"]
        )
        assert (
            "does not appear"
            in unverified["details"][bl.ANCHOR_REASON_DETAIL_KEY]["value"]
        )

    def test_a_corrected_anchor_reports_the_line_originally_claimed(self):
        report = _writer()._build_report(
            [
                {
                    "file": "container.go",
                    "new_line": 9,
                    "anchor_status": bl.ANCHOR_CORRECTED,
                    "anchor_claimed_line": 4,
                }
            ],
        )
        v = report["vulnerabilities"][0]

        assert v["location"]["start_line"] == 9
        assert v["details"][bl.ANCHOR_DETAIL_KEY]["value"] == bl.ANCHOR_CORRECTED
        assert v["details"][bl.ANCHOR_CLAIMED_DETAIL_KEY]["value"] == "4"

    def test_a_finding_verification_never_saw_is_not_reported_as_verified(self):
        """``_verify_anchors`` is best-effort and ``_execute`` swallows its failure, so a report CAN be built from
        unannotated findings.

        Absent must not read as checked - that is the same collapse, arrived at
        from the other side.
        """
        report = _writer()._build_report([{"file": "a.go", "new_line": 3}])
        v = report["vulnerabilities"][0]

        assert v["details"][bl.ANCHOR_DETAIL_KEY]["value"] == bl.ANCHOR_UNVERIFIED

    def test_the_verbatim_quote_travels_with_the_finding(self):
        report = _writer()._build_report(
            [
                {
                    "file": "a.go",
                    "new_line": 3,
                    "code_excerpt": "upload.Commit()",
                    "anchor_status": bl.ANCHOR_VERIFIED,
                }
            ],
        )

        # The evidence the check was made against is auditable downstream.
        assert report["vulnerabilities"][0]["raw_source_code_extract"] == (
            "upload.Commit()"
        )

    def test_no_quote_means_no_empty_extract_field(self):
        report = _writer()._build_report([{"file": "a.go", "new_line": 3}])

        assert "raw_source_code_extract" not in report["vulnerabilities"][0]

    def test_the_anchor_details_are_shaped_the_way_the_schema_defines(self):
        report = _writer()._build_report(
            [
                {
                    "file": "a.go",
                    "new_line": 3,
                    "anchor_status": bl.ANCHOR_CORRECTED,
                    "anchor_claimed_line": 1,
                    "anchor_reason": "why",
                }
            ],
        )

        for detail in report["vulnerabilities"][0]["details"].values():
            # named_field requires a non-empty `name`; detail_type/text requires
            # `type == "text"` and a string `value`.
            assert set(detail) == {"name", "type", "value"}
            assert isinstance(detail["name"], str) and detail["name"]
            assert detail["type"] == "text"
            assert isinstance(detail["value"], str) and detail["value"]

    def test_the_anchor_state_does_not_change_a_finding_id(self):
        """Comparability guard.

        The vulnerability ``id`` is the identity downstream joins on. Anchor
        verification is disclosure and must not move IDs.
        """
        bare = _writer()._build_report(
            [{"cwe": "CWE-862", "file": "a.go", "code_excerpt": "x()"}]
        )["vulnerabilities"][0]["id"]
        annotated = _writer()._build_report(
            [
                {
                    "cwe": "CWE-862",
                    "file": "a.go",
                    "code_excerpt": "x()",
                    "anchor_status": bl.ANCHOR_UNVERIFIED,
                    "anchor_reason": "whatever",
                }
            ],
        )["vulnerabilities"][0]["id"]

        assert bare == annotated

    def test_a_clean_run_says_so_at_info_level(self):
        report = _writer()._build_report(
            [{"file": "a.go", "new_line": 3, "anchor_status": bl.ANCHOR_VERIFIED}],
        )
        message = _anchor_message(report)

        assert message["level"] == "info"
        assert "1 verified" in message["value"]

    def test_any_unverified_anchor_makes_the_disclosure_a_warning(self):
        report = _writer()._build_report(
            [
                {"file": "a.go", "new_line": 3, "anchor_status": bl.ANCHOR_VERIFIED},
                {"file": "b.go", "new_line": 4, "anchor_status": bl.ANCHOR_CORRECTED},
                {"file": "c.go", "new_line": 5, "anchor_status": bl.ANCHOR_UNVERIFIED},
            ],
        )
        message = _anchor_message(report)

        assert message["level"] == "warn"
        assert "1 verified" in message["value"]
        assert "1 corrected" in message["value"]
        assert "1 could not be verified" in message["value"]

    def test_a_report_with_no_findings_still_states_the_anchor_position(self):
        report = _writer()._build_report([])
        message = _anchor_message(report)

        assert message["level"] == "info"
        assert "0 findings" in message["value"]


class TestAnchorVerificationEndToEnd:
    def test_execute_emits_all_three_states_into_the_written_report(self, monkeypatch):
        """Each anchor outcome is distinguishable in the written report."""
        writes = {}

        async def _fake(metadata, action):
            if action.HasField("runReadFile"):
                return _TWO_FUNCS
            if action.HasField("runWriteFile"):
                writes["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = json.dumps(
            [
                {
                    "cwe": "CWE-862",
                    "file": "container.go",
                    "new_line": 10,
                    "body": "missing authz on commit",
                    "code_excerpt": "upload.Commit()",
                },
                {
                    "cwe": "CWE-863",
                    "file": "container.go",
                    "new_line": 4,
                    "body": "uuid is not owner-checked",
                    "code_excerpt": 'upload := uploader.Get(ctx.Params["uuid"])',
                },
                {
                    "cwe": "CWE-284",
                    "file": "container.go",
                    "new_line": 7,
                    "body": "wrong role checked",
                    "code_excerpt": "if !ctx.User.IsAdmin() { return }",
                },
                {
                    "cwe": "CWE-639",
                    "file": "container.go",
                    "new_line": 5,
                    "body": "IDOR, sibling pass emits no excerpt",
                },
            ]
        )

        out = asyncio.run(_writer()._execute(findings))

        assert "Wrote 4 vulnerabilities" in out
        report = json.loads(writes["contents"])
        got = [
            (
                v["location"]["start_line"],
                v["details"][bl.ANCHOR_DETAIL_KEY]["value"],
            )
            for v in report["vulnerabilities"]
        ]
        assert got == [
            (10, bl.ANCHOR_VERIFIED),
            (9, bl.ANCHOR_CORRECTED),  # relocated out of the wrong function
            (7, bl.ANCHOR_UNVERIFIED),  # kept, but flagged as a claim
            (5, bl.ANCHOR_UNVERIFIED),  # kept, but flagged as a claim
        ]
        assert _anchor_message(report)["level"] == "warn"

    def test_the_outcome_mix_is_logged(self, monkeypatch):
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = json.dumps(
            [
                {
                    "file": "container.go",
                    "new_line": 10,
                    "code_excerpt": "upload.Commit()",
                }
            ]
        )

        with capture_logs() as logs:
            asyncio.run(_writer()._execute(findings))

        entry = next(
            log for log in logs if log["event"] == "bl_write_sast_report input"
        )
        assert entry["anchors_verified"] == 1
        assert entry["anchors_corrected"] == 0
        assert entry["anchors_unverified"] == 0


class TestPostReanchorCollapse:
    """Dedup runs BEFORE re-anchoring, so N re-reports of one site never collapse.

    The exact-dedup key is ``(cwe, file, line//4, excerpt)`` and the line it sees is
    the one the MODEL claimed, so re-reports claiming different lines land in
    different buckets and survive dedup. ``_verify_anchors`` then corrects them all
    to the one real line.

    The report collapses again AFTER the anchors are corrected. It must not disturb
    the opposite case: N genuinely copy-pasted sites re-anchor to N different lines
    and have to survive.
    """

    @staticmethod
    def _writer_with(content):
        async def _fake(metadata, action):
            return content

        return _fake

    def test_reports_of_one_site_collapse_after_reanchor(self, monkeypatch):
        # ONE occurrence of the statement in the file, at line 4.
        content = "a\nb\nc\ntoken = params[:token]\ne\nf\ng\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))

        # Six re-reports of that one site, each guessing a different line.
        findings = [
            {
                "cwe": "CWE-862",
                "file": "oauth.rb",
                "new_line": claimed,
                "code_excerpt": "token = params[:token]",
                "body": "revoke accepts a token without client authentication",
            }
            for claimed in (196, 222, 244, 270, 280, 379)
        ]
        counts = asyncio.run(_writer()._verify_anchors(findings))
        assert counts[bl.ANCHOR_CORRECTED] == 6
        assert {f["new_line"] for f in findings} == {4}, (
            "all six should re-anchor to one line"
        )

        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 1, (
            f"six reports of one site should collapse to 1, got {len(out)}"
        )

    def test_copy_pasted_sites_still_survive(self, monkeypatch):
        # FIVE real occurrences of the same statement, 10 lines apart.
        lines = []
        for i in range(5):
            lines.extend(["pad"] * 9)
            lines.append("return self.queryset")
        content = "\n".join(lines) + "\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))

        # One finding per real site, each hinting near its own occurrence.
        findings = [
            {
                "cwe": "CWE-639",
                "file": "views.py",
                "new_line": 10 + i * 10,
                "code_excerpt": "return self.queryset",
                "body": f"handler {i} misses its owner check",
            }
            for i in range(5)
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 5, (
            f"five distinct copy-pasted sites must survive, got {len(out)}"
        )

    def test_distinct_statements_in_one_bucket_survive(self, monkeypatch):
        # Two DIFFERENT statements that re-anchor into the same line bucket.
        content = "activate(user)\ndeactivate(user)\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-862",
                "file": "admin.ex",
                "new_line": 900,
                "code_excerpt": "activate(user)",
                "body": "activate misses a role check",
            },
            {
                "cwe": "CWE-862",
                "file": "admin.ex",
                "new_line": 901,
                "code_excerpt": "deactivate(user)",
                "body": "deactivate misses a role check",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        # lines 1 and 2 -> same bucket (1//4 == 2//4 == 0); the excerpt keeps them apart
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 2, "different statements must not merge on a shared bucket"

    def test_unverified_anchors_are_not_merged_on_a_guessed_line(self, monkeypatch):
        # The excerpt is absent from the file: both stay `unverified`, keeping their
        # CLAIMED lines. They are different sites and must not be collapsed.
        monkeypatch.setattr(bl, "_execute_action", self._writer_with("unrelated\n"))
        findings = [
            {
                "cwe": "CWE-200",
                "file": "x.rb",
                "new_line": 10,
                "code_excerpt": "never appears",
                "body": "one",
            },
            {
                "cwe": "CWE-200",
                "file": "x.rb",
                "new_line": 400,
                "code_excerpt": "never appears",
                "body": "two",
            },
        ]
        counts = asyncio.run(_writer()._verify_anchors(findings))
        assert counts[bl.ANCHOR_UNVERIFIED] == 2
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 2, "unverified anchors keep claimed lines and stay distinct"


class TestContainmentCollapse:
    """Re-reports quoting a longer or shorter span of ONE site keep distinct exact keys.

    The exact key needs a byte-identical excerpt, so all of them survive it.

    Typical residuals: two findings differing only by a trailing ``end``, one quoting just
    the inner line of another, and one carrying an extra ``with`` clause.

    The fallback is LINE-SET SUBSET inside an already-narrow ``(cwe, file, line bucket)``
    group -- never raw substring, which would merge ``activate(user)`` into
    ``deactivate(user)``.
    """

    @staticmethod
    def _writer_with(content):
        async def _fake(metadata, action):
            return content

        return _fake

    def test_trailing_end_only_difference_collapses_keeping_fuller_quote(
        self, monkeypatch
    ):
        content = 'def revoke(conn, params) do\n  token = params["token"]\nend\n'
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-862",
                "file": "oauth.ex",
                "new_line": 120,
                "code_excerpt": 'def revoke(conn, params) do\n  token = params["token"]',
                "body": "revoke accepts a token without client authentication",
            },
            {
                "cwe": "CWE-862",
                "file": "oauth.ex",
                "new_line": 260,
                "code_excerpt": 'def revoke(conn, params) do\n  token = params["token"]\nend',
                "body": "revoke accepts a token without client authentication",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 1, (
            f"a trailing `end` is not a second defect, got {len(out)} findings"
        )
        assert out[0]["code_excerpt"].strip().endswith("end"), (
            "the survivor must be the FULLER quote"
        )

    def test_sub_span_of_another_excerpt_collapses(self, monkeypatch):
        content = 'def revoke(conn, params) do\n  token = params["token"]\nend\n'
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-862",
                "file": "oauth.ex",
                "new_line": 300,
                "code_excerpt": 'def revoke(conn, params) do\n  token = params["token"]\nend',
                "body": "revoke accepts a token without client authentication",
            },
            {
                # Only the inner line -- a strict sub-span of the finding above.
                "cwe": "CWE-862",
                "file": "oauth.ex",
                "new_line": 44,
                "code_excerpt": '  token = params["token"]',
                "body": "the revoke token is read without authenticating the client",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 1, (
            f"a sub-span quote of one site is not a second defect, got {len(out)}"
        )
        assert "def revoke" in out[0]["code_excerpt"], (
            "the survivor must be the FULLER quote"
        )

    def test_extra_with_clause_collapses(self, monkeypatch):
        content = (
            "with {:ok, user} <- fetch(conn) do\n"
            "  with {:ok, app} <- fetch_app(conn) do\n"
            "    update(user, params)\n"
            "  end\n"
            "end\n"
        )
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        shorter = "with {:ok, user} <- fetch(conn) do\n  update(user, params)\nend"
        longer = (
            "with {:ok, user} <- fetch(conn) do\n"
            "  with {:ok, app} <- fetch_app(conn) do\n"
            "  update(user, params)\n"
            "end"
        )
        findings = [
            {
                "cwe": "CWE-639",
                "file": "user_controller.ex",
                "new_line": 210,
                "code_excerpt": shorter,
                "body": "update writes another user's record",
            },
            {
                "cwe": "CWE-639",
                "file": "user_controller.ex",
                "new_line": 480,
                "code_excerpt": longer,
                "body": "update writes another user's record",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 1, (
            f"one extra `with` clause is not a second defect, got {len(out)}"
        )
        assert "fetch_app" in out[0]["code_excerpt"], (
            "the survivor must be the FULLER quote"
        )

    def test_substring_must_not_merge_distinct_handlers(self, monkeypatch):
        """GUARD. ``activate(user)`` IS a substring of ``deactivate(user)``.

        These are two distinct handlers that re-anchor into one bucket in one file. A
        substring containment rule would silently merge them; line-set subset keeps them apart because
        ``{"activate(user)"}`` is not a subset of ``{"deactivate(user)"}``.
        """
        content = "deactivate(user)\nactivate(user)\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-862",
                "file": "admin_api_controller.ex",
                "new_line": 900,
                "code_excerpt": "deactivate(user)",
                "body": "deactivate misses an admin role check",
            },
            {
                "cwe": "CWE-862",
                "file": "admin_api_controller.ex",
                "new_line": 901,
                "code_excerpt": "activate(user)",
                "body": "activate misses an admin role check",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        assert {f["new_line"] for f in findings} == {1, 2}, (
            "both must land in one line bucket for this guard to be exercised"
        )
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 2, (
            "substring containment must NOT merge activate/deactivate, "
            f"got {len(out)} findings"
        )

    def test_degenerate_structural_only_subset_does_not_merge(self, monkeypatch):
        """GUARD -- a line-set of nothing but ``end`` is a subset of almost anything."""
        content = "update(user, params)\nend\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-639",
                "file": "user_controller.ex",
                "new_line": 300,
                "code_excerpt": "update(user, params)\nend",
                "body": "update writes another user's record",
            },
            {
                "cwe": "CWE-639",
                "file": "user_controller.ex",
                "new_line": 301,
                "code_excerpt": "end",
                "body": "a block terminator quoted on its own",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 2, (
            f"a structural-only excerpt must not be merged away, got {len(out)}"
        )

    def test_containment_does_not_cross_line_buckets(self, monkeypatch):
        """Containment is scoped to an already-narrow group, so two real sites far apart in one file that quote the same
        statement both survive."""
        content = "run_query(sql)\n" + "pad\n" * 39 + "run_query(sql)\n"
        monkeypatch.setattr(bl, "_execute_action", self._writer_with(content))
        findings = [
            {
                "cwe": "CWE-89",
                "file": "repo.rb",
                "new_line": 1,
                "code_excerpt": "run_query(sql)\nend",
                "body": "first site interpolates user input",
            },
            {
                "cwe": "CWE-89",
                "file": "repo.rb",
                "new_line": 41,
                "code_excerpt": "run_query(sql)",
                "body": "second site interpolates user input",
            },
        ]
        asyncio.run(_writer()._verify_anchors(findings))
        assert {f["new_line"] for f in findings} == {1, 41}, (
            "the two sites must re-anchor to different buckets"
        )
        out = bl._collapse_after_reanchor(findings)
        assert len(out) == 2, (
            f"containment must not reach across line buckets, got {len(out)}"
        )

    def test_is_structural_line_classifies_noise_and_code(self):
        for noise in (
            "end",
            "}",
            ")",
            "do",
            "else",
            "{",
            "];",
            "end)",
            "  END  ",
            ";",
            "",
        ):
            assert bl._is_structural_line(noise), f"{noise!r} should be structural"
        for code in ("token = params[:token]", "end_user = fetch()", "do_thing(user)"):
            assert not bl._is_structural_line(code), f"{code!r} should be substantive"
        assert bl._has_substantive_line({"end", "token = x"})
        assert not bl._has_substantive_line({"end", "}"})
        assert not bl._has_substantive_line(set())


class TestOverlappingSpanMerge:
    # Re-reports of one handler quote different parts of it.
    ORDERS_PY = """import sqlite3
from flask import Blueprint

from app.auth import login_required
from app.db import get_db

bp = Blueprint("orders", __name__)


@login_required
@bp.route("/orders/<int:order_id>")
def get_order(order_id):
    db = get_db()
    row = db.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    return dict(row)

@bp.route("/orders/<int:order_id>/refund", methods=["POST"])
@login_required
def refund_order(order_id):
    get_db().execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))
    get_db().commit()
    return {"ok": True}


@bp.route("/orders/<int:order_id>/lines")
@login_required
def order_lines(order_id):
    return {"lines": []}

@bp.route("/orders/<int:order_id>/cancel", methods=["POST"])
def cancel_order(order_id):
    order = get_db().execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if order["status"] != "shipped":
        get_db().execute("DELETE FROM orders WHERE id = ?", (order_id,))
    return {"ok": True}
"""

    def _collapse(self, monkeypatch, *spans):
        async def _fake(metadata, action):
            return self.ORDERS_PY

        monkeypatch.setattr(bl, "_execute_action", _fake)
        lines = self.ORDERS_PY.splitlines()
        findings = [
            {
                "cwe": cwe,
                "file": "app/orders.py",
                "new_line": start,
                "code_excerpt": "\n".join(lines[start - 1 : end]),
                "body": f"finding at {start}",
            }
            for start, end, cwe in spans
        ]
        contents: dict = {}
        asyncio.run(_writer()._verify_anchors(findings, contents))
        return [f["body"] for f in bl._collapse_after_reanchor(findings, contents)]

    @pytest.mark.parametrize(
        "first,second",
        [
            ((11, 12), (12, 13)),
            ((17, 20), (20, 20)),
            ((31, 32), (32, 33)),
            ((32, 34), (34, 35)),
        ],
    )
    def test_quotes_of_one_handler_merge(self, monkeypatch, first, second):
        out = self._collapse(monkeypatch, (*first, "CWE-639"), (*second, "CWE-639"))
        assert out == [f"finding at {first[0]}"]

    def test_the_report_writer_merges_them(self, monkeypatch):
        writes = {}

        async def _fake(metadata, action):
            if action.HasField("runReadFile"):
                return self.ORDERS_PY
            writes["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        lines = self.ORDERS_PY.splitlines()
        findings = [
            {
                "cwe": "CWE-639",
                "file": "app/orders.py",
                "new_line": start,
                "code_excerpt": "\n".join(lines[start - 1 : end]),
                "body": "Any user can refund any order.",
            }
            for start, end in ((17, 20), (20, 20))
        ]
        asyncio.run(_writer()._execute(json.dumps(findings)))
        assert len(json.loads(writes["contents"])["vulnerabilities"]) == 1

    VIEWS_PY = """@login_required
def update_a(request):
    obj = A.objects.get(pk=request.GET["id"])
    obj.save()
def update_b(request):
    obj = B.objects.get(pk=request.GET["id"])
    obj.delete()
@login_required
def other(request):
    return None
"""

    def _collapse_views(self, monkeypatch, *excerpts):
        async def _fake(metadata, action):
            return self.VIEWS_PY

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = [
            {"cwe": "CWE-639", "file": "views.py", "new_line": 1, "code_excerpt": exc}
            for exc in excerpts
        ]
        sources: dict = {}
        asyncio.run(_writer()._verify_anchors(findings, sources))
        return findings, bl._collapse_after_reanchor(findings, sources)

    def test_a_decorator_first_quote_does_not_reach_the_next_function(
        self, monkeypatch
    ):
        lines = self.VIEWS_PY.splitlines()
        findings, out = self._collapse_views(
            monkeypatch, "\n".join(lines[0:4]), "\n".join(lines[4:7])
        )
        assert findings[0]["new_line"] == 2
        assert len(out) == 2

    def test_an_unverified_quote_is_not_merged(self):
        lines = self.VIEWS_PY.splitlines()
        findings = [
            {
                "cwe": "CWE-639",
                "file": "views.py",
                "new_line": line,
                "code_excerpt": lines[1],
                "anchor_status": status,
            }
            # Lines in different buckets, so only this pass could merge them.
            for line, status in ((2, bl.ANCHOR_VERIFIED), (9, bl.ANCHOR_UNVERIFIED))
        ]
        out = bl._collapse_after_reanchor(findings, {"views.py": self.VIEWS_PY})
        assert len(out) == 2

    def _merge(self, *specs):
        lines = self.ORDERS_PY.splitlines()
        findings = [
            {
                "cwe": "CWE-639",
                "file": "app/orders.py",
                "new_line": start,
                "code_excerpt": "\n".join(lines[start - 1 : end]),
                "anchor_status": status,
            }
            for start, end, status in specs
        ]
        out = bl._collapse_after_reanchor(findings, {"app/orders.py": self.ORDERS_PY})
        return [f["new_line"] for f in out]

    def test_a_verified_finding_is_kept_over_a_corrected_one(self):
        kept = self._merge((17, 20, bl.ANCHOR_CORRECTED), (20, 20, bl.ANCHOR_VERIFIED))
        assert kept == [20]

    def test_the_lowest_line_is_kept_when_both_are_verified(self):
        kept = self._merge((20, 20, bl.ANCHOR_VERIFIED), (17, 20, bl.ANCHOR_VERIFIED))
        assert kept == [17]

    def test_different_functions_stay_apart(self, monkeypatch):
        out = self._collapse(monkeypatch, (14, 14, "CWE-639"), (20, 20, "CWE-639"))
        assert len(out) == 2

    def test_a_line_repeated_in_the_file_is_not_shared(self, monkeypatch):
        out = self._collapse(monkeypatch, (20, 22, "CWE-639"), (34, 35, "CWE-639"))
        assert len(out) == 2

    def test_a_different_cwe_stays_apart(self, monkeypatch):
        out = self._collapse(monkeypatch, (11, 12, "CWE-639"), (12, 13, "CWE-862"))
        assert len(out) == 2


# --------------------------------------------------------------------------- #
# Anchor verification must read the file IN FULL
#
# The executor that serves these actions (the Node `duo` CLI) caps a read at
# 51,200 bytes / 2,000 lines on a whole-line boundary and appends a
# "(Showing lines X-Y of N total. Use offset=Z ...)" footer. An unbounded
# `runReadFile` therefore returns a SILENT PREFIX: nothing raises, the result is
# a perfectly ordinary non-empty string, and only the footer says otherwise.
#
# If `_verify_anchors` searched only that prefix, quotes below the cut would be
# declared "not present in the file" (a false `unverified`) or matched to a
# lookalike inside the prefix and MOVED there.
#
# The other fakes in this module return whole files with no footer, so none of
# them exercise pagination. This one reproduces the truncation.
# --------------------------------------------------------------------------- #
class _PaginatingReader:
    """An `_execute_action` fake that truncates a read and advertises a resume offset.

    Mirrors `_PaginatingExecutor` in tests/duo_workflow_service/executor/test_action.py,
    which documents the production behaviour being imitated.
    """

    def __init__(self, files: dict, max_bytes: int = 51_200):
        self.files = files
        self.max_bytes = max_bytes
        self.reads: list = []

    async def __call__(self, metadata, action):
        if not action.HasField("runReadFile"):
            return "ok"
        req = action.runReadFile
        lines = self.files[req.filepath].split("\n")
        start = req.offset if req.HasField("offset") else 0
        self.reads.append(start)
        if start >= len(lines):
            return f"[Offset {start} is beyond end of file ({len(lines)} lines).]"
        page: list = []
        size = 0
        for i in range(start, len(lines)):
            size += len(lines[i]) + 1
            if page and size > self.max_bytes:
                break
            page.append(lines[i])
        end = start + len(page)
        body = "\n".join(page)
        if end < len(lines):
            body += (
                f"\n\n[Showing lines {start}-{end - 1} of {len(lines)} total. "
                f"Use offset={end} to continue reading.]"
            )
        return body


# A handler file over the cap, in the shape that does the damage: one call line
# that recurs at hundreds of unrelated sites, and the ONE signature that pins the
# real site sitting past the cut. 1,000 decoy blocks puts the true site well
# beyond both the 51,200-byte and the 2,000-line cap.
_DECOY_BLOCKS = 1_000


def _oversized_handler_file() -> tuple[str, int]:
    """Return (content, 1-based line the anchor pass should settle on).

    That is the SIGNATURE line, not the ``db.Exec`` line below it:
    :func:`bl._find_excerpt_line` deliberately prefers the excerpt key that pins
    exactly ONE site, and here only the signature does.
    """
    lines = ["package handlers", ""]
    for i in range(_DECOY_BLOCKS):
        lines += [
            f"func decoyHandler{i:04d}(ctx *Context) {{",
            "\tdb.Exec(query)",
            "}",
            "",
        ]
    signature_line = len(lines) + 1  # 1-based
    lines += [
        "func DeleteProjectHandler(ctx *Context) {",
        "\tdb.Exec(query)",  # <- the vulnerable statement this finding is about
        "}",
    ]
    return "\n".join(lines), signature_line


_OVERSIZED, _TRUE_LINE = _oversized_handler_file()

# The quote the detect prompt requires: verbatim, and it spans the signature that
# makes the site identifiable plus the statement itself.
_EXCERPT = "func DeleteProjectHandler(ctx *Context) {\n\tdb.Exec(query)"


class TestAnchorVerificationReadsTheWholeFile:
    def test_the_fake_reproduces_the_production_truncation(self):
        """Control: the fake must genuinely truncate.

        If the true site already fits inside page one, every assertion below is
        vacuous and the regression test proves nothing.
        """
        assert len(_OVERSIZED.encode()) > 51_200
        assert len(_OVERSIZED.splitlines()) > 2_000
        reader = _PaginatingReader({"handlers.go": _OVERSIZED})
        page = asyncio.run(
            reader(
                {},
                contract_pb2.Action(
                    runReadFile=contract_pb2.ReadFile(filepath="handlers.go")
                ),
            )
        )
        assert "Showing lines" in page
        assert "DeleteProjectHandler" not in page, (
            "the true site must sit BEYOND the first page for this test to bite"
        )
        # ...while the lookalike the truncated matcher would settle for is inside it.
        assert "db.Exec(query)" in page

    def test_a_quote_beyond_the_first_page_anchors_on_the_true_line(self, monkeypatch):
        """A quote past the first page is located by reading the whole file.

        Without pagination the only key that survives is the recurring
        ``db.Exec(query)``, which matches hundreds of decoys inside the prefix,
        and the finding is MOVED onto the nearest one -- thousands of lines from
        the code it describes. Reading in full restores the unique signature key,
        which pins exactly one site.
        """
        reader = _PaginatingReader({"handlers.go": _OVERSIZED})
        monkeypatch.setattr(bl, "_execute_action", reader)
        findings = [
            {
                "cwe": "CWE-862",
                "file": "handlers.go",
                "new_line": _TRUE_LINE,
                "code_excerpt": _EXCERPT,
            }
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert findings[0]["new_line"] == _TRUE_LINE, (
            f"anchor MOVED to line {findings[0]['new_line']} instead of "
            f"{_TRUE_LINE} -- a lookalike inside the truncated prefix was matched"
        )
        assert findings[0]["anchor_status"] == bl.ANCHOR_VERIFIED
        assert counts[bl.ANCHOR_CORRECTED] == 0
        assert counts[bl.ANCHOR_VERIFIED] == 1
        assert len(reader.reads) > 1, (
            "the read never paginated -- the anchor pass is issuing one "
            "unbounded runReadFile and getting a silent prefix"
        )

    def test_a_quote_beyond_the_first_page_is_not_reported_missing(self, monkeypatch):
        """The other half of the damage: a quote past the cut read as absent from the file.

        Such a quote is locatable once the file is read in full.
        """
        reader = _PaginatingReader({"handlers.go": _OVERSIZED})
        monkeypatch.setattr(bl, "_execute_action", reader)
        findings = [
            {
                "cwe": "CWE-862",
                "file": "handlers.go",
                # Only present past the cut, and nothing in the prefix resembles it.
                "code_excerpt": "func DeleteProjectHandler(ctx *Context) {",
                "new_line": 1,
            }
        ]

        asyncio.run(_writer()._verify_anchors(findings))

        assert findings[0]["anchor_status"] != bl.ANCHOR_UNVERIFIED, (
            "a quote that IS in the file was reported absent -- the read was truncated"
        )
        assert findings[0]["new_line"] == _TRUE_LINE

    def test_a_non_paginating_executor_is_unaffected(self, monkeypatch):
        """`_read_file_fully` is a documented no-op against a server that returns whole files.

        This is what keeps every footer-less fake in this module honest.
        """
        monkeypatch.setattr(bl, "_execute_action", _reader(_TWO_FUNCS))
        findings = [
            {"file": "container.go", "new_line": 10, "code_excerpt": "upload.Commit()"}
        ]

        counts = asyncio.run(_writer()._verify_anchors(findings))

        assert counts[bl.ANCHOR_VERIFIED] == 1
        assert findings[0]["new_line"] == 10


class TestFormatDisplayMessage:
    """The default DuoBaseTool.format_display_message dumps every arg's str() into the chat-log `content` field with no
    size cap -- unlike tool_info.tool_response, which IS capped at TOOL_RESPONSE_MAX_DISPLAY_MSG.

    `batches`/`batches2`/`findings` here can carry the full inline findings
    blob; this override must never let
    that reach `content`.
    """

    def test_write_report_reports_findings_count_not_content(self):
        args = bl.BlWriteSastReportInput(
            findings=[{"cwe": "CWE-284", "code_excerpt": "y" * 10_000}]
        )
        message = _writer().format_display_message(args, None)
        assert message == "Writing SAST report for 1 findings"
        assert "y" * 100 not in message


class TestPartialScanMarking:
    """Only a `target_files` scan is marked `partial_scan`, as with diff-based GitLab Advanced SAST.

    A capped run, or one that lost some units or findings, still ran over the whole repository: it stays a full scan,
    and its coverage records are disclosed in `scan.messages` instead.
    """

    @staticmethod
    def _report(monkeypatch, **kwargs):
        written = {}

        async def _fake(metadata, action):
            written["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        asyncio.run(_writer()._execute([], **kwargs))
        return json.loads(written["contents"])

    @staticmethod
    def _coverage(emitted, dispatched):
        return {"emitted": emitted, "dispatched": dispatched, "completed": dispatched}

    def test_a_target_files_scan_is_partial(self, monkeypatch):
        report = self._report(
            monkeypatch,
            target_files="app/a.rb, app/b.rb",
            review_coverage=self._coverage(2, 2),
        )
        assert report["scan"]["partial_scan"] == {"mode": "differential"}

    def test_a_capped_scan_is_full(self, monkeypatch):
        report = self._report(
            monkeypatch,
            review_coverage=self._coverage(200, 150),
            sibling_coverage=json.dumps(self._coverage(200, 150)),
            triage_coverage=self._coverage(80, 50),
        )
        assert "partial_scan" not in report["scan"]

    def test_a_scan_that_lost_units_is_full_and_says_so(self, monkeypatch):
        loss = (
            "2 of 10 review units were DISCARDED without their findings being "
            "counted (2 errored)."
        )
        report = self._report(
            monkeypatch, review_coverage={**self._coverage(10, 10), "loss": loss}
        )
        assert "partial_scan" not in report["scan"]
        assert {"level": "warn", "value": loss} in report["scan"]["messages"]

    @pytest.mark.parametrize("target_files", [None, "", " , ;"])
    def test_a_full_uncapped_scan_is_not_partial(self, monkeypatch, target_files):
        report = self._report(
            monkeypatch,
            target_files=target_files,
            review_coverage=self._coverage(120, 120),
            sibling_coverage=self._coverage(None, 120),
            triage_coverage=self._coverage(50, 50),
        )
        assert "partial_scan" not in report["scan"]


class TestTrackingSignature:
    SOURCE = """import x

@login_required
@bp.route("/a")
def get_order(order_id):
    row = db.get(order_id)
    return row


def refund(order_id):
    db.update(order_id)
    db.delete(order_id)

TOP = load()
"""
    # The source form of `vulnerabilities[].tracking`, abridged from the SAST report schema 15.2.2:
    # https://gitlab.com/gitlab-org/security-products/security-report-schemas/-/blob/v15.2.2/dist/sast-report-format.json
    TRACKING_SCHEMA = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "type": {"const": "source"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["signatures"],
                    "properties": {
                        "file": {"type": "string"},
                        "start_line": {"type": "number"},
                        "end_line": {"type": "number"},
                        "signatures": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["algorithm", "value"],
                                "properties": {
                                    "algorithm": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    def _vulns(self, lines, sources=None, status=bl.ANCHOR_VERIFIED):
        findings = [
            {
                "cwe": "CWE-639",
                "file": "app/orders.py",
                "new_line": n,
                "body": "b",
                "anchor_status": status,
            }
            for n in lines
        ]
        if sources is None:
            sources = {"app/orders.py": self.SOURCE}
        return _writer()._build_report(findings, sources=sources)["vulnerabilities"]

    def _signature(self, vuln):
        return vuln["tracking"]["items"][0]["signatures"][0]["value"]

    @pytest.mark.parametrize("line", [3, 4, 5, 6])
    def test_decorator_def_and_body_quotes_share_one_signature(self, line):
        (vuln,) = self._vulns([line])
        assert self._signature(vuln) == "app/orders.py|get_order[0]:639"

    def test_same_cwe_bugs_in_one_function_are_numbered_by_line(self):
        vulns = self._vulns([12, 11])
        assert [self._signature(v) for v in vulns] == [
            "app/orders.py|refund[1]:639",
            "app/orders.py|refund[0]:639",
        ]

    def test_different_functions_differ(self):
        a, b = self._vulns([6, 11])
        assert self._signature(a) != self._signature(b)

    @pytest.mark.parametrize("line", [14, 99])
    def test_code_outside_any_function_is_module_scope(self, line):
        (vuln,) = self._vulns([line])
        assert self._signature(vuln) == "app/orders.py|<module>[0]:639"

    @pytest.mark.parametrize("sources", [{}, {"app/orders.py": None}])
    def test_an_unread_file_gets_no_tracking(self, sources):
        (vuln,) = self._vulns([6], sources)
        assert "tracking" not in vuln

    def test_an_unlocated_quote_gets_no_tracking_and_no_slot(self):
        findings = [
            {
                "cwe": "CWE-639",
                "file": "app/orders.py",
                "new_line": n,
                "anchor_status": status,
            }
            for n, status in (
                (11, bl.ANCHOR_UNVERIFIED),
                (12, bl.ANCHOR_VERIFIED),
            )
        ]
        sources = {"app/orders.py": self.SOURCE}
        first, second = _writer()._build_report(findings, sources=sources)[
            "vulnerabilities"
        ]
        assert "tracking" not in first
        assert self._signature(second) == "app/orders.py|refund[0]:639"

    def test_findings_on_one_line_are_numbered_by_quote_not_input_order(self):
        def _run(order):
            findings = [
                {
                    "cwe": "CWE-639",
                    "file": "app/orders.py",
                    "new_line": 11,
                    "code_excerpt": quote,
                    "anchor_status": bl.ANCHOR_VERIFIED,
                }
                for quote in order
            ]
            vulns = _writer()._build_report(
                findings, sources={"app/orders.py": self.SOURCE}
            )["vulnerabilities"]
            return {v["raw_source_code_extract"]: self._signature(v) for v in vulns}

        quotes = ["db.update(order_id)", "update(order_id)"]
        assert _run(quotes) == _run(list(reversed(quotes)))

    def test_the_written_report_carries_tracking(self, monkeypatch):
        writes = {}

        async def _fake(metadata, action):
            if action.HasField("runReadFile"):
                return self.SOURCE
            writes["contents"] = action.runWriteFile.contents
            return "ok"

        monkeypatch.setattr(bl, "_execute_action", _fake)
        finding = {
            "cwe": "CWE-639",
            "file": "app/orders.py",
            "new_line": 11,
            "code_excerpt": "db.update(order_id)",
            "body": "b",
        }
        asyncio.run(_writer()._execute(json.dumps([finding])))
        (vuln,) = json.loads(writes["contents"])["vulnerabilities"]
        assert self._signature(vuln) == "app/orders.py|refund[0]:639"

    def test_tracking_matches_the_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        (vuln,) = self._vulns([6])
        jsonschema.validate(vuln["tracking"], self.TRACKING_SCHEMA)
        assert vuln["tracking"]["items"][0]["signatures"][0]["algorithm"] == (
            "scope_offset"
        )

    @pytest.mark.parametrize(
        "source,line,name",
        [
            ("func (s *Server) Get(w http.ResponseWriter) {\n\tx := 1\n}", 2, "Get"),
            ("export const handler = async (req, res) => {\n  x();\n};", 2, "handler"),
            ("function show($id) {\n  x();\n}", 2, "show"),
            ("  public ResponseEntity<Order> get(Long id) {\n    x();\n  }", 2, "get"),
            ("[Authorize]\npublic IActionResult Get(int id)\n{\n  x();\n}", 1, "Get"),
            ("fun update(id: Long): Order {\n    x()\n}", 2, "update"),
            ("  def destroy\n    x\n  end", 2, "destroy"),
            ("  defp load(id) do\n    x\n  end", 2, "load"),
            ("class C {\n  async remove(req) {\n    x();\n  }\n}", 3, "remove"),
            ("def f():\n    if x(y):\n        return g(y)", 3, "f"),
            ("def f(p):\n    with open(p) as fh:\n        x", 3, "f"),
            ("def f():\n    try:\n        x\n    except Error(y):\n        z", 5, "f"),
            ("func F() {\n\tgo handle(c)\n}", 2, "F"),
            ("function f($a) {\n  foreach ($a as $b) {\n    x();\n  }\n}", 3, "f"),
            ("void f() {\n  synchronized (m) {\n    x();\n  }\n}", 3, "f"),
            ("void F() {\n  lock (m) {\n    x();\n  }\n}", 3, "F"),
            (
                "export const h = async (req: Request, res: Response): Promise<void> => {\n  x();\n};",
                2,
                "h",
            ),
            ("export const h = async (\n  req: Request,\n) => {\n  x();\n};", 4, "h"),
        ],
    )
    def test_the_enclosing_function_per_language(self, source, line, name):
        assert bl._enclosing_function(source, line) == name


class TestPathRepair:
    REAL = "netbox/vpn/graphql/schema.py"
    SOURCE = "class Query:\n    def tunnel(self, info, id):\n        return Tunnel.objects.get(pk=id)\n"
    QUOTE = "return Tunnel.objects.get(pk=id)"

    def _fake(self, listing, calls):
        async def _fake(metadata, action):
            if action.HasField("findFiles"):
                calls.append(action.findFiles.name_pattern)
                return "\n".join(listing)
            return self.SOURCE if action.runReadFile.filepath == self.REAL else ""

        return _fake

    def _run(self, monkeypatch, listing, files):
        calls: list = []
        monkeypatch.setattr(bl, "_execute_action", self._fake(listing, calls))
        findings = [
            {"cwe": "CWE-639", "file": file, "new_line": 3, "code_excerpt": self.QUOTE}
            for file in files
        ]
        sources: dict = {}
        asyncio.run(_writer()._verify_anchors(findings, sources))
        return findings, bl._collapse_after_reanchor(findings, sources), calls

    def test_a_dropped_prefix_is_repaired_and_merges_with_its_twin(self, monkeypatch):
        findings, out, calls = self._run(
            monkeypatch,
            [self.REAL, "netbox/vpn/models.py"],
            [self.REAL, "vpn/graphql/schema.py", "vpn/graphql/schema.py"],
        )
        assert findings[1]["file"] == self.REAL
        assert findings[1]["claimed_file"] == "vpn/graphql/schema.py"
        assert findings[1]["anchor_status"] == bl.ANCHOR_VERIFIED
        assert len(out) == 1
        assert calls == ["schema.py"]

    def test_an_ambiguous_basename_is_left_alone(self, monkeypatch):
        other = "netbox/other/vpn/graphql/schema.py"
        findings, _, _ = self._run(
            monkeypatch, [self.REAL, other], ["vpn/graphql/schema.py"]
        )
        assert findings[0]["file"] == "vpn/graphql/schema.py"
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED

    def test_no_match_is_left_alone(self, monkeypatch):
        findings, _, _ = self._run(monkeypatch, ["netbox/vpn/models.py"], ["vpn/x.py"])
        assert findings[0]["file"] == "vpn/x.py"
        assert "claimed_file" not in findings[0]
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED

    def test_a_failed_lookup_is_left_alone(self, monkeypatch):
        async def _fake(metadata, action):
            if action.HasField("findFiles"):
                raise RuntimeError("no listing")
            return ""

        monkeypatch.setattr(bl, "_execute_action", _fake)
        findings = [{"cwe": "CWE-639", "file": "vpn/x.py", "code_excerpt": "x()"}]
        asyncio.run(_writer()._verify_anchors(findings, {}))
        assert findings[0]["file"] == "vpn/x.py"
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED

    @pytest.mark.parametrize(
        "listing",
        [
            ["[... output truncated: 60000 bytes ...]", "x/vpn/graphql/schema.py"],
            [
                "Maximum allowed size exceeded. Showing a sample.",
                "/vpn/graphql/schema.py",
            ],
        ],
    )
    def test_a_truncated_listing_is_left_alone(self, monkeypatch, listing):
        findings, _, _ = self._run(monkeypatch, listing, ["vpn/graphql/schema.py"])
        assert findings[0]["file"] == "vpn/graphql/schema.py"
        assert findings[0]["anchor_status"] == bl.ANCHOR_UNVERIFIED

    def test_an_exit_code_header_is_ignored(self, monkeypatch):
        findings, _, _ = self._run(
            monkeypatch, ["Exit code: 0", self.REAL], ["vpn/graphql/schema.py"]
        )
        assert findings[0]["file"] == self.REAL

    def test_an_unreadable_match_is_not_used(self, monkeypatch):
        other = "netbox/old/vpn/graphql/schema.py"
        findings, _, _ = self._run(monkeypatch, [other], ["vpn/graphql/schema.py"])
        assert findings[0]["file"] == "vpn/graphql/schema.py"
        assert "claimed_file" not in findings[0]

    def test_lookups_stop_at_the_cap(self, monkeypatch):
        monkeypatch.setattr(bl, "_MAX_PATH_LOOKUPS", 0)
        findings, _, calls = self._run(
            monkeypatch, [self.REAL], ["vpn/graphql/schema.py"]
        )
        assert calls == []
        assert findings[0]["file"] == "vpn/graphql/schema.py"

    def test_the_report_records_the_claimed_path(self, monkeypatch):
        _, out, _ = self._run(monkeypatch, [self.REAL], ["vpn/graphql/schema.py"])
        (vuln,) = _writer()._build_report(out)["vulnerabilities"]
        detail = vuln["details"][bl.ANCHOR_CLAIMED_FILE_DETAIL_KEY]
        assert detail["value"] == "vpn/graphql/schema.py"
        assert vuln["location"]["file"] == self.REAL
