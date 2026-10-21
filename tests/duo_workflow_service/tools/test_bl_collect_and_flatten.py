import json

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.experimental.components.for_each import (
    ITEM_ERROR_SUBKEY,
)
from duo_workflow_service.tools.bl_collect_and_flatten import (
    _UNIT_ERROR_KEY,
    BlCollectAndFlatten,
    BlCollectAndFlattenInput,
    _coverage_sentences,
    _unit_findings,
)
from duo_workflow_service.tools.bl_report import BlDedupFindings, _flatten_findings
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD


def _failed(message: str = "boom") -> dict:
    """A failed unit's results entry, as `for_each` records it."""
    return {ITEM_ERROR_SUBKEY: {"type": "RuntimeError", "message": message}}


def _unit(*findings: dict) -> str:
    """A unit answer as it is stored: JSON of the structured answer."""
    return json.dumps(
        {"explanation": "walked every handler", "findings": list(findings)}
    )


@pytest.mark.asyncio
class TestBlCollectAndFlatten:
    async def test_units_with_a_for_each_error_record_are_errored_without_losing_completed(
        self,
    ):
        tool = BlCollectAndFlatten(metadata={})
        results = [
            {"final_answer": _unit({"id": "ok"})},
            _failed("could not parse"),
            _failed("boom"),
        ]
        result = await tool._execute(results=results, emitted_count=3)
        parsed = json.loads(result["final_answer"])
        assert parsed == [{"id": "ok"}]
        assert result["coverage"]["completed"] == 1
        assert result["coverage"]["errored"] == 2
        assert "2 of 3" in result["coverage"]["loss"]
        assert "2 failed" in result["coverage"]["loss"]

    async def test_a_for_each_error_record_wins_over_an_answer_in_the_same_entry(
        self,
    ):
        # The record is what marks a unit failed: an answer sitting beside it is
        # not a completed unit's answer.
        results = [
            {"final_answer": _unit({"id": "ok"})},
            {"final_answer": _unit({"id": "stale"}), **_failed()},
        ]
        result = await BlCollectAndFlatten(metadata={})._execute(results=results)
        assert json.loads(result["final_answer"]) == [{"id": "ok"}]
        assert result["coverage"]["completed"] == 1
        assert result["coverage"]["errored"] == 1

    async def test_the_error_key_is_the_one_for_each_writes(self):
        assert _UNIT_ERROR_KEY == ITEM_ERROR_SUBKEY

    async def test_final_answer_is_the_flattened_findings_not_the_unit_answers(self):
        tool = BlCollectAndFlatten(metadata=None)
        results = [{"final_answer": _unit({"id": "x"})}]
        result = await tool._execute(results=results)
        parsed = json.loads(result["final_answer"])
        assert parsed == [{"id": "x"}]

    async def test_empty_results_yields_empty_aggregate(self):
        tool = BlCollectAndFlatten(metadata={})
        result = await tool._execute(results=[])
        assert json.loads(result["final_answer"]) == []
        assert result["coverage"]["completed"] == 0
        assert result["coverage"]["dispatched"] == 0


class TestFormatDisplayMessage:
    """The default DuoBaseTool.format_display_message dumps every arg's str() into the chat-log `content` field with no
    size cap.

    `results` here is the
    full for_each results list and can be hundreds of KB; this override must
    never let it reach `content`.
    """

    def test_reports_result_count_not_content(self):
        tool = BlCollectAndFlatten(metadata={})
        args = BlCollectAndFlattenInput(results=[{"final_answer": "x" * 10_000}] * 50)
        message = tool.format_display_message(args, None)
        assert message == "Collecting and flattening 50 unit results"
        assert "x" * 100 not in message


@pytest.mark.asyncio
class TestResolveEntry:
    """Only a completed unit's own answer contributes to the aggregate."""

    async def test_a_non_dict_entry_resolves_to_nothing(self):
        tool = BlCollectAndFlatten(metadata={})

        assert tool._resolve_entry("not an entry") is None

    async def test_a_failed_entry_resolves_to_nothing(self):
        """A unit carrying a `for_each_error` record contributes no answer -- its `final_answer`, if any, is not one."""
        tool = BlCollectAndFlatten(metadata={})

        assert tool._resolve_entry({"final_answer": '[{"i": 1}]', **_failed()}) is None


@pytest.mark.asyncio
async def test_a_garbled_unit_answer_is_counted_unread_not_reviewed():
    """An answer that does not parse as a structured answer must not pass as a reviewed unit with no findings."""
    tool = BlCollectAndFlatten(metadata={})
    results = [
        {"final_answer": '{"explanation": "r", "findings": [{'},
        {
            "final_answer": '{"explanation": "r", "findings": [{"id": "ok"}]}',
        },
    ]

    out = await tool._execute(results=results)
    cov = out["coverage"]

    assert cov["summary"] == "Reviewed 1 of 2 units."
    assert "1 returned an answer that could not be read" in cov["loss"]


@pytest.mark.asyncio
class TestAUnitWithNoAnswer:
    """A unit that completed without an answer contributes nothing, so it must not be counted as reviewed."""

    async def test_it_is_reported_as_lost_not_reviewed(self):
        tool = BlCollectAndFlatten(metadata={})
        results = [
            {},
            {"final_answer": _unit({"id": "ok"})},
        ]

        cov = (await tool._execute(results=results))["coverage"]

        assert cov["summary"] == "Reviewed 1 of 2 units."
        assert "1 of 2 units are missing from the results" in cov["loss"]
        assert "1 returned an answer that could not be read" in cov["loss"]


@pytest.mark.asyncio
class TestADictFinalAnswer:
    """In schema mode the plain AgentComponent stores `final_answer` as the decoded dict, not its JSON string."""

    async def test_its_findings_are_flattened_and_the_unit_counted_completed(self):
        tool = BlCollectAndFlatten(metadata=None)
        answer = {"explanation": "walked every handler", "findings": [{"id": "d1"}]}

        out = await tool._execute(results=[{"final_answer": answer}])

        assert json.loads(out["final_answer"]) == [{"id": "d1"}]
        assert out["coverage"]["completed"] == 1
        assert out["coverage"]["summary"] == "Reviewed 1 of 1 units."
        assert "loss" not in out["coverage"]

    @pytest.mark.parametrize("answer", [None, {}])
    async def test_an_empty_answer_is_completed_but_unread(self, answer):
        tool = BlCollectAndFlatten(metadata=None)

        out = await tool._execute(results=[{"final_answer": answer}])
        cov = out["coverage"]

        assert cov["completed"] == 1
        assert cov["summary"] == "Reviewed 0 of 1 units."
        assert "1 returned an answer that could not be read" in cov["loss"]


class TestCoverageSentencesReachTheReport:
    """Coverage carries the `summary`/`truncation`/`loss` sentences alongside the raw counts.

    A reader that renders sentences rather than numbers must have something to render for every stage, including a fully
    covered one.
    """

    def test_summary_names_the_stage_s_own_unit_noun(self):
        cov = _coverage_sentences(
            noun="findings",
            emitted=12,
            dispatched=12,
            completed=12,
            errored=0,
        )
        assert cov["summary"] == "Reviewed 12 of 12 findings."
        # A clean stage discloses no cap and no loss.
        assert "truncation" not in cov
        assert "loss" not in cov

    def test_a_cap_that_bound_is_disclosed_separately_from_loss(self):
        cov = _coverage_sentences(
            noun="review units",
            emitted=150,
            dispatched=62,
            completed=62,
            errored=0,
        )
        assert "62 of 150 review units were sent for review" in cov["truncation"]
        assert "88 were never reviewed" in cov["truncation"]
        # A cap is a CHOICE; it must not read as discarded findings.
        assert "loss" not in cov

    def test_discarded_units_are_reported_as_loss_not_as_a_cap(self):
        cov = _coverage_sentences(
            noun="review units",
            emitted=40,
            dispatched=40,
            completed=40,
            errored=0,
            unread=14,
        )
        assert "14 of 40 review units are missing from the results" in cov["loss"]
        assert "14 returned an answer that could not be read" in cov["loss"]
        assert "truncation" not in cov

    def test_errored_and_unread_units_are_counted_together_but_named_apart(self):
        cov = _coverage_sentences(
            noun="units",
            emitted=10,
            dispatched=10,
            completed=9,
            errored=1,
            unread=2,
        )
        assert "3 of 10 units are missing from the results" in cov["loss"]
        assert "2 returned an answer that could not be read" in cov["loss"]
        assert "1 failed" in cov["loss"]

    def test_a_stage_that_dispatched_nothing_still_says_so(self):
        cov = _coverage_sentences(
            noun="findings",
            emitted=0,
            dispatched=0,
            completed=0,
            errored=0,
        )
        assert cov["summary"] == "No findings were dispatched for this stage."

    @pytest.mark.asyncio
    async def test_the_tool_publishes_the_sentences_alongside_the_counts(self):
        tool = BlCollectAndFlatten(name="bl_collect_and_flatten", description="d")
        out = await tool._execute(
            results=[{"final_answer": _unit()}],
            unit_noun="findings",
        )
        cov = out["coverage"]
        # Counts stay for programmatic readers...
        assert cov["dispatched"] == 1 and cov["completed"] == 1
        # ...and the sentence the report actually renders is present.
        assert cov["summary"] == "Reviewed 1 of 1 findings."
        assert cov["unit_noun"] == "findings"


def test_the_tool_is_hidden_from_list_tools():
    # ListTools publishes only tools at or above STABLE_VERSION_THRESHOLD.
    assert BlCollectAndFlatten.tool_version < STABLE_VERSION_THRESHOLD


@pytest.mark.asyncio
async def test_what_collect_publishes_is_what_dedup_reads():
    """Collect publishes flat finding dicts inline; dedup flattens them unchanged and keeps every distinct finding."""
    results = [
        {"final_answer": _unit({"cwe": "CWE-639", "file": "a.py", "line": 1})},
        {"final_answer": _unit({"cwe": "CWE-284", "file": "b.py", "line": 2})},
    ]

    collected = await BlCollectAndFlatten(metadata={})._execute(results=results)
    published = json.loads(collected["final_answer"])
    deduped = await BlDedupFindings(metadata={})._execute(collected["final_answer"])

    assert _flatten_findings(published) == published
    assert sorted(f["file"] for f in deduped) == ["a.py", "b.py"]


_FINDING = {
    "file": "app/c.rb",
    "new_line": 12,
    "body": "missing check",
    "anchor_status": "exact",
}
_VERDICT = {
    "explanation": "read app/c.rb:12 and its policy",
    "verdict": "KEEP",
    "clause": "KEEP-cross-principal",
    "evidence": "app/c.rb:12 no owner scope",
    "triage_evidence": "app/c.rb:12 reachable",
}


@pytest.mark.asyncio
class TestTriageVerdictMerge:
    """Triage answers with a verdict only; the collect step records it on the finding that unit judged."""

    @staticmethod
    async def _collect(verdicts: list, items: list) -> list:
        results = [{"final_answer": json.dumps(v)} for v in verdicts]
        out = await BlCollectAndFlatten(metadata=None)._execute(
            results=results, verdict_items=items
        )
        return json.loads(out["final_answer"])

    async def test_each_verdict_is_merged_onto_its_own_item(self):
        other = {**_FINDING, "file": "app/d.rb"}
        drop = {**_VERDICT, "verdict": "DROP", "clause": "DROP-2"}

        merged = await self._collect([_VERDICT, drop], [dict(_FINDING), other])

        assert merged == [
            {**_FINDING, **{k: v for k, v in _VERDICT.items() if k != "explanation"}},
            {
                **other,
                "verdict": "DROP",
                "clause": "DROP-2",
                "evidence": _VERDICT["evidence"],
                "triage_evidence": _VERDICT["triage_evidence"],
            },
        ]

    async def test_the_verdict_overrides_stale_fields_and_lands_on_the_original_item(
        self,
    ):
        stale = {**_FINDING, "verdict": "KEEP", "evidence": "old"}
        verdict = {
            **_VERDICT,
            "verdict": "DROP",
            "evidence": "new",
            "file": "app/other.rb",
            "finding": {"file": "app/other.rb"},
        }

        (merged,) = await self._collect([verdict], [stale])

        assert merged == {
            **_FINDING,
            "verdict": "DROP",
            "clause": _VERDICT["clause"],
            "evidence": "new",
            "triage_evidence": _VERDICT["triage_evidence"],
        }

    async def test_an_unset_optional_field_is_not_merged(self):
        verdict = {**_VERDICT, "verdict": "DROP", "triage_evidence": None}

        (merged,) = await self._collect([verdict], [dict(_FINDING)])

        assert "triage_evidence" not in merged
        assert merged["verdict"] == "DROP"

    async def test_a_json_string_item_is_merged_too(self):
        (merged,) = await self._collect([_VERDICT], [json.dumps(_FINDING)])

        assert merged["anchor_status"] == "exact"

    @pytest.mark.parametrize("item", ["not a json finding", ["a", "list"]])
    async def test_an_item_that_is_not_a_finding_keeps_the_verdict_alone(self, item):
        verdict = {
            "explanation": "r",
            "verdict": "DROP",
            "clause": "c",
            "evidence": "e",
        }

        with capture_logs() as logs:
            merged = await self._collect([verdict], [item])

        assert merged == [{"verdict": "DROP", "clause": "c", "evidence": "e"}]
        assert any("item is not a finding" in log["event"] for log in logs)

    @pytest.mark.parametrize("answer", ['["a", "list"]', "not json"])
    async def test_a_verdict_that_is_not_an_object_is_counted_unread(self, answer):
        out = await BlCollectAndFlatten(metadata=None)._execute(
            results=[{"final_answer": answer}],
            verdict_items=[dict(_FINDING)],
        )
        cov = out["coverage"]

        assert json.loads(out["final_answer"]) == []
        assert cov["summary"] == "Reviewed 0 of 1 units."
        assert "1 returned an answer that could not be read" in cov["loss"]

    async def test_a_verdict_with_no_verdict_field_is_counted_unread(self):
        out = await BlCollectAndFlatten(metadata=None)._execute(
            results=[
                {"final_answer": json.dumps({"explanation": "x", "error": "oops"})}
            ],
            verdict_items=[dict(_FINDING)],
        )

        assert json.loads(out["final_answer"]) == []
        assert out["coverage"]["summary"] == "Reviewed 0 of 1 units."

    async def test_a_failed_unit_does_not_shift_the_items(self):
        results = [
            _failed(),
            {"final_answer": json.dumps(_VERDICT)},
        ]
        out = await BlCollectAndFlatten(metadata=None)._execute(
            results=results,
            verdict_items=[{"file": "first.rb"}, dict(_FINDING)],
        )

        (merged,) = json.loads(out["final_answer"])
        assert merged["file"] == "app/c.rb"


_ONE_FINDING = {
    "file": "api/widget.go",
    "new_line": 88,
    "cwe": "CWE-639",
    "severity": "high",
    "tier": 1,
}


class TestUnitFindings:
    """What one unit answer contributes: its finding dicts, ``[]`` for an honest empty answer, ``None`` when lost."""

    def test_the_findings_are_the_dicts_under_findings(self):
        assert _unit_findings(_unit(_ONE_FINDING, {"cwe": "CWE-862"})) == [
            _ONE_FINDING,
            {"cwe": "CWE-862"},
        ]

    def test_empty_findings_is_an_answer_not_a_loss(self):
        # A unit that honestly found nothing; re-asking would pay for a second
        # model run to be told the same.
        assert _unit_findings(_unit()) == []

    def test_an_already_decoded_answer_is_read_too(self):
        assert _unit_findings(json.loads(_unit(_ONE_FINDING))) == [_ONE_FINDING]

    def test_findings_holding_no_finding_dict_is_lost(self):
        answer = json.dumps({"explanation": "r", "findings": [1, "prose"]})

        assert _unit_findings(answer) is None

    def test_non_dict_items_beside_a_finding_are_not_counted(self):
        answer = json.dumps({"explanation": "r", "findings": [_ONE_FINDING, "x", None]})

        assert _unit_findings(answer) == [_ONE_FINDING]

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param("prose, not the answer tool", id="prose"),
            pytest.param(json.dumps([_ONE_FINDING]), id="bare-array"),
            pytest.param(json.dumps({"explanation": "r"}), id="no-findings-key"),
            pytest.param(json.dumps({"findings": None}), id="null-findings"),
            pytest.param(json.dumps({"findings": "none"}), id="string-findings"),
            pytest.param(None, id="none"),
        ],
    )
    def test_anything_but_a_findings_list_is_lost(self, answer):
        assert _unit_findings(answer) is None

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param("```json\n" + _unit(_ONE_FINDING) + "\n```", id="fenced"),
            pytest.param("Here you go: " + _unit(_ONE_FINDING), id="prose-around"),
        ],
    )
    def test_json_wrapped_in_text_is_lost_not_salvaged(self, answer):
        # Unit answers are json.dumps of a response-schema tool call, so a
        # string that is not strict JSON did not come from that tool.
        assert _unit_findings(answer) is None


@pytest.mark.asyncio
class TestFlatten:
    async def test_each_units_findings_are_concatenated_in_order(self):
        results = [
            {"final_answer": _unit(_ONE_FINDING)},
            {"final_answer": json.loads(_unit({"cwe": "CWE-862"}))},
        ]

        out = await BlCollectAndFlatten(metadata=None)._execute(results=results)

        assert json.loads(out["final_answer"]) == [_ONE_FINDING, {"cwe": "CWE-862"}]

    async def test_an_unreadable_unit_contributes_nothing_and_is_unread(self):
        results = [
            {"final_answer": "prose"},
            {"final_answer": {"explanation": "r"}},
            {"final_answer": "Result: " + _unit(_ONE_FINDING)},
            {"final_answer": _unit(_ONE_FINDING)},
            {"final_answer": _unit()},
        ]

        out = await BlCollectAndFlatten(metadata=None)._execute(results=results)

        assert json.loads(out["final_answer"]) == [_ONE_FINDING]
        assert out["coverage"]["summary"] == "Reviewed 2 of 5 units."
        assert (
            "3 returned an answer that could not be read" in (out["coverage"]["loss"])
        )
