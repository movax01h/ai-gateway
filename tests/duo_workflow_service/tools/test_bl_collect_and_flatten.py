import json

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.experimental.components.for_each import (
    ITEM_ERROR_SUBKEY,
    failed_item_errors,
)
from duo_workflow_service.tools.bl_collect_and_flatten import (
    BlCollectAndFlatten,
    BlCollectAndFlattenInput,
    _unit_findings,
)
from duo_workflow_service.tools.bl_report import BlDedupFindings, _flatten_findings
from duo_workflow_service.tools.duo_base_tool import STABLE_VERSION_THRESHOLD


def _failed(message: str = "boom") -> dict:
    """A failed unit's results entry, as `for_each` records it."""
    return {ITEM_ERROR_SUBKEY: {"type": "RuntimeError", "message": message}}


async def _collect(results: list, **kwargs) -> dict:
    """Run the tool with the counts `for_each` publishes beside these results."""
    failed = len(failed_item_errors({str(i): r for i, r in enumerate(results)}))
    return await BlCollectAndFlatten(metadata=None)._execute(
        results=results, succeeded=len(results) - failed, failed=failed, **kwargs
    )


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
        results = [
            {"final_answer": _unit({"id": "ok"})},
            _failed("could not parse"),
            _failed("boom"),
        ]
        result = await _collect(results, emitted_count=3)
        parsed = json.loads(result["final_answer"])
        assert parsed == [{"id": "ok"}]
        assert result["coverage"]["completed"] == 1
        assert result["coverage"]["errored"] == 2
        assert result["coverage"]["unread"] == 0

    async def test_reviewed_indexes_exclude_failed_and_unreadable_units(self):
        results = [
            {"final_answer": _unit({"id": "ok"})},
            _failed("boom"),
            {"final_answer": "not json"},
            {"final_answer": _unit()},
        ]
        result = await _collect(results, emitted_count=4)
        assert result["coverage"]["reviewed_indexes"] == [0, 3]

    async def test_the_counts_are_for_each_s_not_recounted_from_the_entries(self):
        # The counts disagree with the entries on purpose: one answered entry,
        # but `for_each` reports two failures.
        result = await BlCollectAndFlatten(metadata=None)._execute(
            results=[{"final_answer": _unit({"id": "ok"})}], succeeded=1, failed=2
        )
        assert result["coverage"]["completed"] == 1
        assert result["coverage"]["errored"] == 2

    @pytest.mark.parametrize("field", ["succeeded", "failed"])
    async def test_for_each_s_counts_are_required_inputs(self, field):
        assert BlCollectAndFlattenInput.model_fields[field].is_required()
        counts = {"succeeded": 1, "failed": 0}
        del counts[field]
        with pytest.raises(ValidationError):
            BlCollectAndFlattenInput(results=[], **counts)

    async def test_final_answer_is_the_flattened_findings_not_the_unit_answers(self):
        results = [{"final_answer": _unit({"id": "x"})}]
        result = await _collect(results)
        parsed = json.loads(result["final_answer"])
        assert parsed == [{"id": "x"}]

    async def test_empty_results_yields_empty_aggregate(self):
        result = await _collect([])
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
        args = BlCollectAndFlattenInput(
            results=[{"final_answer": "x" * 10_000}] * 50, succeeded=50, failed=0
        )
        message = tool.format_display_message(args, None)
        assert message == "Collecting and flattening 50 unit results"
        assert "x" * 100 not in message


@pytest.mark.asyncio
class TestADictFinalAnswer:
    """In schema mode the plain AgentComponent stores `final_answer` as the decoded dict, not its JSON string."""

    async def test_its_findings_are_flattened_and_the_unit_counted_completed(self):
        answer = {"explanation": "walked every handler", "findings": [{"id": "d1"}]}

        out = await _collect([{"final_answer": answer}])

        assert json.loads(out["final_answer"]) == [{"id": "d1"}]
        assert out["coverage"]["completed"] == 1
        assert out["coverage"]["unread"] == 0


@pytest.mark.asyncio
async def test_coverage_is_the_stage_s_counts_and_unit_noun():
    """The report words the coverage; the tool publishes only the numbers."""
    out = await _collect(
        [{"final_answer": _unit()}, {}, _failed()],
        emitted_count=5,
        unit_noun="findings",
    )

    assert out["coverage"] == {
        "emitted": 5,
        "dispatched": 3,
        "completed": 2,
        "errored": 1,
        "unread": 1,
        "unit_noun": "findings",
        "reviewed_indexes": [0],
    }


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

    collected = await _collect(results)
    published = json.loads(collected["final_answer"])
    deduped = await BlDedupFindings(metadata={})._execute(collected["final_answer"])

    assert [f.to_dict() for f in _flatten_findings(published)] == published
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
    async def _merge(verdicts: list, items: list) -> list:
        results = [{"final_answer": json.dumps(v)} for v in verdicts]
        out = await _collect(results=results, verdict_items=items)
        return json.loads(out["final_answer"])

    async def test_each_verdict_is_merged_onto_its_own_item(self):
        other = {**_FINDING, "file": "app/d.rb"}
        drop = {**_VERDICT, "verdict": "DROP", "clause": "DROP-2"}

        merged = await self._merge([_VERDICT, drop], [dict(_FINDING), other])

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

        (merged,) = await self._merge([verdict], [stale])

        assert merged == {
            **_FINDING,
            "verdict": "DROP",
            "clause": _VERDICT["clause"],
            "evidence": "new",
            "triage_evidence": _VERDICT["triage_evidence"],
        }

    async def test_an_unset_optional_field_is_not_merged(self):
        verdict = {**_VERDICT, "verdict": "DROP", "triage_evidence": None}

        (merged,) = await self._merge([verdict], [dict(_FINDING)])

        assert "triage_evidence" not in merged
        assert merged["verdict"] == "DROP"

    async def test_a_json_string_item_is_merged_too(self):
        (merged,) = await self._merge([_VERDICT], [json.dumps(_FINDING)])

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
            merged = await self._merge([verdict], [item])

        assert merged == [{"verdict": "DROP", "clause": "c", "evidence": "e"}]
        assert any("item is not a finding" in log["event"] for log in logs)

    async def test_a_failed_unit_does_not_shift_the_items(self):
        results = [
            _failed(),
            {"final_answer": json.dumps(_VERDICT)},
        ]
        out = await _collect(
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

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            pytest.param(
                _unit(_ONE_FINDING, {"cwe": "CWE-862"}),
                [_ONE_FINDING, {"cwe": "CWE-862"}],
                id="the-dicts-under-findings",
            ),
            # A unit that honestly found nothing; re-asking would pay for a
            # second model run to be told the same.
            pytest.param(_unit(), [], id="empty-findings-is-an-answer"),
            pytest.param(
                json.loads(_unit(_ONE_FINDING)), [_ONE_FINDING], id="already-decoded"
            ),
            pytest.param(
                json.dumps({"explanation": "r", "findings": [_ONE_FINDING, "x", None]}),
                [_ONE_FINDING],
                id="non-dicts-beside-a-finding-are-skipped",
            ),
        ],
    )
    def test_a_findings_list_is_read(self, answer, expected):
        assert _unit_findings(answer) == expected

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param("prose, not the answer tool", id="prose"),
            pytest.param(json.dumps([_ONE_FINDING]), id="bare-array"),
            pytest.param(json.dumps({"explanation": "r"}), id="no-findings-key"),
            pytest.param(json.dumps({"findings": None}), id="null-findings"),
            pytest.param(json.dumps({"findings": "none"}), id="string-findings"),
            pytest.param(None, id="none"),
            pytest.param(
                json.dumps({"explanation": "r", "findings": [1, "prose"]}),
                id="no-finding-dict-in-findings",
            ),
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

        out = await _collect(results)

        assert json.loads(out["final_answer"]) == [_ONE_FINDING, {"cwe": "CWE-862"}]

    async def test_unread_units_are_counted_beside_the_readable_ones(self):
        results = [
            {"final_answer": "prose"},
            {"final_answer": {"explanation": "r"}},
            {"final_answer": "Result: " + _unit(_ONE_FINDING)},
            {"final_answer": _unit(_ONE_FINDING)},
            {"final_answer": _unit()},
        ]

        out = await _collect(results)

        assert json.loads(out["final_answer"]) == [_ONE_FINDING]
        assert out["coverage"]["completed"] == 5
        assert out["coverage"]["unread"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "verdict_items"),
    [
        pytest.param({}, None, id="no-answer"),
        pytest.param({"final_answer": None}, None, id="null-answer"),
        pytest.param({"final_answer": {}}, None, id="empty-dict"),
        pytest.param({"final_answer": {"explanation": "r"}}, None, id="no-findings"),
        pytest.param({"final_answer": "prose"}, None, id="prose"),
        pytest.param(
            {"final_answer": '{"explanation": "r", "findings": [{'},
            None,
            id="truncated-json",
        ),
        pytest.param(
            {"final_answer": "Result: " + _unit(_ONE_FINDING)},
            None,
            id="json-wrapped-in-text",
        ),
        pytest.param(
            {"final_answer": '["a", "list"]'}, [_FINDING], id="verdict-not-object"
        ),
        pytest.param({"final_answer": "not json"}, [_FINDING], id="verdict-not-json"),
        pytest.param(
            {"final_answer": json.dumps({"explanation": "x", "error": "oops"})},
            [_FINDING],
            id="verdict-missing",
        ),
    ],
)
async def test_a_completed_unit_with_no_readable_answer_is_counted_unread(
    entry, verdict_items
):
    """What counts as unread: a unit that completed, but whose answer is not a findings list we can read."""
    out = await _collect([entry], verdict_items=verdict_items)

    assert json.loads(out["final_answer"]) == []
    assert out["coverage"] == {
        "emitted": None,
        "dispatched": 1,
        "completed": 1,
        "errored": 0,
        "unread": 1,
        "unit_noun": None,
        "reviewed_indexes": [],
    }
