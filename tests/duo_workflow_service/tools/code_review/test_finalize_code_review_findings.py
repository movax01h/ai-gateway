import json
from unittest.mock import AsyncMock, Mock

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.components.deterministic_step.validation import (
    select_validated_tool,
)
from duo_workflow_service.agent_platform.v1.state import IOKey
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.code_review.finalize_code_review_findings import (
    FinalizeCodeReviewFindings,
    FinalizeCodeReviewFindingsInput,
)
from duo_workflow_service.tools.code_review.findings import build_summary
from duo_workflow_service.tools.code_review.post_duo_code_review_findings import (
    PostDuoCodeReviewFindings,
)
from duo_workflow_service.tools.tool_output_manager import TruncationConfig
from duo_workflow_service.workflows.type_definitions import MAX_MESSAGE_SIZE

FINDING = {
    "file": "app/models/user.rb",
    "new_line": 42,
    "end_line": 42,
    "target_code": "  return true",
    "severity": "critical",
    "category": "fail-open",
    "message": "Fails open when the check errors.",
    "confidence": 9,
}

# The name #2928's config is expected to give the step; the client reads under it until
# flow.outputs binds a stable name.
STEP_NAME = "finalize_review"


def finding(**overrides):
    return {**FINDING, **overrides}


@pytest.fixture(name="untouched_metadata")
def untouched_metadata_fixture():
    return {"workflow_id": "wf-1", "gitlab_client": Mock(), "outbox": Mock()}


@pytest.fixture(name="tool")
def tool_fixture(untouched_metadata):
    return FinalizeCodeReviewFindings(metadata=untouched_metadata)


class TestInputCoercion:
    @pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("7", 7), (5, 5)])
    def test_min_confidence_accepts_flow_config_strings(self, raw, expected):
        assert (
            FinalizeCodeReviewFindingsInput(min_confidence=raw).min_confidence
            == expected
        )

    def test_everything_is_optional_so_a_missing_answer_is_a_clean_review(self):
        args = FinalizeCodeReviewFindingsInput()

        assert args.findings == []
        assert args.summary is None
        assert args.min_confidence == 0

    def test_a_deterministic_step_can_bind_exactly_the_flow_inputs(self, tool):
        """What #2928's YAML will declare under `inputs:` for this step."""
        assert (
            select_validated_tool(
                tool, tool.name, {"findings", "summary", "min_confidence"}
            )
            is tool
        )

    def test_a_deterministic_step_cannot_bind_merge_request_inputs(self, tool):
        with pytest.raises(ValueError, match="Unknown parameters"):
            select_validated_tool(
                tool, tool.name, {"findings", "project_id", "merge_request_iid"}
            )


@pytest.mark.asyncio
async def test_applies_the_gate_and_orders_by_severity(tool):
    # ainvoke, not _arun: the flow passes min_confidence as a string literal and the
    # args schema is what coerces it.
    result = await tool.ainvoke(
        {
            "findings": [
                finding(new_line=1, severity="minor", confidence=8),
                finding(new_line=2, severity="critical", confidence=2),
                finding(new_line=3, severity="major", confidence=6),
                finding(new_line=4, severity="critical", confidence=9),
            ],
            "summary": "I focused on the auth changes.",
            "min_confidence": "5",
        }
    )

    assert [(f["severity"], f["new_line"]) for f in result["findings"]] == [
        ("critical", 4),
        ("major", 3),
        ("minor", 1),
    ]
    assert result["published"] == 3
    assert result["suppressed_below_threshold"] == 1
    assert result["min_confidence"] == 5
    assert result["counts"] == {"critical": 1, "major": 1, "minor": 1}
    assert result["summary"].startswith("I focused on the auth changes.")
    assert (
        "3 findings (1 critical, 1 major, 1 minor) across 1 file:" in result["summary"]
    )


@pytest.mark.asyncio
async def test_returns_a_mapping_addressable_from_flow_state(tool):
    """The interim contract: `context:<step>.tool_responses.findings` resolves without flow.outputs."""
    result = await tool.ainvoke({"findings": [FINDING], "summary": "s"})

    assert isinstance(result, dict)
    tool_responses = IOKey(target="context", subkeys=[STEP_NAME, "tool_responses"])
    state = tool_responses.to_nested_dict(result)
    findings_key = IOKey(
        target="context", subkeys=[STEP_NAME, "tool_responses", "findings"]
    )
    summary_key = IOKey(
        target="context", subkeys=[STEP_NAME, "tool_responses", "summary"]
    )

    assert findings_key.value_from_state(state) == result["findings"]
    assert summary_key.value_from_state(state) == result["summary"]
    json.dumps(result)


@pytest.mark.asyncio
async def test_summary_and_counts_come_from_the_kept_findings(tool):
    kept = [finding(new_line=1), finding(new_line=2, severity="minor", confidence=9)]

    result = await tool.ainvoke(
        {
            "findings": [*kept, finding(new_line=3, confidence=1)],
            "summary": "Narrative claims 10 findings.",
            "min_confidence": 5,
        }
    )

    assert result["summary"] == build_summary(kept, "Narrative claims 10 findings.")
    assert result["counts"] == {"critical": 1, "major": 0, "minor": 1}


@pytest.mark.asyncio
async def test_a_clean_review_is_an_empty_list_with_the_fallback_summary(tool):
    result = await tool.ainvoke({"findings": [], "summary": None})

    assert result == {
        "findings": [],
        "summary": "No issues were raised in this review.",
        "counts": {"critical": 0, "major": 0, "minor": 0},
        "published": 0,
        "suppressed_below_threshold": 0,
        "min_confidence": 0,
    }


@pytest.mark.asyncio
async def test_findings_keep_their_schema_fields_and_gain_the_attribution(tool):
    result = await tool.ainvoke(
        {
            "findings": [
                finding(custom_instruction_ref="No fail-open guards", suggestion="")
            ]
        }
    )

    [rendered] = result["findings"]
    assert rendered == {
        "file": "app/models/user.rb",
        "new_line": 42,
        "target_code": "  return true",
        "severity": "critical",
        "category": "fail-open",
        "confidence": 9,
        "custom_instruction_ref": "No fail-open guards",
        "message": (
            "According to custom instructions in 'No fail-open guards': "
            "Fails open when the check errors."
        ),
    }
    assert "**[" not in rendered["message"]


@pytest.mark.asyncio
async def test_suppressions_are_logged_one_per_drop_plus_the_aggregate(tool):
    with capture_logs() as logs:
        await tool.ainvoke(
            {
                "findings": [
                    finding(new_line=1, confidence=1),
                    finding(new_line=2, confidence=2),
                    finding(new_line=3, confidence=9),
                ],
                "min_confidence": 5,
            }
        )

    drops = [log for log in logs if log["event"].startswith("Suppressing finding")]
    assert [(d["new_line"], d["confidence"], d["threshold"]) for d in drops] == [
        (1, 1, 5),
        (2, 2, 5),
    ]
    [aggregate] = [log for log in logs if log["event"].startswith("Finalizing")]
    assert aggregate["published"] == 1
    assert aggregate["suppressed_below_threshold"] == 2
    assert aggregate["min_confidence"] == 5
    assert aggregate["critical"] == 1


@pytest.mark.asyncio
async def test_writes_nothing_outside_flow_state(tool, untouched_metadata):
    await tool.ainvoke({"findings": [FINDING], "summary": "s", "min_confidence": 3})

    assert untouched_metadata["gitlab_client"].mock_calls == []
    assert untouched_metadata["outbox"].mock_calls == []


@pytest.mark.asyncio
async def test_result_stays_a_mapping_past_the_default_tool_budget(tool):
    """The state contract holds up to the transport limit, not the 200 KiB LLM-context default."""
    default_budget = TruncationConfig().max_bytes
    message = "x" * 1000
    findings = [
        finding(new_line=i, message=message) for i in range(default_budget // 800)
    ]
    assert len(json.dumps(findings)) > default_budget

    result = await tool.ainvoke({"findings": findings})

    assert isinstance(result, dict)
    assert result["published"] == len(findings)
    assert tool.truncation_config.max_bytes == MAX_MESSAGE_SIZE


class TestDisplayMessage:
    def test_before_the_step_runs(self, tool):
        assert (
            tool.format_display_message(FinalizeCodeReviewFindingsInput())
            == "Finalize the review findings"
        )

    @pytest.mark.asyncio
    async def test_reports_what_the_step_produced(self, tool):
        result = await tool.ainvoke(
            {
                "findings": [
                    finding(new_line=1),
                    finding(new_line=2, severity="minor"),
                    finding(new_line=3, confidence=1),
                ],
                "min_confidence": 7,
            }
        )

        message = tool.format_display_message(
            FinalizeCodeReviewFindingsInput(min_confidence=7), result
        )

        assert message == (
            "Review finished: 2 findings (1 critical, 1 minor), "
            "1 suppressed below confidence 7"
        )

    @pytest.mark.asyncio
    async def test_a_clean_review_reads_as_such(self, tool):
        result = await tool.ainvoke({"findings": []})

        assert (
            tool.format_display_message(FinalizeCodeReviewFindingsInput(), result)
            == "Review finished: 0 findings"
        )

    def test_a_truncated_response_falls_back_to_the_plain_message(self, tool):
        assert (
            tool.format_display_message(
                FinalizeCodeReviewFindingsInput(), "<truncated>..."
            )
            == "Finalize the review findings"
        )


REVIEWER_ANSWER = {
    "findings": [
        finding(file="b.rb", new_line=10, severity="minor", confidence=8),
        finding(new_line=42, severity="critical", confidence=6),  # gated
        finding(
            file="c.rb",
            new_line=7,
            end_line=9,
            severity="major",
            confidence=7,
            suggestion="  fixed = true",
            old_line=5,
        ),
        finding(
            file="a.rb",
            new_line=1,
            severity="critical",
            confidence=10,
            custom_instruction_ref="Security",
            suggestion="",
        ),
        finding(file="d.rb", new_line=3, severity="major", confidence=9, end_line=3),
    ],
    "summary": "I looked at the auth changes and the migration.",
}


@pytest.mark.asyncio
async def test_local_step_matches_the_posting_tool_on_the_same_answer(
    gitlab_client_mock, metadata, untouched_metadata
):
    """The acceptance criterion: same gate, same order, same anchors, same summary, header the only delta."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=200,
            body=json.dumps({"message": "Comments added successfully"}),
        )
    )
    posting = PostDuoCodeReviewFindings(metadata=metadata)
    local = FinalizeCodeReviewFindings(metadata=untouched_metadata)

    posted_receipt = json.loads(
        await posting.ainvoke(
            {
                "project_id": 1,
                "merge_request_iid": 2,
                **REVIEWER_ANSWER,
                "min_confidence": "7",
            }
        )
    )
    local_result = await local.ainvoke({**REVIEWER_ANSWER, "min_confidence": "7"})

    body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    posted = json.loads(body["review_output"])

    assert posted_receipt["published"] == local_result["published"] == 4
    assert (
        posted_receipt["suppressed_below_threshold"]
        == local_result["suppressed_below_threshold"]
        == 1
    )
    assert posted["summary"] == local_result["summary"]

    anchor = ("file", "new_line", "severity", "confidence")
    assert [tuple(f[k] for k in anchor) for f in posted["findings"]] == [
        tuple(f[k] for k in anchor) for f in local_result["findings"]
    ]
    for posted_finding, local_finding in zip(
        posted["findings"], local_result["findings"], strict=True
    ):
        for key in ("old_line", "suggestion", "end_line"):
            assert posted_finding.get(key) == local_finding.get(key)
            assert (key in posted_finding) == (key in local_finding)
        header = f"**[{local_finding['severity'].capitalize()}] {local_finding['category']}**"
        assert posted_finding["message"] == f"{header}\n\n{local_finding['message']}"
