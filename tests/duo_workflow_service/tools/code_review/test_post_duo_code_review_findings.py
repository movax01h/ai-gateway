import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.code_review.post_duo_code_review_findings import (
    PostDuoCodeReviewFindings,
    PostDuoCodeReviewFindingsInput,
)

FINDING = {
    "file": "app/models/user.rb",
    "new_line": 42,
    "target_code": "  return true",
    "severity": "critical",
    "category": "fail-open",
    "message": "Fails open when the check errors.",
    "confidence": 9,
}


def finding(**overrides):
    return {**FINDING, **overrides}


def previous(status, file="app/models/user.rb", note="Nil check on `user`."):
    return {"file": file, "status": status, "note": note}


def success_response():
    return GitLabHttpResponse(
        status_code=200, body=json.dumps({"message": "Comments added successfully"})
    )


def posted_review(gitlab_client_mock):
    body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    return body, json.loads(body["review_output"])


class TestInputCoercion:
    @pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("7", 7), (5, 5)])
    def test_min_confidence_accepts_flow_config_strings(self, raw, expected):
        args = PostDuoCodeReviewFindingsInput(
            project_id=1, merge_request_iid=2, min_confidence=raw
        )

        assert args.min_confidence == expected

    def test_summary_may_be_absent(self):
        args = PostDuoCodeReviewFindingsInput(project_id=1, merge_request_iid=2)

        assert args.summary is None


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_success(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    # ainvoke, not _arun: the flow passes min_confidence as a string literal and the
    # args schema is what coerces it.
    response = await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [finding(new_line=1, severity="minor"), finding(new_line=2)],
            "summary": "I focused on the auth changes.",
            "min_confidence": "0",
        }
    )

    assert json.loads(response) == {
        "status": "success",
        "message": "Review posted to MR !45",
        "published": 2,
        "suppressed_below_threshold": 0,
    }
    body, review = posted_review(gitlab_client_mock)
    assert body["project_id"] == 123
    assert body["merge_request_iid"] == 45
    assert body["workflow_id"] == "test-workflow-123"
    # Critical publishes before minor, and the summary carries narrative plus computed counts.
    assert [f["new_line"] for f in review["findings"]] == [2, 1]
    assert review["summary"].startswith("I focused on the auth changes.")
    assert "2 findings (1 critical, 1 minor) across 1 file:" in review["summary"]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_renders_the_comment_body(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123,
        merge_request_iid=45,
        findings=[finding(custom_instruction_ref="No fail-open guards")],
        summary="s",
    )

    _, review = posted_review(gitlab_client_mock)
    assert review["findings"] == [
        {
            "file": "app/models/user.rb",
            "new_line": 42,
            "target_code": "  return true",
            "message": (
                "**[Critical] fail-open**\n\n"
                "According to custom instructions in 'No fail-open guards': "
                "Fails open when the check errors."
            ),
            "severity": "critical",
            "confidence": 9,
        }
    ]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_applies_the_confidence_gate(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    response = await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [
                finding(new_line=1, confidence=2),
                finding(new_line=2, confidence=9),
            ],
            "summary": "",
            "min_confidence": "5",
        }
    )

    assert json.loads(response)["published"] == 1
    assert json.loads(response)["suppressed_below_threshold"] == 1
    _, review = posted_review(gitlab_client_mock)
    assert [f["new_line"] for f in review["findings"]] == [2]
    assert "1 finding (1 critical) across 1 file:" in review["summary"]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_clean_review_posts_summary_only(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123, merge_request_iid=45, findings=[], summary="- Well tested"
    )

    _, review = posted_review(gitlab_client_mock)
    assert review == {"findings": [], "summary": "- Well tested"}


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_accepts_a_null_previous_findings(
    gitlab_client_mock, metadata
):
    """A first review has no previous findings, and the optional flow input arrives as None."""
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool.ainvoke(
        {
            "project_id": 123,
            "merge_request_iid": 45,
            "findings": [],
            "summary": "- Well tested",
            "previous_findings": None,
        }
    )

    _, review = posted_review(gitlab_client_mock)
    assert review == {"findings": [], "summary": "- Well tested"}


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_re_review_with_outstanding_threads(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response())
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    await tool._arun(
        project_id=123,
        merge_request_iid=45,
        findings=[],
        summary="- Nothing new",
        previous_findings=[
            previous("fixed"),
            previous("still_outstanding", file="b.rb"),
        ],
    )

    _, review = posted_review(gitlab_client_mock)
    assert review["findings"] == []
    assert review["previous_findings"].splitlines() == [
        "- **Fixed:** `app/models/user.rb`: Nil check on `user`.",
        "- **Still outstanding:** `b.rb`: Nil check on `user`.",
    ]


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_failure_raises(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=422, body=json.dumps({"message": "Validation failed"})
        )
    )
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    with pytest.raises(ToolException, match="Failed to post review"):
        await tool._arun(
            project_id=123, merge_request_iid=45, findings=[FINDING], summary="s"
        )


@pytest.mark.asyncio
async def test_post_duo_code_review_findings_non_json_response_raises(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=502, body="<html>Bad Gateway</html>"
        )
    )
    tool = PostDuoCodeReviewFindings(metadata=metadata)

    with pytest.raises(ToolException, match="unreadable response"):
        await tool._arun(
            project_id=123, merge_request_iid=45, findings=[FINDING], summary="s"
        )


class TestDisplayMessage:
    def test_format_display_message(self, metadata):
        tool = PostDuoCodeReviewFindings(metadata=metadata)
        args = PostDuoCodeReviewFindingsInput(project_id=123, merge_request_iid=45)

        message = tool.format_display_message(args)

        assert "!45" in message
        assert "123" in message
