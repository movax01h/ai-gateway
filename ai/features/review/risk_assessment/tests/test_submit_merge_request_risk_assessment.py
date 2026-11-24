import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError
from structlog.testing import capture_logs

from ai.features.review.risk_assessment.components.submit_merge_request_risk_assessment import (
    RESULTS_PATH,
    SubmitMergeRequestRiskAssessment,
    SubmitMergeRequestRiskAssessmentInput,
)
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse

DIFF_SHA = "b83d6e391c22777fca1ed3012fce84f633d7fed0"

INPUT_DATA = {
    "project_id": 13,
    "merge_request_iid": 9,
    "diff_sha": DIFF_SHA,
    "areas": [
        {"name": "db", "level": "high", "reason": "db/migrate/x.rb drops a column."},
        {"name": "docs", "level": "low", "reason": "doc/user/index.md: wording only."},
    ],
}


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    client = Mock()
    client.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=204, body=None)
    )
    return client


@pytest.fixture(name="tool")
def tool_fixture(gitlab_client):
    return SubmitMergeRequestRiskAssessment(
        metadata={
            "gitlab_client": gitlab_client,
            "gitlab_host": "gitlab.com",
            "workflow_id": "4242",
        }
    )


@pytest.mark.asyncio
async def test_posts_the_assessment_unchanged(tool, gitlab_client):
    response = json.loads(await tool.arun(INPUT_DATA))

    assert response == {"status": "success", "project_id": 13, "merge_request_iid": 9}
    gitlab_client.apost.assert_awaited_once()
    kwargs = gitlab_client.apost.call_args.kwargs
    assert kwargs["path"] == RESULTS_PATH
    assert json.loads(kwargs["body"]) == {**INPUT_DATA, "workflow_id": 4242}


@pytest.mark.asyncio
async def test_posts_assessments_passed_as_dicts(tool, gitlab_client):
    await tool._arun(**INPUT_DATA)

    body = json.loads(gitlab_client.apost.call_args.kwargs["body"])
    assert body["areas"] == INPUT_DATA["areas"]


@pytest.mark.asyncio
async def test_raises_when_gitlab_rejects_the_assessment(tool, gitlab_client):
    gitlab_client.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=409, body={"message": "stale head SHA"}
        )
    )

    with pytest.raises(ToolException):
        await tool._arun(**INPUT_DATA)


@pytest.mark.asyncio
async def test_logs_and_propagates_a_client_exception(tool, gitlab_client):
    gitlab_client.apost = AsyncMock(side_effect=ConnectionError("connection reset"))

    with capture_logs() as logs:
        with pytest.raises(ConnectionError, match="connection reset"):
            await tool._arun(**INPUT_DATA)

    error = next(
        entry
        for entry in logs
        if entry["event"] == "submit_merge_request_risk_assessment: apost() raised"
    )
    assert error["exception_class"] == "ConnectionError"
    assert error["areas_count"] == 2
    assert "exception_repr" not in error


@pytest.mark.asyncio
async def test_raises_without_a_workflow_id(gitlab_client):
    tool = SubmitMergeRequestRiskAssessment(
        metadata={"gitlab_client": gitlab_client, "gitlab_host": "gitlab.com"}
    )

    with pytest.raises(ValueError, match="workflow_id"):
        await tool._arun(**INPUT_DATA)

    gitlab_client.apost.assert_not_awaited()


@pytest.mark.parametrize("level", ["severe", "none", ""])
def test_rejects_a_level_outside_the_four(level):
    areas = [{"name": "docs", "level": level, "reason": "x"}]

    with pytest.raises(ValidationError):
        SubmitMergeRequestRiskAssessmentInput(**{**INPUT_DATA, "areas": areas})


@pytest.mark.parametrize(
    "tool_response,expected_suffix",
    [
        (None, ""),
        (json.dumps({"status": "success"}), ""),
        ("Error: 409 stale head SHA", " -- Error: 409 stale head SHA"),
    ],
)
def test_format_display_message(tool, tool_response, expected_suffix):
    args = SubmitMergeRequestRiskAssessmentInput(**INPUT_DATA)

    message = tool.format_display_message(args, tool_response)

    assert message == (
        "Submit risk assessment for merge request !9 in project 13 (2 area(s))"
        + expected_suffix
    )
