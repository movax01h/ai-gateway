import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.risk_classification.submit_merge_request_risk_classification import (
    RESULTS_PATH,
    SubmitMergeRequestRiskClassification,
    SubmitMergeRequestRiskClassificationInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


INPUT_DATA = {
    "project_id": 13,
    "merge_request_iid": 9,
    "claims": [
        {"name": "touches_auth", "value": "true", "evidence": "lib/auth.rb:44"},
        {"name": "change_kind", "value": "behavioral", "evidence": None},
    ],
    "summary": "Adds a session token refresh path.",
}


@pytest.mark.asyncio
async def test_submit_risk_classification_success(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=204, body=None)
    )

    tool = SubmitMergeRequestRiskClassification(metadata=metadata)
    response = await tool.arun(INPUT_DATA)

    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["project_id"] == 13
    assert response_data["merge_request_iid"] == 9

    gitlab_client_mock.apost.assert_called_once_with(
        path=RESULTS_PATH,
        body=json.dumps(
            {
                "project_id": 13,
                "merge_request_iid": 9,
                "claims": INPUT_DATA["claims"],
                "summary": INPUT_DATA["summary"],
            }
        ),
    )


@pytest.mark.asyncio
async def test_submit_risk_classification_success_with_empty_claims(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=204, body=None)
    )

    tool = SubmitMergeRequestRiskClassification(metadata=metadata)
    input_data = {**INPUT_DATA, "claims": []}
    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["status"] == "success"

    gitlab_client_mock.apost.assert_called_once_with(
        path=RESULTS_PATH,
        body=json.dumps(
            {
                "project_id": 13,
                "merge_request_iid": 9,
                "claims": [],
                "summary": INPUT_DATA["summary"],
            }
        ),
    )


@pytest.mark.parametrize(
    "status_code,body",
    [
        # forbidden!/not_found! shape a "message" key
        (
            403,
            {"message": "This endpoint can only be accessed by Duo Workflow Service"},
        ),
        (404, {"message": "404 Project Not Found"}),
        # Grape's own param validation failures shape an "error" key instead
        (400, {"error": "claims is missing, claims[0][name] is missing"}),
    ],
)
@pytest.mark.asyncio
async def test_submit_risk_classification_failure_response(
    gitlab_client_mock, metadata, status_code, body
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=status_code, body=body)
    )

    tool = SubmitMergeRequestRiskClassification(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool.arun(INPUT_DATA)

    # The endpoint isn't consistent about which key carries the error text, so
    # the exception must surface the body regardless of which key was used.
    assert str(status_code) in str(exc_info.value)
    assert next(iter(body.values())) in str(exc_info.value)


@pytest.mark.asyncio
async def test_submit_risk_classification_exception_propagates(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network error"))

    tool = SubmitMergeRequestRiskClassification(metadata=metadata)

    with pytest.raises(Exception, match="Network error"):
        await tool.arun(INPUT_DATA)


def test_submit_risk_classification_format_display_message():
    tool = SubmitMergeRequestRiskClassification(metadata={})
    input_data = SubmitMergeRequestRiskClassificationInput(**INPUT_DATA)

    expected_message = (
        "Submit risk classification for merge request !9 in project 13 (2 claim(s))"
    )
    assert tool.format_display_message(input_data) == expected_message


def test_submit_risk_classification_format_display_message_appends_error():
    # The framework passes the raised exception's str() as tool_response on
    # failure (ToolNodeWithErrorCorrection._execute_tool) -- it must reach the
    # UI chat log instead of being silently dropped.
    tool = SubmitMergeRequestRiskClassification(metadata={})
    input_data = SubmitMergeRequestRiskClassificationInput(**INPUT_DATA)

    message = tool.format_display_message(
        input_data, "Insufficient permissions to create a new pipeline"
    )

    assert message == (
        "Submit risk classification for merge request !9 in project 13 (2 claim(s)) "
        "-- Insufficient permissions to create a new pipeline"
    )


def test_submit_risk_classification_format_display_message_omits_success_payload():
    # On success, tool_response is this tool's own JSON return value, which is
    # redundant with base_msg -- it should not be appended.
    tool = SubmitMergeRequestRiskClassification(metadata={})
    input_data = SubmitMergeRequestRiskClassificationInput(**INPUT_DATA)

    success_payload = json.dumps(
        {"status": "success", "project_id": 13, "merge_request_iid": 9}
    )
    message = tool.format_display_message(input_data, success_payload)

    expected_message = (
        "Submit risk classification for merge request !9 in project 13 (2 claim(s))"
    )
    assert message == expected_message
