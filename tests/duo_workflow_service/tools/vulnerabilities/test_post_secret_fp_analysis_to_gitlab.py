import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.vulnerabilities.post_secret_fp_analysis_to_gitlab import (
    PostSecretFpAnalysisToGitlab,
    PostSecretFpAnalysisToGitlabInput,
)


@pytest.fixture
def gitlab_client_mock():
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
    }


@pytest.fixture
def success_response_mock():
    return GitLabHttpResponse(
        status_code=200,
        body={
            "id": 123,
            "vulnerability_id": 567,
            "flag_type": "ai_detection",
            "confidence_score": 85.0,
            "description": "This appears to be a false positive because it's test data.",
            "created_at": "2023-10-01T12:00:00Z",
        },
    )


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_success(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "This appears to be a false positive because it's test data.",
    }

    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["vulnerability_id"] == 567
    assert response_data["false_positive_likelihood"] == 85.0
    assert response_data["response"] == success_response_mock.body

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    assert call_args[1]["path"] == "/api/v4/vulnerabilities/567/flags/ai_detection"

    body = json.loads(call_args[1]["body"])
    assert body["confidence_score"] == 85.0
    assert (
        body["description"]
        == "This appears to be a false positive because it's test data."
    )
    assert body["origin"] == "ai_secret_detection_fp_detection"
    assert "detection_type" not in body


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_with_zero_likelihood(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 0.0,
        "explanation": "Low confidence false positive.",
    }

    response = await tool.arun(input_data)
    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["false_positive_likelihood"] == 0.0

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    assert call_args[1]["path"] == "/api/v4/vulnerabilities/567/flags/ai_detection"
    body = json.loads(call_args[1]["body"])
    assert body["confidence_score"] == 0.0
    assert body["description"] == "Low confidence false positive."
    assert body["origin"] == "ai_secret_detection_fp_detection"


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_with_max_likelihood(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 100.0,
        "explanation": "High confidence false positive.",
    }

    response = await tool.arun(input_data)
    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["false_positive_likelihood"] == 100.0

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    assert call_args[1]["path"] == "/api/v4/vulnerabilities/567/flags/ai_detection"
    body = json.loads(call_args[1]["body"])
    assert body["confidence_score"] == 100.0
    assert body["description"] == "High confidence false positive."
    assert body["origin"] == "ai_secret_detection_fp_detection"


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_api_error_with_status(
    gitlab_client_mock, metadata
):
    error_response = GitLabHttpResponse(
        status_code=404,
        body={"message": "Vulnerability not found"},
    )

    gitlab_client_mock.apost = AsyncMock(return_value=error_response)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 999,
        "false_positive_likelihood": 85.0,
        "explanation": "Analysis for non-existent vulnerability.",
    }

    # Should raise ToolException instead of returning error JSON
    with pytest.raises(ToolException) as exc_info:
        await tool.arun(input_data)

    assert "HTTP 404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network Error"))

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "This appears to be a false positive.",
    }

    with pytest.raises(Exception, match="Network Error"):
        await tool.arun(input_data)


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_with_201_status(
    gitlab_client_mock, metadata, success_response_mock
):
    success_response_mock.status_code = 201
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "False positive analysis.",
    }

    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["status"] == "success"


def test_post_secret_fp_analysis_format_display_message():
    tool = PostSecretFpAnalysisToGitlab(metadata={})
    input_data = PostSecretFpAnalysisToGitlabInput(
        vulnerability_id=567,
        false_positive_likelihood=85.0,
        explanation="This appears to be a false positive.",
    )
    expected_message = (
        "Post secret false positive analysis for vulnerability 567 "
        "(false_positive_likelihood: 85.0%)"
    )
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_post_secret_fp_analysis_with_long_explanation(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSecretFpAnalysisToGitlab(metadata=metadata)

    long_explanation = (
        "This is a detailed analysis explaining why this secret vulnerability "
        "is likely a false positive. " * 10
    )

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 75.5,
        "explanation": long_explanation,
    }

    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["status"] == "success"

    call_args = gitlab_client_mock.apost.call_args
    body = json.loads(call_args[1]["body"])
    assert body["description"] == long_explanation
