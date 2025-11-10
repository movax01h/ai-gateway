import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.vulnerabilities.post_sast_fp_analysis_to_gitlab import (
    PostSastFpAnalysisToGitlab,
    PostSastFpAnalysisToGitlabInput,
)


@pytest.fixture
def gitlab_client_mock():
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture
def success_response_mock():
    response = Mock()
    response.is_success.return_value = True
    response.status_code = 200
    response.body = {
        "id": 123,
        "vulnerability_id": 567,
        "flag_type": "ai_detection",
        "confidence_score": 85.0,
        "description": "This appears to be a false positive because the input is not user-controlled.",
        "created_at": "2023-10-01T12:00:00Z",
    }
    return response


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_success(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "This appears to be a false positive because the input is not user-controlled.",
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
        == "This appears to be a false positive because the input is not user-controlled."
    )


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_with_boundary_values(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 0.0,
        "explanation": "Low confidence false positive.",
    }

    response = await tool.arun(input_data)
    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["false_positive_likelihood"] == 0.0

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 100.0,
        "explanation": "High confidence false positive.",
    }

    response = await tool.arun(input_data)
    response_data = json.loads(response)
    assert response_data["status"] == "success"
    assert response_data["false_positive_likelihood"] == 100.0


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_api_error_with_status(
    gitlab_client_mock, metadata
):
    error_response = Mock()
    error_response.is_success.return_value = False
    error_response.status_code = 404
    error_response.body = {"message": "Vulnerability not found"}

    gitlab_client_mock.apost = AsyncMock(return_value=error_response)

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 999,
        "false_positive_likelihood": 85.0,
        "explanation": "Analysis for non-existent vulnerability.",
    }

    response = await tool.arun(input_data)

    error_response_data = json.loads(response)
    assert "error" in error_response_data
    assert "Unexpected status code: 404" in error_response_data["error"]


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network Error"))

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "This appears to be a false positive.",
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Failed to post SAST false positive analysis" in error_response["error"]
    assert "vulnerability 567" in error_response["error"]
    assert "Network Error" in error_response["error"]


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_with_201_status(
    gitlab_client_mock, metadata, success_response_mock
):
    success_response_mock.status_code = 201
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    input_data = {
        "vulnerability_id": 567,
        "false_positive_likelihood": 85.0,
        "explanation": "False positive analysis.",
    }

    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["status"] == "success"


def test_post_sast_fp_analysis_format_display_message():
    tool = PostSastFpAnalysisToGitlab(metadata={})
    input_data = PostSastFpAnalysisToGitlabInput(
        vulnerability_id=567,
        false_positive_likelihood=85.0,
        explanation="This appears to be a false positive.",
    )
    expected_message = (
        "Post SAST false positive analysis for vulnerability 567 "
        "(false_positive_likelihood: 85.0%)"
    )
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_post_sast_fp_analysis_with_long_explanation(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSastFpAnalysisToGitlab(metadata=metadata)

    long_explanation = (
        "This is a detailed analysis explaining why this vulnerability "
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
