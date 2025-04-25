import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.job import GetLogsFromJob, GetLogsFromJobInput


@pytest.fixture
def gitlab_client_mock():
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,job_id,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project/-/jobs/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/jobs/123/trace",
        ),
        # Test with URL and matching project_id and job_id
        (
            "https://gitlab.com/namespace/project/-/jobs/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/jobs/123/trace",
        ),
    ],
)
async def test_get_job_logs_with_url_success(
    url, project_id, job_id, expected_path, gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(return_value="Job 123 trace log")

    tool = GetLogsFromJob(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, job_id=job_id)

    expected_response = json.dumps(
        {
            "job_id": 123,
            "trace": "Job 123 trace log",
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
async def test_get_job_logs(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(return_value="Job 1 trace log")

    tool = GetLogsFromJob(metadata=metadata)

    response = await tool.arun({"project_id": "1", "job_id": "1"})

    # Validate the output
    expected_response = {
        "job_id": 1,
        "trace": "Job 1 trace log",
    }

    assert json.loads(response) == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/jobs/1/trace", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_job_logs_for_invalid_job(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(return_value="No job found")

    tool = GetLogsFromJob(metadata=metadata)

    response = await tool.arun({"project_id": "1", "job_id": "123"})

    expected_response = {"job_id": 123, "trace": "No job found"}

    assert json.loads(response) == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/jobs/123/trace", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_job_logs_empty_trace(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(return_value="")

    tool = GetLogsFromJob(metadata=metadata)

    response = await tool._arun(project_id="1", job_id="123")

    assert response == "No job found"

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/jobs/123/trace", parse_json=False
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetLogsFromJobInput(project_id=123, job_id=456),
            "Get logs for job #456 in project 123",
        ),
        (
            GetLogsFromJobInput(url="https://gitlab.com/namespace/project/-/jobs/42"),
            "Get logs for https://gitlab.com/namespace/project/-/jobs/42",
        ),
    ],
)
def test_get_job_logs_format_display_message(input_data, expected_message):
    tool = GetLogsFromJob(description="Get job logs description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,job_id,error_contains",
    [
        # URL and project_id both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/jobs/123",
            "different%2Fproject",
            123,
            "Project ID mismatch",
        ),
        # URL and job_id both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/jobs/123",
            "namespace%2Fproject",
            456,
            "Job ID mismatch",
        ),
        # URL given isn't a job URL (it's just a project URL)
        (
            "https://gitlab.com/namespace/project",
            None,
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_get_job_logs_with_url_error(
    url, project_id, job_id, error_contains, gitlab_client_mock, metadata
):
    tool = GetLogsFromJob(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, job_id=job_id)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project_id,job_id,expected_errors",
    [
        # Both project_id and job_id are missing
        (
            None,
            None,
            [
                "'project_id' must be provided when 'url' is not",
                "'job_id' must be provided when 'url' is not",
            ],
        ),
        # Only project_id is missing
        (
            None,
            123,
            ["'project_id' must be provided when 'url' is not"],
        ),
        # Only job_id is missing
        (
            "namespace%2Fproject",
            None,
            ["'job_id' must be provided when 'url' is not"],
        ),
    ],
)
async def test_validate_job_url_missing_params(
    project_id, job_id, expected_errors, gitlab_client_mock, metadata
):
    tool = GetLogsFromJob(metadata=metadata)

    response = await tool._arun(project_id=project_id, job_id=job_id)
    response_json = json.loads(response)

    assert "error" in response_json
    for error in expected_errors:
        assert error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_job_logs_api_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("API error"))

    tool = GetLogsFromJob(metadata=metadata)

    response = await tool._arun(project_id="1", job_id="123")
    response_json = json.loads(response)

    assert "error" in response_json
    assert "API error" in response_json["error"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/jobs/123/trace", parse_json=False
    )
