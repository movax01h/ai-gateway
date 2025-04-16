import json
from unittest.mock import AsyncMock, call

import pytest

from duo_workflow_service.tools.job import GetLogsFromJob, GetLogsFromJobInput


@pytest.mark.asyncio
async def test_get_job_logs():
    responses = [
        "Job 1 trace log",
    ]

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = responses
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetLogsFromJob(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": "1", "job_id": "1"})

    # Validate the output
    expected_response = {
        "job_id": 1,
        "trace": "Job 1 trace log",
    }

    assert json.loads(response) == expected_response

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/jobs/1/trace", parse_json=False),
    ]


@pytest.mark.asyncio
async def test_get_job_logs_for_invalid_job():
    responses = ["No job found"]

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = responses
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetLogsFromJob(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": "1", "job_id": "123"})

    expected_response = {"job_id": 123, "trace": "No job found"}

    assert json.loads(response) == expected_response

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/jobs/123/trace", parse_json=False),
    ]


def test_get_job_logs_format_display_message():
    tool = GetLogsFromJob(description="Get job logs description")

    input_data = GetLogsFromJobInput(project_id=123, job_id=456)

    message = tool.format_display_message(input_data)

    expected_message = "Get logs for job #456 in project 123"
    assert message == expected_message
