import json
from unittest.mock import AsyncMock, call

import pytest

from duo_workflow_service.tools.pipeline import (
    GetPipelineErrorsForMergeRequest,
    GetPipelineErrorsInput,
    PipelineMergeRequestNotFoundError,
    PipelinesNotFoundError,
)


@pytest.mark.asyncio
async def test_get_pipeline_errors():
    responses = [
        {"id": 1, "title": "Merge Request 1"},
        [{"id": 10, "status": "success"}, {"id": 11, "status": "failed"}],
        [
            {"id": 101, "name": "job1", "status": "success"},
            {"id": 102, "name": "job2", "status": "failed"},
        ],
        "Job 102 trace log",
    ]

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = responses
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": "1", "merge_request_id": "1"})

    # Validate the output
    expected_response = {
        "merge_request": {"id": 1, "title": "Merge Request 1"},
        "traces": "Failed Jobs:\nName: job2\nJob ID: 102\nTrace: Job 102 trace log\n",
    }

    assert json.loads(response) == expected_response

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
        call(path="/api/v4/projects/1/pipelines/10/jobs"),
        call(path="/api/v4/projects/1/jobs/102/trace", parse_json=False),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_merge_request_not_found():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = [{"status": 404}]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)  # type: ignore

    with pytest.raises(PipelineMergeRequestNotFoundError):
        await tool.arun({"project_id": "1", "merge_request_id": "1"})

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1")
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_pipelines_not_found():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = [
        {"id": 1, "title": "Merge Request 1"},
        {"status": 404},
    ]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)  # type: ignore

    with pytest.raises(PipelinesNotFoundError):
        await tool.arun({"project_id": "1", "merge_request_id": "1"})

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
    ]


def test_get_pipeline_errors_format_display_message():
    tool = GetPipelineErrorsForMergeRequest(
        description="Get pipeline errors description"
    )

    input_data = GetPipelineErrorsInput(project_id=123, merge_request_id=456)

    message = tool.format_display_message(input_data)

    expected_message = "Get pipeline error logs for merge request !456 in project 123"
    assert message == expected_message
