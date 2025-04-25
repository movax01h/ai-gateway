import json
from unittest.mock import AsyncMock, Mock, call

import pytest

from duo_workflow_service.tools.pipeline import (
    GetPipelineErrorsForMergeRequest,
    GetPipelineErrorsInput,
    PipelineMergeRequestNotFoundError,
    PipelinesNotFoundError,
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


@pytest.mark.asyncio
async def test_get_pipeline_errors(gitlab_client_mock, metadata):
    responses = [
        {"id": 1, "title": "Merge Request 1"},
        [{"id": 10, "status": "success"}, {"id": 11, "status": "failed"}],
        [
            {"id": 101, "name": "job1", "status": "success"},
            {"id": 102, "name": "job2", "status": "failed"},
        ],
        "Job 102 trace log",
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})

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
async def test_get_pipeline_errors_merge_request_not_found(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(return_value={"status": 404})

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    with pytest.raises(PipelineMergeRequestNotFoundError):
        await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/1"
    )


@pytest.mark.asyncio
async def test_get_pipeline_errors_pipelines_not_found(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            {"id": 1, "title": "Merge Request 1"},
            {"status": 404},
        ]
    )

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    with pytest.raises(PipelinesNotFoundError):
        await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
        ),
        # Test with URL and matching project_id and merge_request_iid
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
        ),
    ],
)
async def test_get_pipeline_errors_with_url_success(
    url, project_id, merge_request_iid, expected_path, gitlab_client_mock, metadata
):
    merge_request_response = {"id": 1, "title": "Merge Request 1"}
    pipelines_response = [
        {"id": 10, "status": "success"},
        {"id": 11, "status": "failed"},
    ]
    jobs_response = [
        {"id": 101, "name": "job1", "status": "success"},
        {"id": 102, "name": "job2", "status": "failed"},
    ]
    trace_response = "Job 102 trace log"

    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.aget.side_effect = [
        merge_request_response,
        pipelines_response,
        jobs_response,
        trace_response,
    ]

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    response = await tool._arun(
        url=url, project_id=project_id, merge_request_iid=merge_request_iid
    )

    expected_response = json.dumps(
        {
            "merge_request": {"id": 1, "title": "Merge Request 1"},
            "traces": "Failed Jobs:\nName: job2\nJob ID: 102\nTrace: Job 102 trace log\n",
        }
    )
    assert response == expected_response

    assert gitlab_client_mock.aget.call_args_list[0] == call(path=expected_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    [
        # URL and project_id both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "different%2Fproject",
            123,
            "Project ID mismatch",
        ),
        # URL and merge_request_iid both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            456,
            "Merge Request ID mismatch",
        ),
        # URL given isn't a merge request URL (it's just a project URL)
        (
            "https://gitlab.com/namespace/project",
            None,
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_get_pipeline_errors_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    response = await tool._arun(
        url=url, project_id=project_id, merge_request_iid=merge_request_iid
    )
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetPipelineErrorsInput(project_id=123, merge_request_iid=456),
            "Get pipeline error logs for merge request !456 in project 123",
        ),
        (
            GetPipelineErrorsInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Get pipeline error logs for https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_get_pipeline_errors_format_display_message(input_data, expected_message):
    tool = GetPipelineErrorsForMergeRequest(
        description="Get pipeline errors description"
    )
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project_id,merge_request_iid,expected_errors",
    [
        # Both project_id and merge_request_iid are missing
        (
            None,
            None,
            [
                "'project_id' must be provided when 'url' is not",
                "'merge_request_iid' must be provided when 'url' is not",
            ],
        ),
        # Only project_id is missing
        (
            None,
            123,
            ["'project_id' must be provided when 'url' is not"],
        ),
        # Only merge_request_iid is missing
        (
            "namespace%2Fproject",
            None,
            ["'merge_request_iid' must be provided when 'url' is not"],
        ),
    ],
)
async def test_validate_merge_request_url_missing_params(
    project_id, merge_request_iid, expected_errors, gitlab_client_mock, metadata
):
    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    response = await tool._arun(
        project_id=project_id, merge_request_iid=merge_request_iid
    )
    response_json = json.loads(response)

    assert "error" in response_json
    for error in expected_errors:
        assert error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_pipeline_errors_trace_exception(gitlab_client_mock, metadata):
    # Set up mock responses
    merge_request_response = {"id": 1, "title": "Merge Request 1"}
    pipelines_response = [
        {"id": 10, "status": "success"},
        {"id": 11, "status": "failed"},
    ]
    jobs_response = [
        {"id": 101, "name": "job1", "status": "success"},
        {"id": 102, "name": "job2", "status": "failed"},
    ]

    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.aget.side_effect = [
        merge_request_response,
        pipelines_response,
        jobs_response,
        Exception("Trace error"),
    ]

    tool = GetPipelineErrorsForMergeRequest(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "merge_request" in response_json
    assert "traces" in response_json
    assert "Error fetching trace: Trace error" in response_json["traces"]

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
        call(path="/api/v4/projects/1/pipelines/10/jobs"),
        call(path="/api/v4/projects/1/jobs/102/trace", parse_json=False),
    ]
