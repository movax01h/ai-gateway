import json
from unittest.mock import AsyncMock, Mock, call

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.pipeline import (
    GetPipelineErrors,
    GetPipelineErrorsInput,
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


@pytest.mark.asyncio
async def test_get_pipeline_errors(gitlab_client_mock, metadata):
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(
            status_code=200,
            body=[{"id": 10, "status": "success"}, {"id": 11, "status": "failed"}],
        ),
        GitLabHttpResponse(
            status_code=200,
            body=[
                {"id": 101, "name": "job1", "status": "success"},
                {"id": 102, "name": "job2", "status": "failed"},
            ],
        ),
        "Job 102 trace log",
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    # Validate the output
    expected_response = {
        "merge_request": {"id": 1, "title": "Merge Request 1"},
        "traces": "Failed Jobs:\nName: job2\nJob ID: 102\nTrace: Job 102 trace log\n",
    }

    assert json.loads(response) == expected_response

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1", use_http_response=True),
        call(
            path="/api/v4/projects/1/merge_requests/1/pipelines", use_http_response=True
        ),
        call(path="/api/v4/projects/1/pipelines/10/jobs", use_http_response=True),
        call(
            path="/api/v4/projects/1/jobs/102/trace",
            parse_json=False,
        ),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_merge_request_not_found(
    gitlab_client_mock, metadata
):
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"status": 404},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Merge request with iid 1 not found" in response_json["error"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/1", use_http_response=True
    )


@pytest.mark.asyncio
async def test_get_pipeline_errors_pipelines_not_found(gitlab_client_mock, metadata):
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(status_code=200, body=[]),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "No pipelines found for merge request iid 1" in response_json["error"]

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1", use_http_response=True),
        call(
            path="/api/v4/projects/1/merge_requests/1/pipelines", use_http_response=True
        ),
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

    responses = [
        GitLabHttpResponse(status_code=200, body=merge_request_response),
        GitLabHttpResponse(status_code=200, body=pipelines_response),
        GitLabHttpResponse(status_code=200, body=jobs_response),
        trace_response,
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

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

    # Check that the first call uses use_http_response=True, but trace calls don't
    assert len(gitlab_client_mock.aget.call_args_list) >= 4
    # First three calls should use use_http_response=True
    assert gitlab_client_mock.aget.call_args_list[0][1]["use_http_response"] == True
    assert gitlab_client_mock.aget.call_args_list[1][1]["use_http_response"] == True
    assert gitlab_client_mock.aget.call_args_list[2][1]["use_http_response"] == True
    # Trace call should not use use_http_response=True
    assert "use_http_response" not in gitlab_client_mock.aget.call_args_list[3][1]


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
    tool = GetPipelineErrors(metadata=metadata)

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
    tool = GetPipelineErrors(description="Get pipeline errors description")
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
    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(
        project_id=project_id, merge_request_iid=merge_request_iid
    )
    response_json = json.loads(response)

    assert "error" in response_json
    for error in expected_errors:
        assert error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_pipeline_errors_jobs_error(gitlab_client_mock, metadata):
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(status_code=200, body=[{"id": 10, "status": "success"}]),
        GitLabHttpResponse(
            status_code=404, body={"status": 404, "message": "Jobs not found"}
        ),
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Failed to fetch jobs for pipeline 10" in response_json["error"]

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1", use_http_response=True),
        call(
            path="/api/v4/projects/1/merge_requests/1/pipelines", use_http_response=True
        ),
        call(path="/api/v4/projects/1/pipelines/10/jobs", use_http_response=True),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_trace_exception(gitlab_client_mock, metadata):
    # Set up mock responses
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(
            status_code=200,
            body=[
                {"id": 10, "status": "success"},
                {"id": 11, "status": "failed"},
            ],
        ),
        GitLabHttpResponse(
            status_code=200,
            body=[
                {"id": 101, "name": "job1", "status": "success"},
                {"id": 102, "name": "job2", "status": "failed"},
            ],
        ),
        Exception("Trace error"),
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "merge_request" in response_json
    assert "traces" in response_json
    assert "Error fetching trace: Trace error" in response_json["traces"]

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1", use_http_response=True),
        call(
            path="/api/v4/projects/1/merge_requests/1/pipelines", use_http_response=True
        ),
        call(path="/api/v4/projects/1/pipelines/10/jobs", use_http_response=True),
        call(path="/api/v4/projects/1/jobs/102/trace", parse_json=False),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_network_exception(gitlab_client_mock, metadata):
    # Test that network/client exceptions are properly caught and returned as JSON
    gitlab_client_mock.aget = AsyncMock(
        side_effect=Exception("Network connection error")
    )

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Network connection error" in response_json["error"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/1", use_http_response=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,expected_project_id,expected_pipeline_id",
    [
        (
            "https://gitlab.com/namespace/project/-/pipelines/123",
            "namespace%2Fproject",
            123,
        ),
        (
            "https://gitlab.com/group/subgroup/project/-/pipelines/456",
            "group%2Fsubgroup%2Fproject",
            456,
        ),
        (
            "https://gitlab.com/namespace/project/-/pipelines/789",
            "namespace%2Fproject",
            789,
        ),
    ],
)
async def test_get_pipeline_errors_with_pipeline_url_success(
    url, expected_project_id, expected_pipeline_id, gitlab_client_mock, metadata
):
    jobs_response = [
        {"id": 101, "name": "job1", "status": "success"},
        {"id": 102, "name": "job2", "status": "failed"},
        {"id": 103, "name": "job3", "status": "failed"},
    ]
    trace_response_1 = "Job 102 trace log"
    trace_response_2 = "Job 103 trace log"

    responses = [
        GitLabHttpResponse(status_code=200, body=jobs_response),
        "Job 102 trace log",
        "Job 103 trace log",
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(url=url)
    response_json = json.loads(response)

    expected_traces = (
        "Failed Jobs:\n"
        "Name: job2\nJob ID: 102\nTrace: Job 102 trace log\n"
        "Name: job3\nJob ID: 103\nTrace: Job 103 trace log\n"
    )

    assert response_json == {
        "pipeline_id": expected_pipeline_id,
        "traces": expected_traces,
    }

    assert gitlab_client_mock.aget.call_args_list == [
        call(
            path=f"/api/v4/projects/{expected_project_id}/pipelines/{expected_pipeline_id}/jobs",
            use_http_response=True,
        ),
        call(
            path=f"/api/v4/projects/{expected_project_id}/jobs/102/trace",
            parse_json=False,
        ),
        call(
            path=f"/api/v4/projects/{expected_project_id}/jobs/103/trace",
            parse_json=False,
        ),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_errors_with_pipeline_url_no_failed_jobs(
    gitlab_client_mock, metadata
):
    jobs_response = [
        {"id": 101, "name": "job1", "status": "success"},
        {"id": 102, "name": "job2", "status": "success"},
    ]

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=jobs_response,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert response_json == {
        "pipeline_id": 123,
        "traces": "Failed Jobs:\n",
    }

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs",
        use_http_response=True,
    )


@pytest.mark.asyncio
async def test_get_pipeline_errors_with_pipeline_url_jobs_not_found(
    gitlab_client_mock, metadata
):
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"status": 404, "message": "Pipeline not found"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Failed to fetch jobs for pipeline 123" in response_json["error"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs",
        use_http_response=True,
    )


@pytest.mark.asyncio
async def test_get_pipeline_errors_with_pipeline_url_trace_error(
    gitlab_client_mock, metadata
):
    jobs_response = [
        {"id": 101, "name": "job1", "status": "failed"},
    ]

    responses = [
        GitLabHttpResponse(status_code=200, body=jobs_response),
        Exception("Trace fetch failed"),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    expected_traces = (
        "Failed Jobs:\n"
        "Name: job1\nJob ID: 101\nError fetching trace: Trace fetch failed\n"
    )

    assert response_json == {
        "pipeline_id": 123,
        "traces": expected_traces,
    }

    assert gitlab_client_mock.aget.call_args_list == [
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs",
            use_http_response=True,
        ),
        call(
            path="/api/v4/projects/namespace%2Fproject/jobs/101/trace", parse_json=False
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,error_contains",
    [
        (
            "https://gitlab.com/namespace/project",
            "Failed to parse URL",
        ),
        (
            "https://gitlab.com/namespace/project/-/pipelines/",
            "Failed to parse URL",
        ),
        (
            "https://gitlab.com/namespace/project/-/pipelines/abc",
            "Failed to parse URL",
        ),
        (
            "invalid-url",
            "Failed to parse URL",
        ),
    ],
)
async def test_get_pipeline_errors_with_invalid_pipeline_url(
    url, error_contains, gitlab_client_mock, metadata
):
    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(url=url)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_pipeline_errors_with_pipeline_url_and_conflicting_params(
    gitlab_client_mock, metadata
):
    # Test that when both pipeline URL and merge request params are provided,
    # the pipeline URL takes precedence
    jobs_response = [
        {"id": 101, "name": "job1", "status": "failed"},
    ]
    trace_response = "Job 101 trace log"

    responses = [
        GitLabHttpResponse(status_code=200, body=jobs_response),
        "Job 101 trace log",
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineErrors(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123",
        project_id="different_project",
        merge_request_iid=456,
    )
    response_json = json.loads(response)

    expected_traces = (
        "Failed Jobs:\n" "Name: job1\nJob ID: 101\nTrace: Job 101 trace log\n"
    )

    assert response_json == {
        "pipeline_id": 123,
        "traces": expected_traces,
    }

    # Should use the pipeline URL, not the merge request params
    assert gitlab_client_mock.aget.call_args_list == [
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs",
            use_http_response=True,
        ),
        call(
            path="/api/v4/projects/namespace%2Fproject/jobs/101/trace", parse_json=False
        ),
    ]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetPipelineErrorsInput(
                url="https://gitlab.com/namespace/project/-/pipelines/42"
            ),
            "Get pipeline error logs for https://gitlab.com/namespace/project/-/pipelines/42",
        ),
    ],
)
def test_get_pipeline_errors_format_display_message_with_pipeline_url(
    input_data, expected_message
):
    tool = GetPipelineErrors(description="Get pipeline errors description")
    message = tool.format_display_message(input_data)
    assert message == expected_message
