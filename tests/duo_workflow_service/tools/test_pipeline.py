import json
from unittest.mock import AsyncMock, Mock, call

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.pipeline import (
    GetDownstreamPipelines,
    GetPipelineFailingJobs,
    GetPipelineFailingJobsInput,
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
async def test_get_pipeline_failing_jobs(gitlab_client_mock, metadata):
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
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    # Validate the output
    expected_response = {
        "merge_request": {"id": 1, "title": "Merge Request 1"},
        "failed_jobs": (
            "Failed Jobs:\n<jobs>\n"
            "  <job>\n"
            "    <job_name>job2</job_name>\n"
            "    <job_id>102</job_id>\n"
            "  </job>\n"
            "</jobs>\n"
        ),
    }

    assert json.loads(response) == expected_response

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
        call(
            path="/api/v4/projects/1/pipelines/10/jobs?per_page=100&page=1",
        ),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_merge_request_not_found(
    gitlab_client_mock, metadata
):
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"status": 404},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Merge request with iid 1 not found" in response_json["error"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/1"
    )


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_pipelines_not_found(
    gitlab_client_mock, metadata
):
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(status_code=200, body=[]),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})
    response_json = json.loads(response)

    assert "error" in response_json
    assert "No pipelines found for merge request iid 1" in response_json["error"]

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
async def test_get_pipeline_failing_jobs_with_url_success(
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

    responses = [
        GitLabHttpResponse(status_code=200, body=merge_request_response),
        GitLabHttpResponse(status_code=200, body=pipelines_response),
        GitLabHttpResponse(status_code=200, body=jobs_response),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(
        url=url, project_id=project_id, merge_request_iid=merge_request_iid
    )

    expected_response = json.dumps(
        {
            "merge_request": {"id": 1, "title": "Merge Request 1"},
            "failed_jobs": (
                "Failed Jobs:\n<jobs>\n"
                "  <job>\n"
                "    <job_name>job2</job_name>\n"
                "    <job_id>102</job_id>\n"
                "  </job>\n"
                "</jobs>\n"
            ),
        }
    )
    assert response == expected_response


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
async def test_get_pipeline_failing_jobs_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetPipelineFailingJobs(metadata=metadata)

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
            GetPipelineFailingJobsInput(project_id=123, merge_request_iid=456),
            "Get pipeline failing jobs for merge request !456 in project 123",
        ),
        (
            GetPipelineFailingJobsInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Get pipeline failing jobs for https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_get_pipeline_failing_jobs_format_display_message(input_data, expected_message):
    tool = GetPipelineFailingJobs(description="Get pipeline errors description")
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
    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(
        project_id=project_id, merge_request_iid=merge_request_iid
    )
    response_json = json.loads(response)

    assert "error" in response_json
    for error in expected_errors:
        assert error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_jobs_error(gitlab_client_mock, metadata):
    responses = [
        GitLabHttpResponse(status_code=200, body={"id": 1, "title": "Merge Request 1"}),
        GitLabHttpResponse(status_code=200, body=[{"id": 10, "status": "success"}]),
        GitLabHttpResponse(
            status_code=404, body={"status": 404, "message": "Jobs not found"}
        ),
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    assert "Jobs not found" in str(exc_info.value)

    assert gitlab_client_mock.aget.call_args_list == [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
        call(path="/api/v4/projects/1/pipelines/10/jobs?per_page=100&page=1"),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_network_exception(
    gitlab_client_mock, metadata
):
    # Test that network/client exceptions are properly caught and returned as JSON
    gitlab_client_mock.aget = AsyncMock(
        side_effect=Exception("Network connection error")
    )

    tool = GetPipelineFailingJobs(metadata=metadata)

    with pytest.raises(Exception) as exc_info:
        await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    assert "Network connection error" in str(exc_info.value)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/1"
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
async def test_get_pipeline_failing_jobs_with_pipeline_url_success(
    url, expected_project_id, expected_pipeline_id, gitlab_client_mock, metadata
):
    jobs_response = [
        {"id": 101, "name": "job1", "status": "success"},
        {"id": 102, "name": "job2", "status": "failed"},
        {"id": 103, "name": "job3", "status": "failed"},
    ]
    responses = [
        GitLabHttpResponse(status_code=200, body=jobs_response),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(url=url)
    response_json = json.loads(response)

    expected_traces = (
        "Failed Jobs:\n<jobs>\n"
        "  <job>\n"
        "    <job_name>job2</job_name>\n"
        "    <job_id>102</job_id>\n"
        "  </job>\n"
        "  <job>\n"
        "    <job_name>job3</job_name>\n"
        "    <job_id>103</job_id>\n"
        "  </job>\n"
        "</jobs>\n"
    )

    assert response_json == {
        "pipeline_id": expected_pipeline_id,
        "failed_jobs": expected_traces,
    }

    assert gitlab_client_mock.aget.call_args_list == [
        call(
            path=f"/api/v4/projects/{expected_project_id}/pipelines/{expected_pipeline_id}/jobs?per_page=100&page=1",
        ),
    ]


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_with_pipeline_url_no_failed_jobs(
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

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert response_json == {"error": "No Failing Jobs Found."}

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs?per_page=100&page=1",
    )


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_with_pipeline_url_jobs_not_found(
    gitlab_client_mock, metadata
):
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"status": 404, "message": "Pipeline not found"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetPipelineFailingJobs(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(url="https://gitlab.com/namespace/project/-/pipelines/123")

    assert "Pipeline not found" in str(exc_info.value)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs?per_page=100&page=1",
    )


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
async def test_get_pipeline_failing_jobs_with_invalid_pipeline_url(
    url, error_contains, gitlab_client_mock, metadata
):
    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(url=url)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_with_pipeline_url_and_conflicting_params(
    gitlab_client_mock, metadata
):
    # Test that when both pipeline URL and merge request params are provided,
    # the pipeline URL takes precedence
    jobs_response = [
        {"id": 101, "name": "job1", "status": "failed"},
    ]

    responses = [
        GitLabHttpResponse(status_code=200, body=jobs_response),
    ]
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123",
        project_id="different_project",
        merge_request_iid=456,
    )
    response_json = json.loads(response)

    expected_traces = (
        "Failed Jobs:\n<jobs>\n"
        "  <job>\n"
        "    <job_name>job1</job_name>\n"
        "    <job_id>101</job_id>\n"
        "  </job>\n"
        "</jobs>\n"
    )

    assert response_json == {
        "pipeline_id": 123,
        "failed_jobs": expected_traces,
    }

    # Should use the pipeline URL, not the merge request params
    assert gitlab_client_mock.aget.call_args_list == [
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/123/jobs?per_page=100&page=1",
        ),
    ]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetPipelineFailingJobsInput(
                url="https://gitlab.com/namespace/project/-/pipelines/42"
            ),
            "Get pipeline failing jobs for https://gitlab.com/namespace/project/-/pipelines/42",
        ),
    ],
)
def test_get_pipeline_failing_jobs_format_display_message_with_pipeline_url(
    input_data, expected_message
):
    tool = GetPipelineFailingJobs(description="Get pipeline errors description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_jobs_pagination(gitlab_client_mock, metadata):
    """Test that pagination works correctly when fetching jobs across multiple pages."""
    # Mock responses for merge request and pipelines
    merge_request_response = GitLabHttpResponse(
        status_code=200, body={"id": 1, "title": "Merge Request 1"}
    )
    pipelines_response = GitLabHttpResponse(
        status_code=200, body=[{"id": 10, "status": "success"}]
    )

    # Mock first page of jobs with X-Next-Page header indicating there's a second page
    jobs_page_1_response = GitLabHttpResponse(
        status_code=200,
        body=[
            {"id": 101, "name": "job1", "status": "success"},
            {"id": 102, "name": "job2", "status": "failed"},
        ],
        headers={"X-Next-Page": "2"},
    )

    # Mock second page of jobs with no X-Next-Page header (last page)
    jobs_page_2_response = GitLabHttpResponse(
        status_code=200,
        body=[
            {"id": 103, "name": "job3", "status": "failed"},
            {"id": 104, "name": "job4", "status": "success"},
        ],
        headers={},
    )

    responses = [
        merge_request_response,
        pipelines_response,
        jobs_page_1_response,
        jobs_page_2_response,
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool.arun({"project_id": "1", "merge_request_iid": "1"})

    expected_response = {
        "merge_request": {"id": 1, "title": "Merge Request 1"},
        "failed_jobs": (
            "Failed Jobs:\n<jobs>\n"
            "  <job>\n"
            "    <job_name>job2</job_name>\n"
            "    <job_id>102</job_id>\n"
            "  </job>\n"
            "  <job>\n"
            "    <job_name>job3</job_name>\n"
            "    <job_id>103</job_id>\n"
            "  </job>\n"
            "</jobs>\n"
        ),
    }

    assert json.loads(response) == expected_response

    expected_calls = [
        call(path="/api/v4/projects/1/merge_requests/1"),
        call(path="/api/v4/projects/1/merge_requests/1/pipelines"),
        call(
            path="/api/v4/projects/1/pipelines/10/jobs?per_page=100&page=1",
        ),
        call(
            path="/api/v4/projects/1/pipelines/10/jobs?per_page=100&page=2",
        ),
    ]

    assert gitlab_client_mock.aget.call_args_list == expected_calls


@pytest.mark.asyncio
async def test_get_pipeline_failing_jobs_jobs_pagination_with_pipeline_url(
    gitlab_client_mock, metadata
):
    """Test pagination works correctly when using a pipeline URL directly."""
    # Mock first page of jobs with X-Next-Page header
    jobs_page_1_response = GitLabHttpResponse(
        status_code=200,
        body=[
            {"id": 201, "name": "build", "status": "failed"},
            {"id": 202, "name": "test", "status": "success"},
        ],
        headers={"X-Next-Page": "2"},
    )

    # Mock second page of jobs with X-Next-Page header for third page
    jobs_page_2_response = GitLabHttpResponse(
        status_code=200,
        body=[
            {"id": 203, "name": "deploy", "status": "failed"},
        ],
        headers={"X-Next-Page": "3"},
    )

    # Mock third page of jobs with no X-Next-Page header (last page)
    jobs_page_3_response = GitLabHttpResponse(
        status_code=200,
        body=[
            {"id": 204, "name": "cleanup", "status": "success"},
        ],
        headers={},
    )

    responses = [
        jobs_page_1_response,
        jobs_page_2_response,
        jobs_page_3_response,
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = GetPipelineFailingJobs(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/456"
    )
    response_json = json.loads(response)

    # Validate the output includes traces from failed jobs across all pages
    expected_traces = (
        "Failed Jobs:\n<jobs>\n"
        "  <job>\n"
        "    <job_name>build</job_name>\n"
        "    <job_id>201</job_id>\n"
        "  </job>\n"
        "  <job>\n"
        "    <job_name>deploy</job_name>\n"
        "    <job_id>203</job_id>\n"
        "  </job>\n"
        "</jobs>\n"
    )

    assert response_json == {
        "pipeline_id": 456,
        "failed_jobs": expected_traces,
    }

    # Verify pagination calls were made correctly
    expected_calls = [
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/456/jobs?per_page=100&page=1",
        ),
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/456/jobs?per_page=100&page=2",
        ),
        call(
            path="/api/v4/projects/namespace%2Fproject/pipelines/456/jobs?per_page=100&page=3",
        ),
    ]

    assert gitlab_client_mock.aget.call_args_list == expected_calls


@pytest.mark.asyncio
async def test_get_downstream_pipelines_success(gitlab_client_mock, metadata):
    """Test successful retrieval of downstream pipelines."""
    bridges_response = [
        {
            "id": 1001,
            "name": "downstream_job_1",
            "status": "success",
            "downstream_pipeline": {
                "id": 2001,
                "status": "failed",
                "web_url": "https://gitlab.com/namespace/project/-/pipelines/1233",
            },
        },
        {
            "id": 1002,
            "name": "downstream_job_2",
            "status": "success",
            "downstream_pipeline": {
                "id": 2002,
                "status": "failed",
                "web_url": "https://gitlab.com/namespace/project/-/pipelines/1232",
            },
        },
    ]

    mock_response = GitLabHttpResponse(status_code=200, body=bridges_response)
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert isinstance(response_json, list)
    assert response_json == [
        {"url": "https://gitlab.com/namespace/project/-/pipelines/1233"},
        {"url": "https://gitlab.com/namespace/project/-/pipelines/1232"},
    ]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_invalid_downstream_url(
    gitlab_client_mock, metadata
):
    """Test unsuccessful retrieval of invalid downstream pipeline url."""
    bridges_response = [
        {
            "id": 1001,
            "name": "downstream_job_1",
            "status": "success",
            "downstream_pipeline": {
                "id": 2001,
                "status": "failed",
                "web_url": "https://gitlab.com/",
            },
        }
    ]

    mock_response = GitLabHttpResponse(status_code=200, body=bridges_response)
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert response_json == {
        "error": "Failed to parse URL: Could not parse pipeline URL: https://gitlab.com/"
    }

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_no_multi_project(gitlab_client_mock, metadata):
    """Test does not return multi-project downstream pipelines."""
    bridges_response = [
        {
            "id": 1001,
            "name": "downstream_job_1",
            "status": "success",
            "downstream_pipeline": {
                "id": 2001,
                "status": "failed",
                "web_url": "https://gitlab.com/namespace/different_project/-/pipelines/1233",
            },
        },
        {
            "id": 1002,
            "name": "downstream_job_2",
            "status": "success",
            "downstream_pipeline": {
                "id": 2002,
                "status": "failed",
                "web_url": "https://gitlab.com/different_namespace/project/-/pipelines/1232",
            },
        },
    ]

    mock_response = GitLabHttpResponse(status_code=200, body=bridges_response)
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert isinstance(response_json, list)
    assert response_json == []

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_no_downstream_pipelines(
    gitlab_client_mock, metadata
):
    """Test when there are no downstream pipelines."""
    bridges_response = []

    mock_response = GitLabHttpResponse(status_code=200, body=bridges_response)
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/pipelines/123"
    )
    response_json = json.loads(response)

    assert isinstance(response_json, list)
    assert len(response_json) == 0

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_invalid_url(gitlab_client_mock, metadata):
    """Test with invalid pipeline URL."""
    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(url="https://gitlab.com/namespace/project")
    response_json = json.loads(response)

    assert "error" in response_json
    assert "Failed to parse URL" in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_downstream_pipelines_missing_url(gitlab_client_mock, metadata):
    """Test when URL is not provided."""
    tool = GetDownstreamPipelines(metadata=metadata)

    with pytest.raises(TypeError) as exc_info:
        await tool._arun()

    assert "missing 1 required positional argument: 'url'" in str(exc_info.value)
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_downstream_pipelines_api_error(gitlab_client_mock, metadata):
    """Test when GitLab API returns an error."""
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"status": 404, "message": "Pipeline not found"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(url="https://gitlab.com/namespace/project/-/pipelines/999")

    assert "Pipeline not found" in str(exc_info.value)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/999/bridges"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,expected_project_id,expected_pipeline_id,downstream_url",
    [
        (
            "https://gitlab.com/namespace/project/-/pipelines/123",
            "namespace%2Fproject",
            123,
            "https://gitlab.com/namespace/project/-/pipelines/1234",
        ),
        (
            "https://gitlab.com/group/subgroup/project/-/pipelines/456",
            "group%2Fsubgroup%2Fproject",
            456,
            "https://gitlab.com/group/subgroup/project/-/pipelines/4567",
        ),
    ],
)
async def test_get_downstream_pipelines_url_parsing(
    url,
    expected_project_id,
    expected_pipeline_id,
    downstream_url,
    gitlab_client_mock,
    metadata,
):
    """Test correct parsing of various pipeline URL formats."""
    bridges_response = [
        {
            "id": 1001,
            "name": "downstream_job",
            "status": "success",
            "downstream_pipeline": {
                "id": 2002,
                "status": "failed",
                "web_url": downstream_url,
            },
        },
    ]

    mock_response = GitLabHttpResponse(status_code=200, body=bridges_response)
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    response = await tool._arun(url=url)
    response_json = json.loads(response)

    assert isinstance(response_json, list)
    assert response_json == [{"url": downstream_url}]

    gitlab_client_mock.aget.assert_called_once_with(
        path=f"/api/v4/projects/{expected_project_id}/pipelines/{expected_pipeline_id}/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_format_display_message():
    """Test the format_display_message method."""
    from duo_workflow_service.tools.gitlab_resource_input import GitLabResourceInput

    tool = GetDownstreamPipelines(description="Get downstream pipelines description")
    input_data = GitLabResourceInput(
        url="https://gitlab.com/namespace/project/-/pipelines/42"
    )
    message = tool.format_display_message(input_data)

    assert (
        message
        == "Get downstream pipelines for https://gitlab.com/namespace/project/-/pipelines/42"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_network_error(gitlab_client_mock, metadata):
    """Test handling of network errors."""
    gitlab_client_mock.aget = AsyncMock(
        side_effect=Exception("Network connection error")
    )

    tool = GetDownstreamPipelines(metadata=metadata)

    with pytest.raises(Exception) as exc_info:
        await tool._arun(url="https://gitlab.com/namespace/project/-/pipelines/123")

    assert "Network connection error" in str(exc_info.value)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_invalid_response_format(
    gitlab_client_mock, metadata
):
    """Test handling of invalid response format from API."""
    # API returns a dict instead of a list
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"error": "Invalid response"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetDownstreamPipelines(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(url="https://gitlab.com/namespace/project/-/pipelines/123")

    assert "Failed to fetch downstream pipelines for url" in str(exc_info.value)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/pipelines/123/bridges"
    )


@pytest.mark.asyncio
async def test_get_downstream_pipelines_with_malformed_url(
    gitlab_client_mock, metadata
):
    """Test with various malformed URLs."""
    tool = GetDownstreamPipelines(metadata=metadata)

    malformed_urls = [
        "https://gitlab.com/namespace/project/-/pipelines/",
        "https://gitlab.com/namespace/project/-/pipelines/abc",
        "invalid-url",
        "https://gitlab.com/namespace/project/-/merge_requests/123",
    ]

    for url in malformed_urls:
        response = await tool._arun(url=url)
        response_json = json.loads(response)
        assert "error" in response_json
        assert "Failed to parse URL" in response_json["error"]

    gitlab_client_mock.aget.assert_not_called()
