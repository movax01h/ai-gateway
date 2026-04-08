"""Job data source tests.

Validates that the agent generates correct GLQL queries for Job type, including query fields, display fields, sorting
constraints, and filter requirements.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_JOBS, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_job_query_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use correct query fields for jobs: type, project, status."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me failed jobs in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Job",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by status = failed",
        ]
    )


@pytest.mark.asyncio
async def test_job_kind_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use kind enum values (bridge/build) correctly."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all bridge jobs (trigger jobs) in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Job",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by kind = bridge",
        ]
    )


@pytest.mark.asyncio
async def test_job_pipeline_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter jobs by pipeline IID."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all jobs in pipeline 12345 in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Job",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by pipeline = 12345",
        ]
    )


@pytest.mark.asyncio
async def test_job_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use appropriate display fields for jobs."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me jobs in project gitlab-org/gitlab with their name, stage, status and duration",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Job",
            "The GLQL embedded view fields include name, stage, status, and duration",
        ]
    )


@pytest.mark.asyncio
async def test_job_no_sorting(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should not include sort parameter for jobs since sorting is not supported."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me jobs in project gitlab-org/gitlab sorted by duration",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view does NOT include a sort parameter "
            "and/or the response explains that jobs do not support sorting",
        ]
    )


@pytest.mark.asyncio
async def test_job_requires_project_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use project filter for jobs, not group filter."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me failed jobs in the gitlab-org group",
    )

    await result.assert_llm_validates(
        [
            "The GLQL query uses project filter, not group filter, "
            "and/or the response explains that jobs require a project filter",
        ]
    )
