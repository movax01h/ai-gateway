"""Pipeline data source tests.

Validates that the agent generates correct GLQL queries for Pipeline type, including query fields, display fields,
sorting constraints, and scope requirements.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_PIPELINES, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_pipeline_query_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use correct query fields for pipelines: type, project, status, ref."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        'Show me failed pipelines on the "main" branch in project gitlab-org/gitlab',
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by status = failed",
            'The GLQL query filters by ref = "main"',
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_status_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use status enum values correctly."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all running pipelines in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query includes status = running",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_date_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use date comparison operators for updated field."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me pipelines created in the last week in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by updated using a relative time expression like -1w",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use appropriate display fields for pipelines."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me recent pipelines in project gitlab-org/gitlab with their status, duration and ref",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            "The GLQL embedded view fields include status, duration, and ref",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_no_sorting(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should not include sort parameter for pipelines since sorting is not supported."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me pipelines in project gitlab-org/gitlab sorted by most recent",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view does NOT include a sort parameter "
            "and/or the response explains that sorting is not supported",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_requires_project_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use project filter for pipelines, not group scope."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me failed pipelines in the gitlab-org group",
    )

    await result.assert_llm_validates(
        [
            "The GLQL query uses project filter, not group filter, "
            "and/or the response explains that pipelines require a project scope",
        ]
    )
