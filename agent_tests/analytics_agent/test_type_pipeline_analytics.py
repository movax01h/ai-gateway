"""Pipeline analytics data source tests.

Validates that the agent generates correct GLQL queries for Pipeline analytics, including analytics mode, dimensions,
metrics, filters, and result interpretation.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_PIPELINE_ANALYTICS_BY_REF,
    SAMPLE_PIPELINE_ANALYTICS_BY_STATUS,
    SAMPLE_PIPELINE_ANALYTICS_WEEKLY,
    glql_analytics_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_success_rate_by_ref(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query for success rate by ref and surface the data."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_PIPELINE_ANALYTICS_BY_REF),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What's the pipeline success rate by branch in project gitlab-org/gitlab?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query uses ref as a dimension",
            "The GLQL query includes successRate as a metric",
            "The response states or implies that main has a higher success rate than develop",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_failure_rate_by_status(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query grouped by status and surface the data."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_PIPELINE_ANALYTICS_BY_STATUS),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Break down pipelines by status in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query uses status as a dimension",
            "The GLQL query includes totalCount as a metric",
            "The response identifies success as the dominant status",
        ]
    )


@pytest.mark.asyncio
async def test_pipeline_trends_over_time_group_scope(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query at group scope with a time dimension."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_PIPELINE_ANALYTICS_WEEKLY),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me pipeline success-rate trends for the last 30 days "
        "in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes group = "gitlab-org" (not project)',
            "The GLQL query uses finished (or started) as a dimension",
            "The GLQL query filters by finished or started using a relative time expression",
            "The GLQL query includes successRate as a metric",
            "The response references the weekly trend across the three data points",
        ]
    )
