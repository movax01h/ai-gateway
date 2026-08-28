"""Pipeline analytics data source tests.

Validates that the agent generates correct GLQL queries for Pipeline analytics, including analytics mode, dimensions,
metrics, filters, and result interpretation.
"""

import re

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_PIPELINE_ANALYTICS_BY_REF,
    SAMPLE_PIPELINE_ANALYTICS_BY_STATUS,
    SAMPLE_PIPELINE_ANALYTICS_WEEKLY,
    glql_analytics_response,
    has_group_filter,
    mock_glql_response,
)


def _is_group_scoped_trend_query(glql_yaml: str) -> bool:
    """Whether a single query is an analytics-mode Pipeline trend query at group scope.

    Checks the facts the test proves without pinning the time granularity: analytics mode, Pipeline
    type, group filter, a finished/started time dimension, a relative time filter, and the
    successRate metric.
    """
    return bool(
        re.search(r"^\s*mode:\s*analytics\b", glql_yaml, re.MULTILINE)
        and re.search(r"\btype\s*=\s*Pipeline\b", glql_yaml)
        and has_group_filter(glql_yaml)
        and re.search(
            r"^\s*dimensions:.*\b(finished|started)\b", glql_yaml, re.MULTILINE
        )
        and re.search(r"\b(finished|started)\s*(>=?|<=?)\s*-\d+[dwmy]\b", glql_yaml)
        and re.search(r"^\s*metrics:.*\bsuccessRate\b", glql_yaml, re.MULTILINE)
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_success_rate_by_ref(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
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

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
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


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_failure_rate_by_status(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
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

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query uses status as a dimension",
            "The GLQL query includes totalCount as a metric",
            "The response identifies success as the dominant status",
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_trends_over_time_group_scope(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
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

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")

    # The query facts are checked deterministically; granularity (daily vs
    # weekly bucketing) is the agent's choice and is deliberately not pinned.
    queries = result.get_tool_call_args("run_glql_query", "glql_yaml")
    assert any(_is_group_scoped_trend_query(q) for q in queries), (
        "Expected an analytics-mode Pipeline query at group scope with a "
        "finished/started time dimension, a relative time filter, and the "
        f"successRate metric. Queries: {queries}"
    )
    await result.assert_llm_validates(
        [
            "The response describes the pipeline success rate trend over time",
        ]
    )
