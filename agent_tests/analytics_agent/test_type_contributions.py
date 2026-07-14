"""Contributions data source tests.

Validates that the agent generates correct GLQL queries for Contributions analytics, including analytics mode,
dimensions, metrics, filters, and result interpretation.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_CONTRIBUTIONS_MONTHLY,
    SAMPLE_CONTRIBUTIONS_OVERALL,
    glql_analytics_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_contribution_trend_over_time(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should generate analytics-mode query for a contribution trend by month."""
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CONTRIBUTIONS_MONTHLY)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What's the contribution trend over the last month?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = Contribution",
            "The GLQL query uses created as a dimension",
            "The GLQL query includes totalCount as a metric",
        ]
    )


@pytest.mark.asyncio
async def test_unique_contributor_count(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
) -> None:
    """Agent should use usersCount to answer questions about unique contributors."""
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CONTRIBUTIONS_OVERALL)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many unique contributors did this project have in the last 30 days?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics",
            "The GLQL query includes type = Contribution",
            "The GLQL query includes usersCount as a metric",
            "The response reports a number of unique contributors",
        ]
    )
