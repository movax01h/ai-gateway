"""Time Expression Tests.

Validates preference for relative time expressions over absolute dates.
"""

import pytest

from .helpers import SAMPLE_ISSUES, glql_response, mock_glql_response
from agent_tests.helpers import ask_agent


@pytest.mark.asyncio
async def test_relative_time_for_last_month(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use relative time expression for 'last month'."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues created last month in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses relative time expression like -1m or -30d",
        ]
    )


@pytest.mark.asyncio
async def test_absolute_dates_when_explicitly_requested(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use absolute dates when explicitly requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues created between January 1, 2024 and January 31, 2024 in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses absolute date format like 2024-01-01 since specific dates were requested",
        ]
    )
