"""Edge Cases Tests.

Validates graceful handling of edge cases like empty results.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import EMPTY_RESPONSE, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_empty_results_handled_gracefully(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should handle empty results gracefully."""
    mock_glql_response(mock_gitlab_client, glql_response(EMPTY_RESPONSE, count=0))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues with label ~nonexistent-label-xyz123 in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response acknowledges that no results were found",
            "The response still includes the GLQL query for reference",
        ]
    )
