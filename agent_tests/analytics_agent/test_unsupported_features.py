"""Unsupported Features Tests.

Validates graceful handling of features GLQL doesn't support.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_ISSUES, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_chart_limitation_explained(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should explain chart limitation and offer alternative."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "show all issues created in gitlab-org group as a chart",
    )

    await result.assert_llm_validates(
        [
            "The response explains that GLQL does not currently support charts or graphs",
            "The response offers supported options as an alternative",
        ]
    )


@pytest.mark.asyncio
async def test_text_search_limitation_explained(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should handle text search limitation gracefully."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Find issues where the title contains 'authentication'",
    )

    await result.assert_llm_validates(
        [
            "The response explains that GLQL does not support text search, contains, or like operators."
        ]
    )
