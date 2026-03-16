"""GLQL Syntax Constraints Tests.

Validates adherence to GLQL syntax rules and restrictions.
"""

import pytest

from .helpers import SAMPLE_ISSUES, glql_response, mock_glql_response
from agent_tests.helpers import ask_agent


@pytest.mark.asyncio
async def test_invalid_sort_field_uses_valid_alternative(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use valid sort field instead of assignee."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues sorted by assignee in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses a valid sort field (like created, updated, title) instead of assignee and explains that sorting by assignee is not supported"
        ]
    )


@pytest.mark.asyncio
async def test_label_syntax_with_tilde_prefix(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use proper label syntax with ~ prefix."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Find all issues with the priority-high label in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            'The GLQL query uses the ~ prefix for labels (e.g., ~priority-high or ~"priority-high")',
        ]
    )


@pytest.mark.asyncio
async def test_milestone_syntax_with_percent_prefix(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use proper milestone syntax with % prefix."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues in milestone v1.0 in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            'The GLQL query uses the % prefix for milestones (e.g., %v1.0 or %"v1.0")',
        ]
    )


@pytest.mark.asyncio
async def test_limit_maximum_of_100(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should respect limit maximum of 100."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the last 101 issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query has a limit of 100 or less."
            "The response mentions that 100 is the maximum allowed limit",
        ]
    )


@pytest.mark.asyncio
async def test_and_logic_for_multiple_labels(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use AND logic correctly for multiple labels."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Find issues with both ~bug and ~security labels in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses multiple 'label = ~x and label = ~y' conditions for AND logic."
            "It does not use 'label in (~bug, ~security)' which would be OR logic",
        ]
    )
