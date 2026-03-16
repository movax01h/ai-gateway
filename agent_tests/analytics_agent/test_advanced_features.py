"""GLQL Advanced Features Tests.

Validates advanced GLQL syntax, operators, and currentUser() function.
"""

import pytest

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    glql_response,
    mock_glql_response,
)
from agent_tests.helpers import ask_agent


@pytest.mark.asyncio
async def test_labels_in_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should include labels in fields when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues with their labels displayed in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'labels' in the fields parameter",
        ]
    )


@pytest.mark.asyncio
async def test_negation_operator(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use != operator correctly for negation."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues not assigned to me in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses 'assignee != currentUser()'",
        ]
    )


@pytest.mark.asyncio
async def test_current_user_for_my_queries(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use currentUser() for 'my' queries."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me my open merge requests in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response indicates filtering by the current user, evidenced by: currentUser() in the GLQL query",
        ]
    )


@pytest.mark.asyncio
async def test_current_user_for_assigned_items(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use currentUser() for assigned items."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What issues from the gitlab-org group are assigned to me?",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses 'assignee = currentUser()' to filter by the current user",
        ]
    )
