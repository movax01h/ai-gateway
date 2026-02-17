"""Smoke tests for analytics agent.

Requires ANTHROPIC_API_KEY environment variable.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    glql_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_how_many_open_issues(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Agent must call run_glql_query and report the count."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are there in the gitlab-org group?",
    )

    (result.assert_has_tool_calls().assert_called_tool("run_glql_query"))
    await result.assert_llm_validates(
        [
            "Response says 42 open issues",
        ]
    )


@pytest.mark.asyncio
async def test_show_open_mrs(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Agent must call run_glql_query and present open MRs."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open MRs in the gitlab-org group",
    )

    (result.assert_has_tool_calls().assert_called_tool("run_glql_query"))
    await result.assert_llm_validates(
        [
            "Response presents a GLQL query code block",
        ]
    )
