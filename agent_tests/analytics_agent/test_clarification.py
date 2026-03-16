"""Clarification Behavior Tests.

Validates that the agent asks for clarification on ambiguous terms
but proceeds with assumptions on clear questions.
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
async def test_ambiguous_team_triggers_clarification(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Ambiguous 'team' concept should trigger clarification without tool execution."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What is my team working on?",
    )

    result.assert_not_called_tool("run_glql_query")
    await result.assert_llm_validates(
        ["The response asks for clarification about what 'team' means "]
    )


@pytest.mark.asyncio
async def test_ambiguous_bugs_and_quarter_triggers_clarification(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Ambiguous 'bugs' and 'quarter' should trigger clarification without tool execution."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many bugs were created this quarter?",
    )

    result.assert_not_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response asks for clarification about what 'bugs' means (e.g., label ~bug vs issue type)",
            "The response asks about 'quarter' definition (e.g., calendar quarter vs fiscal year)",
        ]
    )


@pytest.mark.asyncio
async def test_ambiguous_velocity_triggers_clarification(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Vague analytical term 'velocity' should trigger clarification without tool execution."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What's our team's velocity?",
    )

    result.assert_not_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response asks for clarification about how to measure 'velocity' and offers specific options",
        ]
    )


@pytest.mark.asyncio
async def test_clear_question_proceeds_without_clarification(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Clear, unambiguous question should proceed without clarification."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS, count=7))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many merge requests with label ~bug were merged in the last 7 days in the gitlab-org group?",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response provides a direct answer without asking clarifying questions about which project/group to use",
            "The response includes the count of 7 merge requests",
        ]
    )
