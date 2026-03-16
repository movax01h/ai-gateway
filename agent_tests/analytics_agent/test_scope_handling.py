"""Scope Handling Tests.

Validates scope assumptions and scope notes in responses.
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
async def test_scope_note_when_assuming_project(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should include scope note when assuming project-level."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are there?",
    )

    await result.assert_llm_validates(
        [
            "The response indicates the group or project being used, either in the response or in the underlying query, OR ask for clarification on which project/group to use",
        ]
    )


@pytest.mark.asyncio
async def test_explicit_group_level_request(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should handle explicit group-level requests with group filter."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open MRs across the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'group = gitlab-org' to specify group-level scope",
        ]
    )


@pytest.mark.asyncio
async def test_explicit_project_level_request(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should handle explicit project-level requests with project filter."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "List issues in the gitlab-org/gitlab-test project",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'project = gitlab-org/gitlab-test' to specify the project scope",
        ]
    )
