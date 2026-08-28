"""Scope Handling Tests.

Validates scope assumptions and scope notes in responses.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_JOBS,
    SAMPLE_MRS,
    glql_response,
    is_project_scoped,
    mock_glql_response,
)


@pytest.mark.flow_versions("1.0.0", "2.0.0")
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
            "The response indicates the group or project being used, "
            "either in the response or in the underlying query, "
            "OR ask for clarification on which project/group to use",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_explicit_group_level_request(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should handle explicit group-level requests with group filter."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open MRs across the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'group = gitlab-org' to specify group-level scope",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_explicit_project_level_request(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should handle explicit project-level requests with project filter."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "List issues in the gitlab-org/gitlab-test project",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'project = gitlab-org/gitlab-test' to specify the project scope",
        ]
    )


@pytest.mark.flow_versions("2.0.0")
@pytest.mark.asyncio
async def test_scope_the_source_does_not_allow_is_not_used(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """A source's `allowed_scopes` beats the scope the user asked for.

    Jobs allow `project` only, so a group-wide request cannot be honoured as
    asked. Only 2.0.0: the bundled schema does not carry scopes, so 1.0.0
    relies on `test_type_jobs`.
    """
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_JOBS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me failed jobs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    queries = result.get_tool_call_args("run_glql_query", "glql_yaml")
    if not is_project_scoped(queries):
        await result.assert_llm_validates(
            ["The response explains that jobs can only be queried per project"]
        )
