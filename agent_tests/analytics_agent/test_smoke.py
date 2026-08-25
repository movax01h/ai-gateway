"""Smoke tests for analytics agent."""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_JOBS,
    SAMPLE_MRS,
    SAMPLE_PROJECTS,
    glql_response,
    mock_glql_response,
)


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_how_many_open_issues(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Agent must call run_glql_query and report the count."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are there in the gitlab-org group?",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    await result.assert_llm_validates(
        [
            "Response says 42 open issues",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_show_open_mrs(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Agent must call run_glql_query and present open MRs."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open MRs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "Response presents a GLQL query code block",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_multi_source_schema_single_call(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Asking about projects and jobs should fetch both schemas in one call."""
    mock_glql_response(
        mock_gitlab_client,
        [
            glql_response(SAMPLE_PROJECTS),
            glql_response(SAMPLE_JOBS),
        ],
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all projects in the gitlab-org group "
        "and also the failed jobs in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_tool_call_count(schema_tool_name, 1)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
