"""Smoke tests for the Orbit integration."""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    glql_error_response,
    glql_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_orbit_focused_question_routes_to_orbit(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Orbit-focused question must route to Orbit, not GLQL."""
    del mock_gitlab_client

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "In this project, which agent definitions call `run_glql_query`?",
    )

    result.assert_called_tool("orbit_list_commands")
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="get_graph_schema"
    )
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="query_graph"
    )
    result.assert_not_called_tool("run_glql_query")


@pytest.mark.asyncio
async def test_cross_entities_question_routes_to_orbit(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Cross-entities question must route to Orbit, not GLQL."""
    del mock_gitlab_client

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Which MRs were merged last week that touched the GLQL frontend in the gitlab project?",
    )

    result.assert_called_tool("orbit_list_commands")
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="get_graph_schema"
    )
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="query_graph"
    )


@pytest.mark.asyncio
async def test_data_not_in_glql_routes_to_orbit(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Data not covered by GLQL should route to Orbit."""
    mock_glql_response(
        mock_gitlab_client,
        glql_error_response(
            "The filter 'deployed' is not supported in this GitLab instance."
        ),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Get all deployments of the last week in the gitlab-org/gitlab project",
    )

    result.assert_called_tool("orbit_list_commands")
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="get_graph_schema"
    )
    result.assert_called_tool_with_args(
        "orbit_invoke_command", command_name="query_graph"
    )


@pytest.mark.asyncio
async def test_explicit_glql_request_routes_to_glql(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Explicit GLQL request must skip Orbit deep discovery entirely."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        (
            "Give me a GLQL query I can paste into a GitLab issue to see "
            "all merge requests merged this week in gitlab-org/gitlab."
        ),
    )

    result.assert_called_tool("get_glql_schema")
    result.assert_called_tool("run_glql_query")
    result.assert_not_called_tool("orbit_invoke_command")


@pytest.mark.asyncio
async def test_orbit_unavailable_orbit_shaped_question_surfaces_limitation(
    analytics_agent_without_orbit,
    initial_state,
    mock_gitlab_client,
):
    """No Orbit + Orbit-shape question → surface limitation, don't fabricate via GLQL."""
    del mock_gitlab_client

    result = await ask_agent(
        analytics_agent_without_orbit,
        initial_state,
        "Which code definitions call `run_glql_query`?",
    )

    result.assert_not_called_tool("orbit_list_commands")
    result.assert_not_called_tool("orbit_invoke_command")
    await result.assert_llm_validates(
        [
            "Response acknowledges that the question cannot be answered "
            "with the available tools, or surfaces a limitation",
        ]
    )


@pytest.mark.asyncio
async def test_orbit_unavailable_glql_shaped_question_still_works(
    analytics_agent_without_orbit,
    initial_state,
    mock_gitlab_client,
):
    """No Orbit + GLQL-shape question → GLQL answers normally, no Orbit attempts."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent_without_orbit,
        initial_state,
        "How many open issues are there in gitlab-org/gitlab?",
    )

    result.assert_called_tool("run_glql_query")
    result.assert_not_called_tool("orbit_list_commands")
    result.assert_not_called_tool("orbit_invoke_command")
