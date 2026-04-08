"""Project data source tests.

Validates that the agent generates correct GLQL queries for Project type, including query fields, display fields,
sorting, and scope requirements.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_PROJECTS, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_project_query_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use correct query fields for projects: type, group scope."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all projects in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Project",
            'The GLQL query includes group = "gitlab-org" or namespace = "gitlab-org"',
        ]
    )


@pytest.mark.asyncio
async def test_project_boolean_filters(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use boolean filter fields correctly."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me projects with vulnerabilities in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Project",
            'The GLQL query includes group = "gitlab-org" or namespace = "gitlab-org"',
            "The GLQL query filters by hasVulnerabilities = true",
        ]
    )


@pytest.mark.asyncio
async def test_project_archived_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should handle archived project filtering."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me archived projects in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Project",
            "The GLQL query filters for archived projects, using either archivedOnly = true or includeArchived = true",
        ]
    )


@pytest.mark.asyncio
async def test_project_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use appropriate display fields for projects."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me projects in the gitlab-org group with their name, stars, and open issues count",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Project",
            "The GLQL embedded view fields include name, starCount, and openIssuesCount",
        ]
    )


@pytest.mark.asyncio
async def test_project_sorting(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use valid sort fields for projects (fullPath, lastActivity, path)."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me projects in the gitlab-org group sorted by most recently active",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Project",
            "The GLQL embedded view sorts by lastActivity desc",
        ]
    )
