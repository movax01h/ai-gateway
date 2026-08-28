"""Pipeline data source tests.

Validates that the agent generates correct GLQL queries for Pipeline type, including query fields, display fields,
sorting constraints, and scope requirements.
"""

import pytest

from agent_tests.helpers import ask_agent
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse

from .helpers import (
    PIPELINES_GROUP_SCOPE_ERROR,
    SAMPLE_PIPELINES,
    glql_http_error,
    glql_response,
    has_group_filter,
    has_project_filter,
    is_project_scoped,
    mock_glql_responder,
    mock_glql_response,
)


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_query_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use correct query fields for pipelines: type, project, status, ref."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        'Show me failed pipelines on the "main" branch in project gitlab-org/gitlab',
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by status = failed",
            'The GLQL query filters by ref = "main"',
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_status_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use status enum values correctly."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all running pipelines in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query includes status = running",
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_date_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use date comparison operators for updated field."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me pipelines created in the last week in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            'The GLQL query includes project = "gitlab-org/gitlab"',
            "The GLQL query filters by updated using a relative time expression like -1w",
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use appropriate display fields for pipelines."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me recent pipelines in project gitlab-org/gitlab with their status, duration and ref",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Pipeline",
            "The GLQL embedded view fields include status, duration, and ref",
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_no_sorting(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should not include sort parameter for pipelines since sorting is not supported."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PIPELINES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me pipelines in project gitlab-org/gitlab sorted by most recent",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates_any(
        [
            "The GLQL embedded view does NOT include a sort parameter",
            "The response explains that sorting is not supported",
        ]
    )


@pytest.mark.flow_versions("1.0.0")
@pytest.mark.asyncio
async def test_pipeline_requires_project_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Listing individual pipelines (standard mode) must use project, not group, scope."""

    def respond(glql_yaml: str) -> GitLabHttpResponse:
        # Group-scoped queries fail exactly as production does for Pipeline.
        if has_group_filter(glql_yaml):
            return glql_http_error(PIPELINES_GROUP_SCOPE_ERROR)
        return GitLabHttpResponse(status_code=200, body=glql_response(SAMPLE_PIPELINES))

    mock_glql_responder(mock_gitlab_client, respond)

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "List the 20 most recent failed pipelines in the gitlab-org group "
        "with their ref, sha, and duration",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    queries = result.get_tool_call_args("run_glql_query", "glql_yaml")
    group_queries = [q for q in queries if has_group_filter(q)]

    if group_queries:
        # A group query was attempted: the agent must recover with a
        # project-scoped query, or explain the project-scope requirement.
        final_query_recovered = bool(
            queries
            and has_project_filter(queries[-1])
            and not has_group_filter(queries[-1])
        )
        if not final_query_recovered:
            await result.assert_llm_validates(
                [
                    "The response explains that listing individual pipelines "
                    "requires a project scope",
                ]
            )
    else:
        assert is_project_scoped(queries), (
            "Expected the pipeline queries to use a project filter when no "
            f"group filter was attempted. Queries: {queries}"
        )
