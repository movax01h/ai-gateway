"""Merge Request data source tests.

Validates that the agent generates correct GLQL queries for MergeRequest type, including query fields, display fields,
sorting, and scope requirements.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_MRS, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_mr_query_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use correct query fields for merge requests."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me merged MRs from the last month in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL query filters by state = merged",
            "The GLQL query filters merged using a relative time expression like -1m",
        ]
    )


@pytest.mark.asyncio
async def test_mr_draft_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by draft status."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all draft merge requests in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL query filters by draft = true",
        ]
    )


@pytest.mark.asyncio
async def test_mr_reviewer_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by reviewer."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me merge requests I need to review in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL query filters by reviewer = currentUser()",
        ]
    )


@pytest.mark.asyncio
async def test_mr_target_branch_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by target branch."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open MRs targeting the main branch in project gitlab-org/gitlab",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            'The GLQL query filters by targetBranch = "main"',
        ]
    )


@pytest.mark.asyncio
async def test_mr_approver_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by approver."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me merge requests I approved in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL query filters by approver = currentUser() or approvedBy = currentUser()",
        ]
    )


@pytest.mark.asyncio
async def test_mr_deployed_date_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by deployment date."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me merge requests deployed in the last week in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL query filters by deployed using a relative time expression like -1w",
        ]
    )


@pytest.mark.asyncio
async def test_mr_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use appropriate display fields for merge requests."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open MRs in the gitlab-org group with their title, author, reviewer, and source branch",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL embedded view fields include title, author, reviewer, and sourceBranch",
        ]
    )


@pytest.mark.asyncio
async def test_mr_sort_by_merged_date(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should sort by merged date when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me recently merged MRs in the gitlab-org group, most recent first",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL embedded view sorts by merged desc or mergedAt desc",
        ]
    )


@pytest.mark.asyncio
async def test_mr_sort_by_created(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should sort by creation date when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the oldest open MRs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = MergeRequest",
            "The GLQL embedded view sorts by created asc or createdAt asc",
        ]
    )
