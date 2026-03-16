"""Pagination Tests.

Validates proper pagination behavior for different query types.
"""

import pytest

from .helpers import (
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    generate_issues,
    glql_response,
    mock_glql_response,
)
from agent_tests.helpers import ask_agent


@pytest.mark.asyncio
async def test_count_query_single_call(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Count-only queries should NOT paginate (single tool call)."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=150))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many open issues are there in the gitlab-org group?",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_tool_call_count("run_glql_query", 1)
    await result.assert_llm_validates(
        [
            "The response provides a specific count/number of issues, i.e. 150 without paginating.",
        ]
    )


@pytest.mark.asyncio
async def test_limited_results_single_call(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Limited results should NOT paginate (single tool call)."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the last 20 merged MRs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_tool_call_count("run_glql_query", 1)
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'limit: 20' or similar to respect the user's requested limit",
        ]
    )


@pytest.mark.asyncio
async def test_full_analysis_paginates(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Full data analysis should paginate (multiple tool calls)."""
    page1 = glql_response(
        generate_issues(100),
        count=250,
        has_next_page=True,
        end_cursor="cursor_page_1",
    )
    page2 = glql_response(
        generate_issues(100),
        count=250,
        has_next_page=True,
        end_cursor="cursor_page_2",
    )
    page3 = glql_response(
        generate_issues(50),
        count=250,
        has_next_page=False,
    )
    mock_glql_response(mock_gitlab_client, [page1, page2, page3])

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Looking at all the issues in the gitlab-org group, analyse the main areas of work based on the issue title and description",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    result.assert_tool_call_count("run_glql_query", 3)
    await result.assert_llm_validates(
        [
            "The response provides analysis or categorisation of different work areas based on the issue data",
        ]
    )
