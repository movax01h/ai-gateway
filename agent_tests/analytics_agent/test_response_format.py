"""Response Format Tests.

Validates analytical (answer-first) vs query (embedded view) response patterns,
including IDE-specific rendering (standard Markdown instead of GLQL blocks).
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    IDE_ADDITIONAL_CONTEXT,
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    glql_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_ide_analytical_question_answers_first(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """In IDE, analytical questions should answer first with Markdown, not GLQL blocks."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS, count=15))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        IDE_ADDITIONAL_CONTEXT
        + "How many merge requests were merged this month in the gitlab-org group?",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response provides a direct answer or count BEFORE showing any query",
            "Response includes a collapsible/details section containing the underlying GLQL query as a yaml code block",
            "Response does NOT contain a ```glql code block",
            "Response does NOT suggest clicking on a menu or ⋮ icon",
        ]
    )


@pytest.mark.asyncio
async def test_ide_query_request_uses_markdown(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """In IDE, query requests should use standard Markdown instead of GLQL blocks."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        IDE_ADDITIONAL_CONTEXT
        + "Write a GLQL query for open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "Response includes the GLQL query as a yaml code block, not in a collapsible section",
            "Response does NOT contain a ```glql code block",
            "Response does NOT suggest clicking on a menu or ⋮ icon",
        ]
    )


@pytest.mark.asyncio
async def test_ide_visualization_request_uses_markdown(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """In IDE, visualization requests should use standard Markdown instead of GLQL blocks."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        IDE_ADDITIONAL_CONTEXT + "Show me all open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "Response presents the data in standard Markdown format (e.g. a table or list)",
            "Response includes the GLQL query as a yaml code block in a collapsible section",
            "Response does NOT contain a ```glql code block",
            "Response does NOT suggest clicking on a menu or ⋮ icon",
        ]
    )
