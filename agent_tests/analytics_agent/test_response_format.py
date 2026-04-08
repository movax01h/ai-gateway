"""Response Format Tests.

Validates response format
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
async def test_analytical_question_answers_first(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Analytical questions should provide answer BEFORE showing GLQL query."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS, count=15))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "How many merge requests were merged this month in the gitlab-org group?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response provides a direct answer or count BEFORE showing any GLQL query",
            "The GLQL query appears inside a collapsible/details section",
        ]
    )


@pytest.mark.asyncio
async def test_query_request_shows_embedded_view(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Query requests should show embedded view format prominently."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Write a GLQL query for open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response contains a ```glql code block with the query visible directly in the response",
            "The response uses embedded view format with display, fields, and query parameters",
        ]
    )


@pytest.mark.asyncio
async def test_visualization_request_shows_embedded_view(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Visualization requests should show embedded view format prominently."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response contains a ```glql code block with the query visible directly in the response",
            "The response uses embedded view format with display, fields, and query parameters",
        ]
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

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
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

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
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

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "Response presents the data in standard Markdown format (e.g. a table or list)",
            "Response includes the GLQL query as a yaml code block in a collapsible section",
            "Response does NOT contain a ```glql code block",
            "Response does NOT suggest clicking on a menu or ⋮ icon",
        ]
    )
