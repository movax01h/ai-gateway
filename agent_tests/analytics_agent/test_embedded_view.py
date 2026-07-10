"""Embedded View Format Tests.

Validates proper YAML structure for embedded views.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_CODE_SUGGESTIONS,
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    SAMPLE_PIPELINE_ANALYTICS_WEEKLY,
    glql_analytics_response,
    glql_response,
    mock_glql_response,
)


@pytest.mark.asyncio
async def test_embedded_view_has_required_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should include all required embedded view fields."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Create a dashboard view of recent MRs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view includes a 'display' field (table, list, or orderedList)",
            "The GLQL embedded view includes a 'fields' parameter specifying which columns to show",
            "The GLQL embedded view includes a 'query' parameter with the actual GLQL filter",
        ]
    )


@pytest.mark.asyncio
async def test_ordered_list_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use orderedList display type for numbered lists."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Give me a numbered list of open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response presents the issues as a numbered list using 'display: orderedList' in the GLQL query",
        ]
    )


@pytest.mark.asyncio
async def test_list_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use list display type for regular lists."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Give me a list of open MRs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response presents the MRs as a list using 'display: list` or `table` in the GLQL query",
        ]
    )


@pytest.mark.asyncio
async def test_column_chart_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use columnChart display type for visualization requests."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_CODE_SUGGESTIONS),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me a column chart of code suggestion acceptance rates by language in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses 'display: columnChart' in the GLQL embedded view",
            "The GLQL query uses analytics mode (mode: analytics)",
        ]
    )


@pytest.mark.asyncio
async def test_bar_chart_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use barChart display type for horizontal bar comparison requests."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_CODE_SUGGESTIONS),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me a horizontal bar chart of code suggestion acceptance rates by language in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses 'display: barChart' in the GLQL embedded view",
            "The GLQL query uses analytics mode (mode: analytics)",
        ]
    )


@pytest.mark.asyncio
async def test_line_chart_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use lineChart display type for trends over a single dimension."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response(SAMPLE_PIPELINE_ANALYTICS_WEEKLY),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me a line chart of pipeline success rate over time in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses 'display: lineChart' in the GLQL embedded view",
            "The GLQL query uses analytics mode (mode: analytics)",
            "The GLQL query specifies exactly one dimension",
        ]
    )


@pytest.mark.asyncio
async def test_stat_display_type(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use stat display type for single-value requests."""
    mock_glql_response(
        mock_gitlab_client,
        glql_analytics_response([{"totalCount": 3680}]),
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the total number of code suggestions in the gitlab-org group as a single stat",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses 'display: stat' in the GLQL embedded view",
            "The GLQL query uses analytics mode (mode: analytics)",
            "The GLQL query requests a single metric and no dimensions",
        ]
    )


@pytest.mark.asyncio
async def test_title_field_included_when_requested(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should include title when creating named views."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Create a view called 'Critical Bugs' for high priority bugs in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view includes a 'title' field with 'Critical Bugs' or similar",
        ]
    )
