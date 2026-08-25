"""GLQL Syntax Tests.

Validates adherence to GLQL syntax rules, restrictions, operators, and functions that apply across all data sources.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import (
    SAMPLE_CODE_SUGGESTIONS,
    SAMPLE_ISSUES,
    SAMPLE_MRS,
    SAMPLE_PROJECTS,
    glql_analytics_response,
    glql_response,
    mock_glql_response,
)


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_invalid_sort_field_uses_valid_alternative(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use valid sort field instead of assignee."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues sorted by assignee in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response uses a valid sort field (like created, updated, title) "
            "instead of assignee and explains that sorting by assignee is not supported"
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_label_syntax_with_tilde_prefix(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use proper label syntax with ~ prefix."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Find all issues with the priority-high label in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            'The GLQL query uses the ~ prefix for labels (e.g., ~priority-high or ~"priority-high")',
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_milestone_syntax_with_percent_prefix(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use proper milestone syntax with % prefix."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues in milestone v1.0 in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            'The GLQL query uses the % prefix for milestones (e.g., %v1.0 or %"v1.0")',
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_limit_maximum_of_100(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should respect limit maximum of 100."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the last 101 issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query has a limit of 100 or less.",
            "The response mentions that 100 is the maximum allowed limit",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_and_logic_for_multiple_labels(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use AND logic correctly for multiple labels."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Find issues with both ~bug and ~security labels in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses multiple 'label = ~x and label = ~y' conditions for AND logic.",
            "It does not use 'label in (~bug, ~security)' which would be OR logic",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_labels_in_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should include labels in fields when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues with their labels displayed in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes 'labels' in the fields parameter",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_negation_operator(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use != operator correctly for negation."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues not assigned to me in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses 'assignee != currentUser()'",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_current_user_for_my_queries(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use currentUser() for 'my' queries."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_MRS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me my open merge requests in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The response indicates filtering by the current user, evidenced by: currentUser() in the GLQL query",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_current_user_for_assigned_items(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Should use currentUser() for assigned items."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "What issues from the gitlab-org group are assigned to me?",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses 'assignee = currentUser()' to filter by the current user",
        ]
    )


@pytest.mark.flow_versions("1.0.0", "2.0.0")
@pytest.mark.asyncio
async def test_analytics_mode_not_used_on_standard_only_source(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """An analytics-shaped question must still respect the source's modes.

    Work items are standard-only, so aggregating them server-side is not
    available however the question is phrased.
    """
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Break down open issues in the gitlab-org group by milestone, as a column chart",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    await result.assert_llm_validates(
        [
            "No GLQL query in the response uses 'mode: analytics'",
            "No GLQL query in the response specifies 'dimensions' or 'metrics'",
            "No GLQL block uses an aggregate display type - no 'display: stat', "
            "'columnChart', 'barChart', 'lineChart' or 'areaChart'",
        ]
    )


@pytest.mark.flow_versions("2.0.0")
@pytest.mark.asyncio
async def test_analytics_mode_selects_with_dimensions_not_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """Analytics modes have no display fields, so `fields` does not apply.

    1.0.0 gets this from `test_type_code_suggestions`, which reads the schema
    bundled in this repo rather than the one the instance serves.
    """
    mock_glql_response(
        mock_gitlab_client, glql_analytics_response(SAMPLE_CODE_SUGGESTIONS)
    )

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me code suggestion statistics by language in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query uses mode: analytics with dimensions and metrics, not fields",
        ]
    )


@pytest.mark.flow_versions("2.0.0")
@pytest.mark.asyncio
async def test_sort_restriction_direction_is_respected(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
    schema_tool_name,
):
    """A `sort_restrictions` entry allows one direction only.

    Only 2.0.0: the bundled schema has no way to say this, so 1.0.0 relies on
    `test_type_projects` asserting the direction directly.
    """
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_PROJECTS))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me projects in the gitlab-org group sorted by most recently active",
    )

    result.assert_has_tool_calls().assert_called_tool(schema_tool_name)
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query sorts by lastActivity descending, not ascending",
        ]
    )
