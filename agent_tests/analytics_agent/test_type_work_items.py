"""Work Item data source tests.

Validates that the agent generates correct GLQL queries for Work Item types (Issue, Epic, Task, etc.), including query
fields, display fields, sorting, and scope requirements.
"""

import pytest

from agent_tests.helpers import ask_agent

from .helpers import SAMPLE_ISSUES, glql_response, mock_glql_response


@pytest.mark.asyncio
async def test_work_item_type_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by work item type using the type field."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me all open tasks in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query includes type = Task",
            "The GLQL query filters by state = opened",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_date_filters(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use date comparison operators for created and due fields."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues created in the last month that are due within a week in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query filters created using a relative time expression like -1m",
            "The GLQL query filters due using a relative time expression like 1w",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_health_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by health status."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        'Show me issues with health status "needs attention" in the gitlab-org group',
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            'The GLQL query filters by health = "needs attention"',
        ]
    )


@pytest.mark.asyncio
async def test_work_item_iteration_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by iteration."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues in the current iteration in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query filters by iteration = current",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_weight_filter(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should filter by weight."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me issues with weight 5 in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL query filters by weight = 5",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_display_fields(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should use appropriate display fields for work items."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues in the gitlab-org group with their title, assignee, labels, milestone and due date",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view fields include title, assignee, labels, milestone, and due",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_sort_by_due_date(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should sort by due date when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me open issues in the gitlab-org group sorted by due date, earliest first",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view sorts by due asc or dueDate asc",
        ]
    )


@pytest.mark.asyncio
async def test_work_item_sort_by_popularity(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should sort by popularity when requested."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Show me the most popular open issues in the gitlab-org group",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    await result.assert_llm_validates(
        [
            "The GLQL embedded view sorts by popularity desc",
        ]
    )


@pytest.mark.asyncio
async def test_epic_sort_by_milestone_not_supported(
    analytics_agent,
    initial_state,
    mock_gitlab_client,
):
    """Should warn that milestone sort is not supported for epics."""
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES))

    result = await ask_agent(
        analytics_agent,
        initial_state,
        "Can you show all epics in the gitlab-org group, sorted by milestone?",
    )

    result.assert_has_tool_calls().assert_called_tool("get_glql_schema")
    await result.assert_llm_validates(
        [
            "The response explains that sorting by milestone "
            "is not supported for epics, or uses an alternative sort field",
        ]
    )
