"""Ranking tests.

Validates the Top N rule for analytics mode in the agent prompt, and the paging rule it sits next to: a top N needs one
query, a breakdown with more rows than `limit` needs every page.
"""

from typing import Any

import pytest
import yaml

from agent_tests.helpers import AgentResult, ask_agent
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse

from .helpers import dashboard_question, glql_analytics_response, mock_glql_responder


def _project(number: int, team: str) -> dict[str, Any]:
    path = f"gitlab-org/{team}/service-{number:03}"
    return {
        "id": f"gid://gitlab/Project/{1000 + number}",
        "name": f"Service {number:03}",
        "avatarUrl": None,
        "fullPath": path,
        "webPath": f"/{path}",
        "nameWithNamespace": f"GitLab.org / {team} / Service {number:03}",
    }


# DAP users by project, most first. Sessions started outside a project come back as a row with no project, and it
# ranks second.
_USERS_BY_PROJECT: list[dict[str, Any]] = [
    {"usersCount": 200, "project": _project(1, "plan")},
    {"usersCount": 180, "project": None},
    {"usersCount": 100, "project": _project(2, "create")},
    {"usersCount": 66, "project": _project(3, "verify")},
    {"usersCount": 50, "project": _project(4, "secure")},
    {"usersCount": 40, "project": _project(5, "platform")},
]
# The long tail, so that a query with `limit: 100` has a second page.
_USERS_BY_PROJECT += [
    {"usersCount": max(1, 39 - number // 4), "project": _project(number, "tail")}
    for number in range(6, 141)
]


def _users_by_project(glql_yaml: str, after: str | None) -> GitLabHttpResponse:
    """Answer any query with the rows above, one page of the query's `limit` at a time, as GLQL pages them."""
    limit = int(yaml.safe_load(glql_yaml).get("limit", 100))
    start = int(after or 0)
    end = start + limit
    has_next_page = end < len(_USERS_BY_PROJECT)
    return GitLabHttpResponse(
        status_code=200,
        body=glql_analytics_response(
            _USERS_BY_PROJECT[start:end],
            count=len(_USERS_BY_PROJECT),
            has_next_page=has_next_page,
            end_cursor=str(end) if has_next_page else None,
        ),
    )


def _pages_after_first(result: AgentResult) -> list[str]:
    """Return the `after` cursors the agent passed, so an empty list means it never asked for a second page."""
    # get_tool_call_args stringifies, so an explicit `after: null` comes back as "None".
    return [
        after
        for after in result.get_tool_call_args("run_glql_query", "after")
        if after not in ("", "None")
    ]


# 1.0.0 is left out: its bundled schema has no DuoWorkflows source, and dashboard context never reaches it, since
# the button that sends it shipped after GitLab pinned analytics_agent to ^2.0.0.
@pytest.mark.flow_versions("2.0.0")
@pytest.mark.asyncio
async def test_ranking_leaves_out_rows_without_value(
    analytics_agent_without_orbit,
    initial_state,
    mock_gitlab_client,
):
    """Should run one sorted query, and not rank the row with no project as a project."""
    mock_glql_responder(mock_gitlab_client, _users_by_project)

    result = await ask_agent(
        analytics_agent_without_orbit,
        initial_state,
        dashboard_question(
            "Which 3 projects had the most DAP users?", orbit_enabled=False
        ),
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    queries = [
        yaml.safe_load(q)
        for q in result.get_tool_call_args("run_glql_query", "glql_yaml")
    ]
    assert any(
        isinstance(q, dict)
        and q.get("sort") == "usersCount desc"
        and int(q.get("limit", 100)) > 3
        for q in queries
    ), queries
    assert not _pages_after_first(result), "A top N needs one query, not every page"
    await result.assert_llm_validates(
        [
            "The response names Service 001, Service 002 and Service 003, in that order, as the top 3 projects",
            "The response does not count the row with no project as one of the three",
            "The response says that it left out users of sessions outside any project, or how many there were",
        ]
    )


@pytest.mark.flow_versions("2.0.0")
@pytest.mark.asyncio
async def test_breakdown_fetches_every_page(
    analytics_agent_without_orbit,
    initial_state,
    mock_gitlab_client,
):
    """Should page through a breakdown that has more rows than one page holds."""
    mock_glql_responder(mock_gitlab_client, _users_by_project)

    result = await ask_agent(
        analytics_agent_without_orbit,
        initial_state,
        dashboard_question(
            "How are DAP users spread across all projects?", orbit_enabled=False
        ),
    )

    result.assert_has_tool_calls().assert_called_tool("run_glql_query")
    assert _pages_after_first(result), (
        "A breakdown with more rows than one page needs every page"
    )
