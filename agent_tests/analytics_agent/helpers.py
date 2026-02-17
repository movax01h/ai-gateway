"""Analytics-agent-specific test helpers.

Provides GLQL mock data and response builders for analytics agent tests.
"""

from typing import Any
from unittest.mock import AsyncMock

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse


def glql_response(
    nodes: list[dict[str, Any]],
    count: int | None = None,
    has_next_page: bool = False,
    end_cursor: str | None = None,
) -> dict[str, Any]:
    """Build a GLQL API response matching the real GitLab GLQL API shape.

    Args:
        nodes: List of result items (issues, MRs, etc.)
        count: Total count (defaults to len(nodes))
        has_next_page: Whether there are more pages
        end_cursor: Cursor for next page

    Returns:
        GLQL response dict
    """
    keys = list(nodes[0].keys()) if nodes else []
    fields = [{"key": k, "label": k.replace("_", " ").title(), "name": k} for k in keys]

    return {
        "data": {
            "count": count if count is not None else len(nodes),
            "nodes": nodes,
            "pageInfo": {
                "hasNextPage": has_next_page,
                "hasPreviousPage": False,
                "endCursor": end_cursor,
                "startCursor": None,
            },
        },
        "error": None,
        "fields": fields,
        "success": True,
    }


def mock_glql_response(
    mock_gitlab_client: AsyncMock,
    response: dict[str, Any] | list[dict[str, Any]],
) -> None:
    """Configure mock GitLab client to return GLQL response(s).

    Args:
        mock_gitlab_client: The mocked GitLab client
        response: Single response dict or list of responses for pagination
    """
    if isinstance(response, list):
        mock_gitlab_client.apost.side_effect = [
            GitLabHttpResponse(status_code=200, body=r) for r in response
        ]
    else:
        mock_gitlab_client.apost.return_value = GitLabHttpResponse(
            status_code=200, body=response
        )


SAMPLE_ISSUES = [
    {"id": "gid://gitlab/Issue/1", "iid": "1", "title": "Bug fix", "state": "opened"},
    {"id": "gid://gitlab/Issue/2", "iid": "2", "title": "Feature", "state": "opened"},
    {"id": "gid://gitlab/Issue/3", "iid": "3", "title": "Docs", "state": "opened"},
]

SAMPLE_MRS = [
    {
        "id": "gid://gitlab/MergeRequest/1",
        "iid": "1",
        "title": "Fix bug",
        "state": "merged",
    },
    {
        "id": "gid://gitlab/MergeRequest/2",
        "iid": "2",
        "title": "Add feature",
        "state": "merged",
    },
]
