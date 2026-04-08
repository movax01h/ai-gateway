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


IDE_ADDITIONAL_CONTEXT = (
    "User added additional context below enclosed in "
    "<additional_context></additional_context> tags.\n\n"
    "<additional_context>\n"
    "    <agent_user_environment_os_info>\n"
    '    {"platform": "darwin", "architecture": "arm64"}\n'
    "    </agent_user_environment_os_info>\n"
    "    <agent_user_environment_shell_info>\n"
    '    {"shell_name": "zsh", "shell_type": "unix"}\n'
    "    </agent_user_environment_shell_info>\n"
    "</additional_context>\n\n"
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

SAMPLE_PIPELINES = [
    {
        "id": "gid://gitlab/Ci::Pipeline/1001",
        "iid": "101",
        "status": "failed",
        "ref": "main",
        "sha": "abc123def456",
        "source": "push",
        "duration": 320,
        "name": "Build & Test",
    },
    {
        "id": "gid://gitlab/Ci::Pipeline/1002",
        "iid": "102",
        "status": "success",
        "ref": "feature-branch",
        "sha": "789ghi012jkl",
        "source": "merge_request_event",
        "duration": 180,
        "name": "MR Pipeline",
    },
]

SAMPLE_JOBS = [
    {
        "id": "gid://gitlab/Ci::Build/2001",
        "name": "rspec unit",
        "stage": "test",
        "status": "failed",
        "duration": 95,
        "kind": "build",
        "failureMessage": "Exit code 1",
    },
    {
        "id": "gid://gitlab/Ci::Build/2002",
        "name": "deploy-production",
        "stage": "deploy",
        "status": "success",
        "duration": 42,
        "kind": "build",
        "failureMessage": None,
    },
    {
        "id": "gid://gitlab/Ci::Bridge/2003",
        "name": "trigger-downstream",
        "stage": "deploy",
        "status": "success",
        "duration": 5,
        "kind": "bridge",
        "failureMessage": None,
    },
]

SAMPLE_PROJECTS = [
    {
        "id": "gid://gitlab/Project/1",
        "name": "GitLab",
        "fullPath": "gitlab-org/gitlab",
        "visibility": "public",
        "starCount": 8500,
        "openIssuesCount": 45000,
        "openMergeRequestsCount": 1200,
    },
    {
        "id": "gid://gitlab/Project/2",
        "name": "GitLab Runner",
        "fullPath": "gitlab-org/gitlab-runner",
        "visibility": "public",
        "starCount": 2300,
        "openIssuesCount": 800,
        "openMergeRequestsCount": 50,
    },
]

EMPTY_RESPONSE: list[dict[str, Any]] = []

REALISTIC_ISSUE_TITLES = [
    "Fix authentication timeout in SSO login flow",
    "Add dark mode support for the dashboard",
    "Performance regression in search API",
    "Update documentation for CI/CD pipeline",
    "Security vulnerability in file upload",
    "Implement two-factor authentication",
    "Bug: merge request approval not working",
    "Add GraphQL support for project queries",
    "Improve error handling in webhook delivery",
    "Refactor database connection pooling",
    "Add audit logging for admin actions",
    "Fix memory leak in background jobs",
    "Implement rate limiting for API endpoints",
    "Update dependencies to latest versions",
    "Add support for custom project templates",
    "Bug: notification emails not being sent",
    "Improve accessibility of the web interface",
    "Add bulk operations for issue management",
    "Fix timezone handling in scheduled jobs",
    "Implement caching for frequently accessed data",
]


def generate_issues(count: int) -> list[dict[str, Any]]:
    """Generate a list of sample issues with realistic titles."""
    return [
        {
            "id": f"gid://gitlab/Issue/{i}",
            "iid": str(i),
            "title": REALISTIC_ISSUE_TITLES[i % len(REALISTIC_ISSUE_TITLES)],
            "state": "opened",
        }
        for i in range(1, count + 1)
    ]
