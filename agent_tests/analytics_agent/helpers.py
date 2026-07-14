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


def glql_error_response(
    error: str = "The requested field is not supported in this GitLab instance.",
) -> dict[str, Any]:
    """Build a GLQL API response representing a failure (e.g. unsupported field/filter).

    Mirrors the shape the real GLQL API returns when a query references something the instance doesn't expose, which is
    the signal the agent's prompt uses to fall back to Orbit.
    """
    return {
        "data": {
            "count": 0,
            "nodes": [],
            "pageInfo": {
                "hasNextPage": False,
                "hasPreviousPage": False,
                "endCursor": None,
                "startCursor": None,
            },
        },
        "error": error,
        "fields": [],
        "success": False,
    }


def glql_analytics_response(
    nodes: list[dict[str, Any]],
    count: int | None = None,
    has_next_page: bool = False,
    end_cursor: str | None = None,
) -> dict[str, Any]:
    """Build a GLQL analytics-mode API response.

    Analytics responses omit the top-level ``fields`` array that standard
    responses include. The Rust transform also flattens dimension keys into
    each node (e.g. ``{"language": "python", "totalCount": 1500, ...}``),
    so nodes should be passed in already-flat form to match the real API.

    Args:
        nodes: List of aggregated result items (flat dimension + metric keys)
        count: Total count (defaults to len(nodes))
        has_next_page: Whether there are more pages
        end_cursor: Cursor for next page

    Returns:
        GLQL analytics response dict
    """
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

SAMPLE_CODE_SUGGESTIONS = [
    {
        "language": "python",
        "totalCount": 1500,
        "usersCount": 25,
        "acceptanceRate": 0.72,
        "suggestionSizeSum": 45000,
        "acceptedCount": 1080,
        "rejectedCount": 420,
        "shownCount": 1500,
    },
    {
        "language": "ruby",
        "totalCount": 980,
        "usersCount": 18,
        "acceptanceRate": 0.65,
        "suggestionSizeSum": 29400,
        "acceptedCount": 637,
        "rejectedCount": 343,
        "shownCount": 980,
    },
    {
        "language": "javascript",
        "totalCount": 1200,
        "usersCount": 22,
        "acceptanceRate": 0.78,
        "suggestionSizeSum": 36000,
        "acceptedCount": 936,
        "rejectedCount": 264,
        "shownCount": 1200,
    },
]

SAMPLE_CODE_SUGGESTIONS_BY_IDE = [
    {
        "ideName": "VS Code",
        "totalCount": 2500,
        "usersCount": 40,
        "acceptanceRate": 0.75,
        "suggestionSizeSum": 75000,
        "acceptedCount": 1875,
        "rejectedCount": 625,
        "shownCount": 2500,
    },
    {
        "ideName": "JetBrains IDE",
        "totalCount": 800,
        "usersCount": 15,
        "acceptanceRate": 0.68,
        "suggestionSizeSum": 24000,
        "acceptedCount": 544,
        "rejectedCount": 256,
        "shownCount": 800,
    },
]

SAMPLE_PIPELINE_ANALYTICS_BY_REF = [
    {
        "ref": "main",
        "totalCount": 420,
        "successRate": 0.92,
        "failureRate": 0.06,
        "canceledRate": 0.01,
        "skippedRate": 0.01,
        "durationQuantile": 540,
    },
    {
        "ref": "develop",
        "totalCount": 215,
        "successRate": 0.81,
        "failureRate": 0.15,
        "canceledRate": 0.02,
        "skippedRate": 0.02,
        "durationQuantile": 612,
    },
]

SAMPLE_PIPELINE_ANALYTICS_BY_STATUS = [
    {"status": "success", "totalCount": 850},
    {"status": "failed", "totalCount": 92},
    {"status": "canceled", "totalCount": 18},
    {"status": "skipped", "totalCount": 4},
    {"status": "running", "totalCount": 12},
]

SAMPLE_PIPELINE_ANALYTICS_WEEKLY = [
    {"finished": "2026-04-13T00:00:00Z", "totalCount": 210, "successRate": 0.90},
    {"finished": "2026-04-20T00:00:00Z", "totalCount": 245, "successRate": 0.87},
    {"finished": "2026-04-27T00:00:00Z", "totalCount": 198, "successRate": 0.93},
]

SAMPLE_CONTRIBUTIONS_MONTHLY = [
    {"created": "2026-05-01", "totalCount": 320, "usersCount": 12},
    {"created": "2026-06-01", "totalCount": 410, "usersCount": 15},
    {"created": "2026-07-01", "totalCount": 275, "usersCount": 11},
]

SAMPLE_CONTRIBUTIONS_OVERALL = [
    {"totalCount": 1005, "usersCount": 22},
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
