"""Analytics-agent-specific test helpers.

Provides GLQL mock data and response builders for analytics agent tests.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import AsyncMock

import yaml

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig
from duo_workflow_service.agents.prompt_adapter import ChatAgentPromptTemplate
from duo_workflow_service.entities.message_ingestion import assemble_user_message
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.version import resolve_version

# Real production error from the GLQL compiler when a Pipeline query uses a
# group filter (POST api/v4/glql returns it with HTTP 400). Captured verbatim
# from a live GDK instance's compiler on 2026-08-25.
PIPELINES_GROUP_SCOPE_ERROR = (
    "Error: `group` is not a recognized field for pipelines. "
    "Valid fields: status, ref, sha, source, scope, updated, author."
)


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


def _query_line(glql_yaml: str) -> str:
    """The `query:` line of a GLQL block, or the whole string when there is none.

    The GLQL endpoint also accepts a bare query expression with no YAML keys
    (Analytics::Glql::ParserService treats non-hash input as the query), so the
    fallback keeps filter checks working for that shape. When a `query:` key is
    present, only that line is scanned so titles or field lists cannot
    false-positive.
    """
    match = re.search(r"^\s*query:(.*)$", glql_yaml, re.MULTILINE)
    return match.group(1) if match else glql_yaml


def has_group_filter(glql_yaml: str) -> bool:
    """Whether the GLQL query line filters on group (via =, != or in)."""
    return bool(re.search(r"\bgroup\s*(!?=|in\b)", _query_line(glql_yaml)))


def has_project_filter(glql_yaml: str) -> bool:
    """Whether the GLQL query line filters on project (via =, != or in)."""
    return bool(re.search(r"\bproject\s*(!?=|in\b)", _query_line(glql_yaml)))


def is_project_scoped(queries: list[str]) -> bool:
    """Whether at least one query filters on project and none filters on group."""
    return (
        bool(queries)
        and any(has_project_filter(q) for q in queries)
        and not any(has_group_filter(q) for q in queries)
    )


def glql_http_error(message: str, status_code: int = 400) -> GitLabHttpResponse:
    """Build the HTTP error response the GLQL endpoint returns for compile failures.

    The real api/v4/glql endpoint answers a rejected query with a non-2xx status and body ``{"error": message}``.

    Args:
        message: The compiler error message
        status_code: HTTP status code (defaults to 400)

    Returns:
        GitLabHttpResponse carrying the error body
    """
    return GitLabHttpResponse(status_code=status_code, body={"error": message})


def mock_glql_responder(
    mock_gitlab_client: AsyncMock,
    fn: Callable[[str, str | None], GitLabHttpResponse],
) -> None:
    """Configure the mock GitLab client to answer each GLQL call via `fn`.

    `fn` receives the posted glql_yaml string and `after` cursor (None for the
    first page) and returns a GitLabHttpResponse, so a test can make the
    response depend on the query, or serve real pages of one result. Example:
    return the 400 group-scope error when the query uses a group filter,
    success otherwise:

        def respond(glql_yaml: str, _after: str | None) -> GitLabHttpResponse:
            if has_group_filter(glql_yaml):
                return glql_http_error(PIPELINES_GROUP_SCOPE_ERROR)
            return GitLabHttpResponse(
                status_code=200, body=glql_response(SAMPLE_PIPELINES)
            )

        mock_glql_responder(mock_gitlab_client, respond)

    Args:
        mock_gitlab_client: The mocked GitLab client
        fn: Callable mapping the posted glql_yaml string and `after` cursor to a response
    """

    async def apost(path: str, body: str, **_kwargs: Any) -> GitLabHttpResponse:
        del path  # unused - the mock answers every GLQL POST
        request = json.loads(body)
        return fn(request["glql_yaml"], request.get("after"))

    mock_gitlab_client.apost = AsyncMock(side_effect=apost)


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


_DAP_QUERY = 'type = DuoWorkflow and created >= "2026-09-02" and created <= "2026-10-02" and group = "gitlab-org"'

# A representative dashboard_context, shaped like what the "Analyze with Duo" button sends. It carries no
# additional_instructions, so tests measure what the agent prompt does on its own.
DASHBOARD_CONTEXT: dict[str, Any] = {
    "dashboard_scope": {"groups": ["gitlab-org"], "projects": []},
    "date_range": {
        "option": "30d",
        "start_date": "2026-09-02",
        "end_date": "2026-10-02",
        "previous": {"start_date": "2026-08-02", "end_date": "2026-09-01"},
    },
    "panels": [
        {
            "title": "Active projects",
            "description": "Distinct projects with at least one GitLab Duo Agent Platform (DAP) session in the selected date "
            "range. Sessions not scoped to a project are not counted.",
            "glql": f"display: stat\nmode: analytics\nquery: {_DAP_QUERY}\nmetrics: projectsCount\n"
            "displayConfig:\n  description: Projects with at least one DAP session\n",
        },
        {
            "title": "Active users",
            "description": "Distinct users who ran at least one GitLab Duo Agent Platform (DAP) session in the selected "
            "date range.",
            "glql": f"display: stat\nmode: analytics\nquery: {_DAP_QUERY}\nmetrics: usersCount\n"
            "displayConfig:\n  description: Distinct users with at least one DAP session\n",
        },
        {
            # Not a dap_impact.yaml panel. It sets up the ranking case, where sessions outside any project rank second.
            "title": "Users by project",
            "description": "Distinct users per project.",
            "glql": f"display: table\nmode: analytics\nquery: {_DAP_QUERY}\ndimensions: project\nmetrics: usersCount\n"
            "sort: usersCount desc\n",
        },
        {
            # barList is a dashboard panel display, as in dap_impact.yaml, not a GLQL embedded view display type.
            "title": "Where credits went",
            "description": "Credits consumed by each GitLab Duo Agent Platform (DAP) flow type, such as Chat or Software "
            "Development, in the selected date range. Credits are fetched from billing after a session is billed and can "
            "lag behind the session counts, so recent totals can understate spend.",
            "glql": f"display: barList\nmode: analytics\nquery: {_DAP_QUERY}\ndimensions: flowType\n"
            "metrics: creditsUsedSum\nsort: creditsUsedSum desc\ndisplayConfig:\n  maxRows: 30\n",
        },
    ],
}

# Duo Chat renders the user turn with the chat/agent prompt, resolved as chat_agent_factory.py resolves it. The tests
# run on Claude, so this takes the claude family's definition.
_CHAT_PROMPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "ai_gateway"
    / "prompts"
    / "definitions"
    / "chat"
    / "agent"
    / "claude"
)
_CHAT_PROMPT_VERSION = "^1.0.0"


@lru_cache(maxsize=1)
def _chat_prompt_template() -> ChatAgentPromptTemplate:
    versions = {path.stem: path for path in _CHAT_PROMPT_DIR.glob("*.yml")}
    version = resolve_version(versions, _CHAT_PROMPT_VERSION)
    config = PromptConfig(**yaml.safe_load(versions[version].read_text()))
    return ChatAgentPromptTemplate(ModelClassProvider.ANTHROPIC, config)


def dashboard_question(
    question: str,
    context: dict[str, Any] | None = None,
    *,
    orbit_enabled: bool = True,
) -> str:
    """Return a question asked from a dashboard, as the agent receives it.

    Args:
        question: The question the user asks
        context: The dashboard_context the button sends. Defaults to DASHBOARD_CONTEXT.
        orbit_enabled: The value the orbit_context item reports

    Returns:
        The user turn Duo Chat's prompt template renders, with both context items before the question.
    """
    # The button sends no ids, so neither item gets tags of its own. orbit_context comes first, as in
    # duo_agentic_chat_state_manager.vue.
    additional_context = [
        AdditionalContext(
            category="orbit_context",
            content=json.dumps({"orbit_enabled": orbit_enabled}),
        ),
        AdditionalContext(
            category="dashboard_context",
            content=json.dumps(DASHBOARD_CONTEXT if context is None else context),
        ),
    ]
    # The template reads only the conversation history from the state.
    state = cast(
        ChatWorkflowState,
        {
            "conversation_history": {
                "chat": [assemble_user_message(question, additional_context)]
            }
        },
    )
    *_system, turn = (
        _chat_prompt_template().invoke(state, agent_name="chat").to_messages()
    )
    return str(turn.content)


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

SAMPLE_PIPELINE_ANALYTICS_PARAMETERISED = [
    {
        "finished(weekly)": "2026-04-13T00:00:00Z",
        "totalCount": 210,
        "successRate": 0.90,
        "durationQuantile(0.5)": 540,
    },
    {
        "finished(weekly)": "2026-04-20T00:00:00Z",
        "totalCount": 245,
        "successRate": 0.87,
        "durationQuantile(0.5)": 612,
    },
    {
        "finished(weekly)": "2026-04-27T00:00:00Z",
        "totalCount": 198,
        "successRate": 0.93,
        "durationQuantile(0.5)": 498,
    },
]

SAMPLE_PIPELINE_ANALYTICS_PERCENTILES = [
    {
        "ref": "main",
        "durationQuantile(0.5)": 540,
        "durationQuantile(0.95)": 1200,
    },
    {
        "ref": "develop",
        "durationQuantile(0.5)": 612,
        "durationQuantile(0.95)": 1450,
    },
]


EMPTY_RESPONSE: list[dict[str, Any]] = []

# Product areas and kinds of work for generate_issues. 25 areas x 10 kinds give 250 distinct issues, so a paginated
# analysis never sees the same title twice. Repeated titles or pages are what a model flags as synthetic data.
_ISSUE_AREAS = [
    ("SSO login", "devops::govern"),
    ("merge request approvals", "devops::create"),
    ("pipeline editor", "devops::verify"),
    ("issue boards", "devops::plan"),
    ("container registry", "devops::package"),
    ("webhook delivery", "devops::foundations"),
    ("Gitaly RPCs", "devops::systems"),
    ("search API", "devops::foundations"),
    ("project imports", "devops::foundations"),
    ("group milestones", "devops::plan"),
    ("dependency scanning", "devops::secure"),
    ("audit events", "devops::govern"),
    ("runner registration", "devops::verify"),
    ("wiki pages", "devops::plan"),
    ("code owners", "devops::create"),
    ("vulnerability report", "devops::secure"),
    ("package registry", "devops::package"),
    ("release notes", "devops::deploy"),
    ("deploy keys", "devops::create"),
    ("notification emails", "devops::plan"),
    ("epic roadmap", "devops::plan"),
    ("Service Desk", "devops::plan"),
    ("repository mirroring", "devops::create"),
    ("value stream analytics", "devops::plan"),
    ("merge trains", "devops::verify"),
]

# (title, description, type label); {area} is filled from _ISSUE_AREAS.
_ISSUE_WORK_KINDS = [
    (
        "Fix timeout in {area}",
        "Requests to {area} time out after 60 seconds on large projects since the 19.2 release.",
        "type::bug",
    ),
    (
        "Performance regression in {area}",
        "p95 latency for {area} roughly doubled after 19.2. Profiling points at an N+1 query.",
        "type::bug",
    ),
    (
        "Improve error handling in {area}",
        "Failures in {area} show a generic 500 page. Users need an error message they can act on.",
        "type::maintenance",
    ),
    (
        "Add GraphQL support for {area}",
        "Only the REST API exposes {area}. Add GraphQL types and mutations so the frontend can migrate.",
        "type::feature",
    ),
    (
        "Add audit logging for {area}",
        "Changes to {area} are not recorded as audit events, which compliance customers need as evidence.",
        "type::feature",
    ),
    (
        "Update documentation for {area}",
        "The {area} docs still describe the behavior before 19.0 and miss the new settings.",
        "type::maintenance",
    ),
    (
        "Regression in {area} after upgrading to 19.3",
        "Several self-managed customers report that {area} stopped working after upgrading to 19.3.",
        "type::bug",
    ),
    (
        "Refactor {area} service layer",
        "The {area} code mixes controller and service logic. Split it so it can be unit tested.",
        "type::maintenance",
    ),
    (
        "Add rate limiting to {area}",
        "A single client can saturate {area} with requests. Add a per-user rate limit.",
        "type::feature",
    ),
    (
        "Accessibility issues in {area}",
        "Screen readers cannot reach the main actions in {area}, and focus order skips the dialogs.",
        "type::bug",
    ),
]


def generate_issues(count: int, start: int = 1) -> list[dict[str, Any]]:
    """Generate sample issues with distinct titles, descriptions and labels.

    Args:
        count: Number of issues to generate
        start: IID of the first issue. Pages of one result set pass consecutive ranges so no IID repeats.

    Returns:
        Issues with IIDs start..start+count-1. Up to 250 consecutive issues have distinct titles.
    """
    issues = []
    for iid in range(start, start + count):
        index = iid - 1
        area, stage_label = _ISSUE_AREAS[index % len(_ISSUE_AREAS)]
        # Offsetting the kind by the area position spreads every kind over every area, so each block of 25 issues
        # mixes kinds and 250 consecutive issues never repeat an (area, kind) pair.
        kind = (index // len(_ISSUE_AREAS) + index % len(_ISSUE_AREAS)) % len(
            _ISSUE_WORK_KINDS
        )
        title, description, type_label = _ISSUE_WORK_KINDS[kind]
        issues.append(
            {
                "id": f"gid://gitlab/Issue/{iid}",
                "iid": str(iid),
                "title": title.format(area=area),
                "description": description.format(area=area),
                "state": "opened",
                "labels": {"nodes": [{"title": type_label}, {"title": stage_label}]},
            }
        )
    return issues
