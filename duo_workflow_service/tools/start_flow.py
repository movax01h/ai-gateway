import json
from typing import Annotated, Any, ClassVar, Literal, Optional, Type, Union
from urllib.parse import unquote

import structlog
from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger(__name__)


FLOW_IDENTIFIER_MAP = {
    "developer": "developer/v1",
    "fix_pipeline": "fix_pipeline/v1",
    "code_review": "code_review/v1",
}


class StartFlowError(ToolException):
    """Raised when starting a flow fails.

    ``response`` is read by ToolsExecutor._handle_tool_error to surface the
    failure reason in the UI chat log, not just the LLM-facing message.

    Args:
        message: Human-readable description of the failure, forwarded to the LLM.
        response: Optional user-facing reason surfaced in the UI chat log. Defaults to None.
    """

    def __init__(self, message: str, response: Optional[str] = None):
        super().__init__(message)
        self.response = response


class StartDeveloperFlowInput(BaseModel):
    """Input for the developer agent."""

    name: Literal["developer"]
    goal: str = Field(
        description=(
            "Task description for the agent — include intent, relevant "
            "GitLab URLs, and context from chat the agent would otherwise miss."
        ),
    )
    project_url: Optional[str] = Field(
        default=None,
        description=(
            "Full URL of the GitLab project to run the task in. "
            "Omit to use the project from the current chat context."
        ),
    )
    issue_url: Optional[str] = Field(
        default=None,
        description=(
            "Full URL of the GitLab issue or work item the task relates to "
            "(e.g. https://gitlab.com/group/project/-/issues/42). "
            "When provided, the workflow session is explicitly linked to "
            "this issue so progress is tracked there."
        ),
    )


class StartFixPipelineFlowInput(BaseModel):
    """Input for the fix_pipeline agent."""

    name: Literal["fix_pipeline"]
    pipeline_url: str = Field(description="Full URL of the failing pipeline.")
    merge_request_url: str = Field(
        description="Full URL of the merge request the pipeline ran for.",
    )
    source_branch: str = Field(
        description="Branch the failing pipeline ran on.",
    )


class StartCodeReviewFlowInput(BaseModel):
    """Input for the code_review agent."""

    name: Literal["code_review"]
    merge_request_url: str = Field(
        description="Full URL of the merge request to review.",
    )


class StartFlowInput(BaseModel):
    """Input schema for the start_flow tool."""

    flow: Annotated[
        Union[
            StartDeveloperFlowInput,
            StartFixPipelineFlowInput,
            StartCodeReviewFlowInput,
        ],
        Field(discriminator="name"),
    ]


class StartFlow(DuoBaseTool):
    name: str = "start_flow"
    tool_version: ClassVar[Version] = Version("0.0.1")
    description: str = (
        "Delegate a task to a specialist GitLab agent that works "
        "asynchronously over multiple steps. Always use this tool when the "
        "user's request matches one of the agents below — specialist "
        "agents are purpose-built for their domain and deliver better "
        "outcomes than inline handling.\n"
        "\n"
        "Available agents:\n"
        "- developer: General-purpose agent for tasks that involve writing or "
        "changing code, resolving an issue, or implementing changes described "
        "in chat. The `goal` is the agent's only briefing — it does not see this "
        "conversation, so include the user's intent, any relevant GitLab URLs, "
        "and context from chat the agent would otherwise miss.\n"
        "- fix_pipeline: fixing, debugging, or investigating a failing pipeline or broken "
        'build (e.g. "fix this pipeline", "the build is broken").\n'
        "- code_review: read-only review and assessment of the changes in a merge request — "
        "not for implementing feedback or making changes "
        '(e.g. "review this MR", "can you review my merge request?").\n'
        "Returns a session URL the user can follow to track progress. The user is prompted to "
        "approve the handoff before the agent starts."
    )
    args_schema: Type[BaseModel] = StartFlowInput

    async def _execute(
        self,
        flow: (
            StartDeveloperFlowInput
            | StartFixPipelineFlowInput
            | StartCodeReviewFlowInput
        ),
        **_kwargs: Any,
    ) -> str:
        if isinstance(flow, BaseModel):
            flow_data = flow.model_dump()
        else:
            raise ToolException(f"Unexpected flow input type: {type(flow)}")

        flow_name = flow_data["name"]
        backend_flow_id = FLOW_IDENTIFIER_MAP.get(flow_name)
        if not backend_flow_id:
            raise ToolException(f"Unknown flow: {flow_name!r}")

        effective_goal, project_id, linkable_ids = (
            self._resolve_goal_project_and_linkable(flow_name, flow_data)
        )

        payload: dict[str, Any] = {
            "workflow_definition": backend_flow_id,
            "goal": effective_goal,
            "environment": "ambient",
            "start_workflow": True,
        }

        if project_id:
            payload["project_id"] = project_id

        if linkable_ids.get("issue_id"):
            payload["issue_id"] = linkable_ids["issue_id"]
        if linkable_ids.get("merge_request_id"):
            payload["merge_request_id"] = linkable_ids["merge_request_id"]

        if flow_name == "fix_pipeline":
            payload["additional_context"] = [
                {
                    "Category": "merge_request",
                    "Content": json.dumps({"url": flow_data["merge_request_url"]}),
                },
                {
                    "Category": "pipeline",
                    "Content": json.dumps(
                        {"source_branch": flow_data["source_branch"]}
                    ),
                },
            ]

        response = await self.gitlab_client.apost(
            path="/api/v4/ai/duo_workflows/agent_workflows",
            body=json.dumps(payload),
        )

        if not response.is_success():
            log.error(
                "start_flow: failed to create workflow",
                status_code=response.status_code,
                body=response.body,
                workflow_definition=flow_name,
            )
            if response.status_code == 403:
                detail = (
                    "This flow isn't available, or you don't have sufficient "
                    "permissions to start it."
                )
            else:
                detail = "An internal error occurred while starting the flow."
            raise StartFlowError(
                f"Failed to start flow: HTTP {response.status_code}: {detail}",
                response=detail,
            )

        body = response.body
        if isinstance(body, str):
            body = json.loads(body)

        workflow_id = body.get("id")
        session_url = (
            f"{self.project['web_url']}/-/automate/agent-sessions/{workflow_id}"
            if self.project and workflow_id
            else None
        )

        return json.dumps(
            {
                "status": "started",
                "workflow_id": workflow_id,
                "session_url": session_url,
                "flow_name": flow_name,
            }
        )

    def _resolve_goal_project_and_linkable(
        self, flow_name: str, flow_data: dict
    ) -> tuple[str, Optional[str | int], dict[str, int]]:
        """Return the goal, project identifier, and linkable IDs for the flow.

        For flows that accept a URL (fix_pipeline, code_review), the project
        is extracted from the URL so the workflow runs against the correct
        project — even when it differs from the current chat context.

        Args:
            flow_name: The flow identifier (e.g. ``"developer"``).
            flow_data: The full flow input as a dict.

        Returns:
            A tuple of ``(goal_string, project_id_or_path, linkable_ids)``.
            ``linkable_ids`` is a dict that may contain ``"issue_id"`` and/or
            ``"merge_request_id"`` as IIDs (integers) when the flow input
            provides enough information to resolve them.

        Pydantic validation on ``StartFlowInput`` guarantees that all
        required fields are present before this method is reached.
        """
        linkable_ids: dict[str, int] = {}

        if flow_name == "developer":
            project_url = flow_data.get("project_url")
            if project_url:
                project_path = self._parse_project_url(project_url)
                project_id: Optional[str | int] = project_path
            else:
                project_id = self.project.get("id") if self.project else None

            issue_url = flow_data.get("issue_url")
            if issue_url:
                issue_project_path, issue_iid = self._parse_issue_url(issue_url)
                linkable_ids["issue_id"] = issue_iid
                if not project_id:
                    # No project from project_url or chat context —
                    # fall back to the issue's project so Rails can
                    # resolve the IID.
                    project_id = issue_project_path
                elif project_url and project_id != issue_project_path:
                    # An explicit project_url was given but it points
                    # to a different project than the issue.  Override
                    # with the issue project so the IID resolves
                    # correctly, mirroring the fix_pipeline pattern.
                    log.warning(
                        "start_flow: project_url and issue_url belong to "
                        "different projects; using issue project for "
                        "linkable resolution",
                        project_id=project_id,
                        issue_project=issue_project_path,
                    )
                    project_id = issue_project_path

            return flow_data["goal"], project_id, linkable_ids

        if flow_name == "fix_pipeline":
            project_path, _pipeline_iid = self._parse_pipeline_url(
                str(flow_data["pipeline_url"])
            )
            mr_project_path, mr_iid = self._parse_merge_request_url(
                flow_data["merge_request_url"]
            )
            if mr_project_path != project_path:
                log.warning(
                    "start_flow: pipeline and merge request belong to "
                    "different projects; using MR project for linkable "
                    "resolution",
                    pipeline_project=project_path,
                    mr_project=mr_project_path,
                )
                project_path = mr_project_path
            linkable_ids["merge_request_id"] = mr_iid
            return str(flow_data["pipeline_url"]), project_path, linkable_ids

        if flow_name == "code_review":
            project_path, mr_iid = self._parse_merge_request_url(
                flow_data["merge_request_url"]
            )
            linkable_ids["merge_request_id"] = mr_iid
            return str(mr_iid), project_path, linkable_ids

        raise ToolException(f"Unknown flow: {flow_name!r}")

    def _parse_merge_request_url(self, url: str) -> tuple[str, int]:
        """Parse a merge request URL into (project_path, iid).

        Returns the decoded project path (e.g. ``group/project``) so it
        can be used directly with the Rails ``find_project!`` helper.
        """
        try:
            encoded_path, iid = GitLabUrlParser.parse_merge_request_url(
                url, self.gitlab_host
            )
            return unquote(encoded_path), iid
        except GitLabUrlParseError as exc:
            raise ToolException(
                f"Could not parse merge request URL '{url}': {exc}"
            ) from exc

    def _parse_issue_url(self, url: str) -> tuple[str, int]:
        """Parse an issue or work-item URL into (project_path, iid).

        Accepts both ``/-/issues/<iid>`` and ``/-/work_items/<iid>`` URL
        formats.  Returns the decoded project path so it can serve as a
        fallback ``project_id`` when no other project context is available.
        """
        try:
            encoded_path, iid = GitLabUrlParser.parse_issue_url(url, self.gitlab_host)
            return unquote(encoded_path), iid
        except GitLabUrlParseError as exc:
            raise ToolException(f"Could not parse issue URL '{url}': {exc}") from exc

    def _parse_pipeline_url(self, url: str) -> tuple[str, int]:
        """Parse a pipeline URL into (project_path, iid).

        Returns the decoded project path (e.g. ``group/project``) so it
        can be used directly with the Rails ``find_project!`` helper.
        """
        try:
            encoded_path, iid = GitLabUrlParser.parse_pipeline_url(
                url, self.gitlab_host
            )
            return unquote(encoded_path), iid
        except GitLabUrlParseError as exc:
            raise ToolException(f"Could not parse pipeline URL '{url}': {exc}") from exc

    def _parse_project_url(self, url: str) -> str:
        """Parse a project URL into a decoded project path."""
        try:
            encoded_path = GitLabUrlParser.parse_project_url(url, self.gitlab_host)
            return unquote(encoded_path)
        except GitLabUrlParseError as exc:
            raise ToolException(f"Could not parse project URL '{url}': {exc}") from exc

    def format_display_message(
        self,
        args: StartFlowInput,
        _tool_response: Any = None,
    ) -> str:
        flow_dict = args.flow.model_dump()

        flow_name = flow_dict.get("name", "unknown")

        if _tool_response:
            try:
                content = getattr(_tool_response, "content", _tool_response)
                if isinstance(content, str):
                    data = json.loads(content)
                    workflow_id = data.get("workflow_id")
                    if workflow_id:
                        session_url = data.get("session_url")
                        msg = (
                            f"Started flow **{flow_name}** (workflow ID: {workflow_id})"
                        )
                        if session_url:
                            msg += f" — [View session]({session_url})"
                        return msg
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        # Fallback: build a human-readable summary.
        if flow_name == "developer":
            detail = flow_dict.get("goal", "")
        elif flow_name == "fix_pipeline":
            detail = str(flow_dict.get("pipeline_url", ""))
        elif flow_name == "code_review":
            detail = str(flow_dict.get("merge_request_url", ""))
        else:
            detail = str(flow_dict)

        return f"Starting flow {flow_name} with goal: {detail}"
