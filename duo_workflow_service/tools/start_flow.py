import json
from typing import Annotated, Any, ClassVar, Literal, Optional, Type, Union
from urllib.parse import unquote

import structlog
from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field, create_model, model_validator

from duo_workflow_service.gitlab.gitlab_api import WorkflowFeatures
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger(__name__)


FLOW_IDENTIFIER_MAP = {
    "developer": "developer/v1",
    "fix_pipeline": "fix_pipeline/v1",
    "code_review": "code_review/v1",
    "sast_fp_detection": "sast_fp_detection/v1",
    "resolve_sast_vulnerability": "resolve_sast_vulnerability/v1",
    "secrets_fp_detection": "secrets_fp_detection/v1",
}

# Custom AI Catalog flows are addressed by item consumer ID rather than by a
# fixed identifier, so this name is deliberately absent from
# FLOW_IDENTIFIER_MAP and handled on its own branch throughout.
CATALOG_FLOW_NAME = "catalog_flow"

_DESCRIPTION_PREFIX = (
    "Delegate a task to a specialist GitLab agent that works "
    "asynchronously over multiple steps. Always use this tool when the "
    "user's request matches one of the agents below — specialist "
    "agents are purpose-built for their domain and deliver better "
    "outcomes than inline handling.\n"
    "\n"
    "Available agents:\n"
)

_FLOW_AGENT_DESCRIPTIONS = {
    "developer": (
        "- developer: General-purpose agent for tasks that involve writing or "
        "changing code, resolving an issue, or implementing changes described "
        "in chat. The `goal` is the agent's only briefing — it does not see this "
        "conversation, so include the user's intent, any relevant GitLab URLs, "
        "and context from chat the agent would otherwise miss.\n"
    ),
    "fix_pipeline": "- fix_pipeline: fixing, debugging, or investigating a failing pipeline or broken "
    "build (e.g. 'fix this pipeline', 'the build is broken').\n",
    "code_review": "- code_review: read-only review and assessment of the changes in a merge request — "
    "not for implementing feedback or making changes "
    '(e.g. "review this MR", "can you review my merge request?").\n',
    "sast_fp_detection": "- sast_fp_detection: analyse a SAST vulnerability for false positives. Requires a "
    "vulnerability_id.\n",
    "resolve_sast_vulnerability": "- resolve_sast_vulnerability: resolve a SAST vulnerability by generating a code fix. Requires a "
    "vulnerability_id.\n",
    "secrets_fp_detection": "- secrets_fp_detection: analyse a secret detection vulnerability for false positives. "
    "Requires a vulnerability_id.\n",
}

_CATALOG_FLOW_DESCRIPTION = (
    "- catalog_flow: run a custom AI Catalog flow that a project has enabled. "
    "Requires an ai_catalog_item_consumer_id, which cannot be looked up from "
    "here — only use this agent when the user supplies that ID.\n"
)

_DESCRIPTION_SUFFIX = (
    "\n\nReturns a session URL the user can follow to track progress. The user "
    "is prompted to approve the handoff before the agent starts."
)


def enabled_flow_identifiers(
    features: Optional[WorkflowFeatures],
) -> Optional[list[str]]:
    """Enabled foundational flow identifiers for a workflow."""
    foundational_flows = (features or {}).get("foundational_flows")
    if foundational_flows is None:
        return None
    if not foundational_flows.get("enabled", True):
        return []
    return foundational_flows.get("enabled_flows")


def enabled_flow_names(flow_identifiers: Optional[list[str]]) -> list[str]:
    """Supported flow names enabled for the project."""
    if flow_identifiers is None:
        return list(FLOW_IDENTIFIER_MAP)
    return [
        name
        for name, identifier in FLOW_IDENTIFIER_MAP.items()
        if identifier in flow_identifiers
    ]


def _build_description(enabled_names: list[str]) -> str:
    agents = "\n".join(
        _FLOW_AGENT_DESCRIPTIONS[name]
        for name in FLOW_IDENTIFIER_MAP
        if name in enabled_names
    )
    return (
        _DESCRIPTION_PREFIX + agents + _CATALOG_FLOW_DESCRIPTION + _DESCRIPTION_SUFFIX
    )


_GENERIC_FAILURE_DETAIL = "An internal error occurred while starting the flow."

_FORBIDDEN_FAILURE_DETAIL = (
    "This flow isn't available, or you don't have sufficient permissions to start it."
)

# Catalog flows are addressed by an ID the user supplies by hand, so a wrong
# one is the likeliest failure and worth naming. Rails distinguishes the cases:
# an unknown consumer ID is a 404, one belonging to another project or group is
# a 403, and its service layer folds everything else — including permission
# failures — into a 400.
_CATALOG_FAILURE_DETAILS = {
    400: (
        "This flow can't be started. Check that the ID is correct and that the "
        "flow is still enabled in this project."
    ),
    403: (
        "This flow isn't enabled in this project, or you don't have sufficient "
        "permissions to start it."
    ),
    404: "No flow with that ID is enabled in this project.",
}


def _failure_detail(status_code: int, flow_name: str) -> str:
    """User-facing explanation for a failed start request.

    Args:
        status_code: HTTP status Rails returned.
        flow_name: Flow the request was for, used to pick the catalog-specific
            wording.

    Returns:
        A reason suitable for both the LLM-facing message and the UI chat log.
    """
    if flow_name == CATALOG_FLOW_NAME and status_code in _CATALOG_FAILURE_DETAILS:
        return _CATALOG_FAILURE_DETAILS[status_code]
    if status_code == 403:
        return _FORBIDDEN_FAILURE_DETAIL
    return _GENERIC_FAILURE_DETAIL


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


class StartSastFpDetectionFlowInput(BaseModel):
    """Input for the SAST False Positive Detection flow."""

    name: Literal["sast_fp_detection"]
    vulnerability_id: str = Field(
        description="The ID of the SAST vulnerability to analyse for false positives.",
    )


class StartResolveSastVulnerabilityFlowInput(BaseModel):
    """Input for the SAST Vulnerability Resolution flow."""

    name: Literal["resolve_sast_vulnerability"]
    vulnerability_id: str = Field(
        description="The ID of the SAST vulnerability to resolve.",
    )


class StartSecretsFpDetectionFlowInput(BaseModel):
    """Input for the Secret Detection False Positive Detection flow."""

    name: Literal["secrets_fp_detection"]
    vulnerability_id: str = Field(
        description=(
            "The ID of the secret detection vulnerability to analyse for false positives."
        ),
    )


class StartCatalogFlowInput(BaseModel):
    """Input for a custom AI Catalog flow."""

    name: Literal["catalog_flow"]
    ai_catalog_item_consumer_id: int = Field(
        description=(
            "ID of the AI Catalog item consumer identifying which custom flow "
            "to run. This is the record created when a flow is enabled in a "
            "project."
        ),
    )
    goal: Optional[str] = Field(
        default=None,
        description=(
            "Task description for the flow. Omit to let the flow fall back to "
            "its own description, which is the common case for flows that need "
            "no per-run instruction."
        ),
    )


class StartFlowInput(BaseModel):
    """Input schema for the start_flow tool."""

    flow: Annotated[
        Union[
            StartDeveloperFlowInput,
            StartFixPipelineFlowInput,
            StartCodeReviewFlowInput,
            StartSastFpDetectionFlowInput,
            StartResolveSastVulnerabilityFlowInput,
            StartSecretsFpDetectionFlowInput,
            StartCatalogFlowInput,
        ],
        Field(discriminator="name"),
    ]


_FLOW_INPUT_SCHEMAS: dict[str, Type[BaseModel]] = {
    "developer": StartDeveloperFlowInput,
    "fix_pipeline": StartFixPipelineFlowInput,
    "code_review": StartCodeReviewFlowInput,
    "sast_fp_detection": StartSastFpDetectionFlowInput,
    "resolve_sast_vulnerability": StartResolveSastVulnerabilityFlowInput,
    "secrets_fp_detection": StartSecretsFpDetectionFlowInput,
}


def _build_args_schema(enabled_names: list[str]) -> Type[BaseModel]:
    """Input schema exposing only the enabled flows."""
    enabled_flow_inputs = [
        _FLOW_INPUT_SCHEMAS[name]
        for name in FLOW_IDENTIFIER_MAP
        if name in enabled_names
    ]
    # Full schema when every foundational flow is enabled.
    if len(enabled_flow_inputs) == len(_FLOW_INPUT_SCHEMAS):
        return StartFlowInput
    # Catalog flows are enabled per project rather than by the foundational-flow
    # settings, so the member stays available however those are narrowed.
    enabled_flow_inputs.append(StartCatalogFlowInput)
    if len(enabled_flow_inputs) == 1:
        flow_annotation: Any = enabled_flow_inputs[0]
    else:
        flow_annotation = Annotated[
            Union[tuple(enabled_flow_inputs)], Field(discriminator="name")
        ]
    return create_model("StartFlowInput", flow=(flow_annotation, ...))


class StartFlow(DuoBaseTool):
    name: str = "start_flow"
    tool_version: ClassVar[Version] = Version("0.0.1")
    # Lists all agents by default; narrowed to enabled flows in the validator below.
    description: str = _build_description(list(FLOW_IDENTIFIER_MAP))
    args_schema: Type[BaseModel] = StartFlowInput

    def _enabled_flow_identifiers(self) -> Optional[list[str]]:
        return enabled_flow_identifiers((self.metadata or {}).get("features"))

    @model_validator(mode="after")
    def _configure_enabled_flow_schema(self) -> "StartFlow":
        flow_identifiers = self._enabled_flow_identifiers()
        enabled_names = enabled_flow_names(flow_identifiers)
        self.description = _build_description(enabled_names)
        self.args_schema = _build_args_schema(enabled_names)
        return self

    async def _execute(
        self,
        flow: (
            StartDeveloperFlowInput
            | StartFixPipelineFlowInput
            | StartCodeReviewFlowInput
            | StartSastFpDetectionFlowInput
            | StartResolveSastVulnerabilityFlowInput
            | StartSecretsFpDetectionFlowInput
            | StartCatalogFlowInput
        ),
        **_kwargs: Any,
    ) -> str:
        if isinstance(flow, BaseModel):
            flow_data = flow.model_dump()
        else:
            raise ToolException(f"Unexpected flow input type: {type(flow)}")

        flow_name = flow_data["name"]

        if flow_name == CATALOG_FLOW_NAME:
            return await self._start_catalog_flow(flow_data)

        backend_flow_id = FLOW_IDENTIFIER_MAP.get(flow_name)
        if not backend_flow_id:
            raise ToolException(f"Unknown flow: {flow_name!r}")

        flow_identifiers = self._enabled_flow_identifiers()
        if flow_identifiers is not None and backend_flow_id not in flow_identifiers:
            return json.dumps(
                {
                    "status": "unavailable",
                    "flow_name": flow_name,
                    "message": f"The {flow_name} flow is not enabled for this project.",
                }
            )

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

        return await self._post_flow(payload, flow_name)

    async def _start_catalog_flow(self, flow_data: dict) -> str:
        """Start a custom AI Catalog flow by item consumer ID.

        Rails treats ``ai_catalog_item_consumer_id`` and ``workflow_definition``
        as mutually exclusive branches, routing the former to
        ``Ai::Catalog::Flows::ExecuteService``. That service falls back to the
        flow's own description when no goal is given, so an absent goal is
        omitted rather than sent as null.

        Args:
            flow_data: The validated ``StartCatalogFlowInput`` as a dict.

        Returns:
            The same JSON payload shape as the foundational flows.
        """
        project_id = self.project.get("id") if self.project else None
        if not project_id:
            raise StartFlowError(
                "Custom catalog flows need a project, and this session has none.",
                response="Custom flows can only be started from a project.",
            )

        payload: dict[str, Any] = {
            "ai_catalog_item_consumer_id": flow_data["ai_catalog_item_consumer_id"],
            "project_id": project_id,
            "environment": "ambient",
            "start_workflow": True,
        }

        goal = flow_data.get("goal")
        if goal:
            payload["goal"] = goal

        return await self._post_flow(payload, CATALOG_FLOW_NAME)

    async def _post_flow(self, payload: dict[str, Any], flow_name: str) -> str:
        """POST a start request to Rails and format the tool response.

        Args:
            payload: The request body, already shaped for the target branch.
            flow_name: Name reported back in the response, used by the chat UI
                to label the session card.

        Returns:
            A JSON string carrying ``status``, ``workflow_id``, ``session_url``
            and ``flow_name``.
        """
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
                # flow_name is always "catalog_flow" for a catalog flow, so
                # without this there is no way to tell which one failed.
                ai_catalog_item_consumer_id=payload.get("ai_catalog_item_consumer_id"),
            )
            detail = _failure_detail(response.status_code, flow_name)
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
        For security flows (sast_fp_detection, resolve_sast_vulnerability,
        secrets_fp_detection), the project falls back to ``self.project``
        when available, since the vulnerability ID already encodes the
        resource identity.

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

        if flow_name in (
            "sast_fp_detection",
            "resolve_sast_vulnerability",
            "secrets_fp_detection",
        ):
            project_id = self.project.get("id") if self.project else None
            return flow_data["vulnerability_id"], project_id, linkable_ids

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
        elif flow_name in (
            "sast_fp_detection",
            "resolve_sast_vulnerability",
            "secrets_fp_detection",
        ):
            detail = str(flow_dict.get("vulnerability_id", ""))
        elif flow_name == CATALOG_FLOW_NAME:
            detail = flow_dict.get("goal") or str(
                flow_dict.get("ai_catalog_item_consumer_id", "")
            )
        else:
            detail = str(flow_dict)

        return f"Starting flow {flow_name} with goal: {detail}"
