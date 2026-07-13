from typing import NotRequired, Optional, Tuple, TypedDict

from duo_workflow_service.errors.typing import InvalidWorkflowIdException
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.queries import fetch_query_for_version
from duo_workflow_service.gitlab.schema import PromptInjectionProtectionLevel
from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from lib.context import gitlab_version


def workflow_global_id(workflow_id: str) -> str:
    """Build a GitLab global ID for a Duo Workflows workflow."""
    return f"gid://gitlab/Ai::DuoWorkflows::Workflow/{workflow_id}"


class Language(TypedDict):
    name: str
    share: float


class Project(TypedDict):
    id: int
    description: str
    name: str
    http_url_to_repo: str
    web_url: str
    default_branch: Optional[str]
    languages: Optional[list[Language]]
    exclusion_rules: Optional[list[str]]


class Namespace(TypedDict):
    id: int
    description: str
    name: str
    web_url: str


class Checkpoint(TypedDict, total=False):
    checkpoint: str
    compressedCheckpoint: str
    threadTs: str
    parentTs: str
    metadata: str


class FoundationalFlowsFeature(TypedDict):
    enabled: bool
    enabled_flows: Optional[list[str]]


class WorkflowFeatures(TypedDict, total=False):
    foundational_flows: FoundationalFlowsFeature


class WorkflowConfig(TypedDict):
    workflow_id: str
    agent_privileges_names: list
    pre_approved_agent_privileges_names: list
    workflow_status: str
    mcp_enabled: bool
    incremental_checkpoints_enabled: bool
    allow_agent_to_request_user: bool
    gitlab_host: str
    first_checkpoint: Optional[Checkpoint]
    latest_checkpoint: Optional[Checkpoint]
    prompt_injection_protection_level: PromptInjectionProtectionLevel
    archived: bool
    stalled: bool
    features: NotRequired[WorkflowFeatures]


async def fetch_workflow_and_container_data(
    client: GitlabHttpClient, workflow_id: str
) -> Tuple[Project | None, Namespace | None, WorkflowConfig]:
    query = fetch_query_for_version(gitlab_version.get())

    variables = {"workflowId": workflow_global_id(workflow_id)}

    try:
        response = await client.graphql(query, variables)
    except Exception as e:
        # Check if the error message indicates workflow not found
        if "Workflow not found" in str(e):
            raise InvalidWorkflowIdException(str(e))
        raise

    workflows = response.get("duoWorkflowWorkflows", {}).get("nodes", [])

    if not workflows:
        raise InvalidWorkflowIdException(
            f"No workflow found for workflow ID: {workflow_id}"
        )

    # Get the first workflow (assuming there's at least one)
    workflow = workflows[0]

    # Extract project data
    project_data = workflow.get("project") or {}
    namespace_data = workflow.get("namespace") or {}

    project = (
        Project(
            id=extract_id_from_global_id(workflow.get("projectId", "0")),
            name=project_data.get("name", ""),
            http_url_to_repo=project_data.get("httpUrlToRepo", ""),
            web_url=project_data.get("webUrl", ""),
            description=project_data.get("description", ""),
            languages=project_data.get("languages", []),
            default_branch=extract_default_branch_from_project_repository(workflow),
            exclusion_rules=project_data.get("duoContextExclusionSettings", {}).get(
                "exclusionRules", []
            ),
        )
        if project_data
        else None
    )

    namespace = (
        Namespace(
            id=extract_id_from_global_id(workflow.get("namespaceId", "0")),
            name=namespace_data.get("name", ""),
            web_url=namespace_data.get("webUrl", ""),
            description=namespace_data.get("description", ""),
        )
        if namespace_data
        else None
    )

    # Convert GraphQL response to expected Container format
    # Extract prompt injection protection level from the workflow's namespace or the
    # project's parent namespace.
    # Default to LOG_ONLY for GitLab < 18.8 where aiSettings is not available.
    # LOG_ONLY mode scans in background without interrupting workflow or showing UI messages.
    web_url = project_data.get("webUrl", "") or namespace_data.get("webUrl", "")

    prompt_injection_protection_level = PromptInjectionProtectionLevel.LOG_ONLY

    ai_settings = (
        (project_data.get("namespace") or {}).get("aiSettings", {})
        or namespace_data.get("aiSettings", {})
        or {}  # set ai_settings default to empty dict in case project and namespace data are both None
    )

    prompt_injection_protection_level = PromptInjectionProtectionLevel.from_graphql(
        ai_settings.get("promptInjectionProtectionLevel")
    )

    gitlab_host = GitLabUrlParser.extract_host_from_url(web_url)

    if not gitlab_host:
        raise RuntimeError(
            f"Failed to extract gitlab host from web_url for workflow {workflow_id}"
        )

    status_check = project_data.get("duoWorkflowStatusCheck") or {}

    workflow_config = WorkflowConfig(
        workflow_id=workflow_id,
        agent_privileges_names=workflow.get("agentPrivilegesNames", []),
        pre_approved_agent_privileges_names=workflow.get(
            "preApprovedAgentPrivilegesNames", []
        ),
        workflow_status=workflow.get("statusName", ""),
        mcp_enabled=workflow.get("mcpEnabled", False),
        incremental_checkpoints_enabled=workflow.get(
            "incrementalCheckpointsEnabled", False
        ),
        allow_agent_to_request_user=workflow.get("allowAgentToRequestUser", False),
        first_checkpoint=workflow.get("firstCheckpoint", None),
        latest_checkpoint=workflow.get("latestCheckpoint", None),
        gitlab_host=gitlab_host,
        prompt_injection_protection_level=prompt_injection_protection_level,
        archived=workflow.get("archived", None),
        stalled=workflow.get("stalled", None),
        features={
            "foundational_flows": {
                "enabled": status_check.get("foundationalFlowsEnabled", True),
                "enabled_flows": status_check.get("enabledFoundationalFlows"),
            }
        },
    )

    return project, namespace, workflow_config


def extract_default_branch_from_project_repository(workflow: dict) -> Optional[str]:
    repository_str = (
        workflow.get("project", {}).get("statisticsDetailsPaths") or {}
    ).get("repository", "")

    if repository_str and isinstance(repository_str, str):
        return repository_str.split("/-/tree/", 1)[1]

    return None


def extract_id_from_global_id(global_id: str):
    extracted_id = 0
    if global_id and isinstance(global_id, str) and "gid://" in global_id:
        extracted_id = int(global_id.rsplit("/", maxsplit=1)[-1])
    elif global_id and isinstance(global_id, str) and global_id.startswith("#"):
        extracted_id = int(global_id.rsplit("#", maxsplit=1)[-1])
    else:
        extracted_id = int(global_id) if global_id else 0

    return extracted_id


def empty_workflow_config() -> WorkflowConfig:
    return {
        "workflow_id": "",
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "allow_agent_to_request_user": False,
        "mcp_enabled": False,
        "incremental_checkpoints_enabled": False,
        "first_checkpoint": None,
        "latest_checkpoint": None,
        "workflow_status": "",
        "gitlab_host": "",
        "prompt_injection_protection_level": PromptInjectionProtectionLevel.LOG_ONLY,
        "archived": False,
        "stalled": False,
    }
