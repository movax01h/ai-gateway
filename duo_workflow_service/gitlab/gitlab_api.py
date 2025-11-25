from typing import Optional, Tuple, TypedDict

from packaging.version import InvalidVersion, Version

from ai_gateway.instrumentators.model_requests import gitlab_version
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.tracking.errors import log_exception


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


class Checkpoint(TypedDict):
    checkpoint: str


class WorkflowConfig(TypedDict):
    agent_privileges_names: list
    pre_approved_agent_privileges_names: list
    workflow_status: str
    mcp_enabled: bool
    allow_agent_to_request_user: bool
    gitlab_host: str
    first_checkpoint: Optional[Checkpoint]


GITLAB_18_2_QUERY = """
query($workflowId: AiDuoWorkflowsWorkflowID!) {
    duoWorkflowWorkflows(workflowId: $workflowId) {
        nodes {
            statusName
            projectId
            project {
                id
                name
                description
                httpUrlToRepo
                languages {
                    name
                    share
                }
                webUrl
                statisticsDetailsPaths {
                    repository
                }
            }
            agentPrivilegesNames
            preApprovedAgentPrivilegesNames
            mcpEnabled
            allowAgentToRequestUser
            firstCheckpoint {
                checkpoint
            }
        }
    }
}
"""

# This query requires https://gitlab.com/gitlab-org/gitlab/-/merge_requests/196781 that is available in GitLab 18.3+.
GITLAB_18_3_OR_ABOVE_QUERY = """
query($workflowId: AiDuoWorkflowsWorkflowID!) {
    duoWorkflowWorkflows(workflowId: $workflowId) {
        nodes {
            statusName
            projectId
            project {
                id
                name
                description
                httpUrlToRepo
                languages {
                    name
                    share
                }
                webUrl
                statisticsDetailsPaths {
                    repository
                }
                duoContextExclusionSettings {
                    exclusionRules
                }
            }
            namespaceId
            namespace {
                id
                name
                description
                webUrl
            }
            agentPrivilegesNames
            preApprovedAgentPrivilegesNames
            mcpEnabled
            allowAgentToRequestUser
            firstCheckpoint {
                checkpoint
            }
        }
    }
}
"""

version_18_2 = Version("18.2.0")
version_18_3 = Version("18.3.0")
FALLBACK_VERSION = version_18_2


def fetch_workflow_and_container_query():
    try:
        gl_version = Version(gitlab_version.get())  # type: ignore[arg-type]
    except (InvalidVersion, TypeError) as ex:
        log_exception(ex)
        gl_version = FALLBACK_VERSION

    if version_18_3 <= gl_version:
        return GITLAB_18_3_OR_ABOVE_QUERY

    return GITLAB_18_2_QUERY


async def fetch_workflow_and_container_data(
    client: GitlabHttpClient, workflow_id: str
) -> Tuple[Project | None, Namespace | None, WorkflowConfig]:
    query = fetch_workflow_and_container_query()

    variables = {"workflowId": f"gid://gitlab/Ai::DuoWorkflows::Workflow/{workflow_id}"}

    response = await client.graphql(query, variables)

    workflows = response.get("duoWorkflowWorkflows", {}).get("nodes", [])

    if not workflows:
        raise Exception(f"No workflow found for workflow ID: {workflow_id}")

    # Get the first workflow (assuming there's at least one)
    workflow = workflows[0]

    # Extract project data
    project_data = workflow.get("project")
    namespace_data = workflow.get("namespace")

    # Convert GraphQL response to expected Container format
    web_url = ""
    if project_data:
        project = Project(
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

        web_url = project_data.get("webUrl", "")
    else:
        project = None

    if namespace_data:
        namespace = Namespace(
            id=extract_id_from_global_id(workflow.get("namespaceId", "0")),
            name=namespace_data.get("name", ""),
            web_url=namespace_data.get("webUrl", ""),
            description=namespace_data.get("description", ""),
        )

        web_url = namespace_data.get("webUrl", "")
    else:
        namespace = None

    gitlab_host = GitLabUrlParser.extract_host_from_url(web_url)

    if not gitlab_host:
        raise RuntimeError(
            f"Failed to extract gitlab host from web_url for workflow {workflow_id}"
        )

    # Build workflow config from the response
    workflow_config = WorkflowConfig(
        agent_privileges_names=workflow.get("agentPrivilegesNames", []),
        pre_approved_agent_privileges_names=workflow.get(
            "preApprovedAgentPrivilegesNames", []
        ),
        workflow_status=workflow.get("statusName", ""),
        mcp_enabled=workflow.get("mcpEnabled", False),
        allow_agent_to_request_user=workflow.get("allowAgentToRequestUser", False),
        first_checkpoint=workflow.get("firstCheckpoint", None),
        gitlab_host=gitlab_host,
    )

    return project, namespace, workflow_config


def extract_default_branch_from_project_repository(workflow: dict) -> Optional[str]:
    repository_str = (
        workflow.get("project", {}).get("statisticsDetailsPaths") or {}
    ).get("repository", "")

    default_branch = None
    if repository_str and isinstance(repository_str, str):
        default_branch = str(repository_str.split("/")[-1])

    return default_branch


def extract_id_from_global_id(global_id: str):
    extracted_id = 0
    if global_id and isinstance(global_id, str) and "gid://" in global_id:
        extracted_id = int(global_id.split("/")[-1])
    else:
        extracted_id = int(global_id) if global_id else 0

    return extracted_id


def empty_workflow_config() -> WorkflowConfig:
    return {
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "allow_agent_to_request_user": False,
        "mcp_enabled": False,
        "first_checkpoint": None,
        "workflow_status": "",
        "gitlab_host": "",
    }
