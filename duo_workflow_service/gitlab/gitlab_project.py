from typing import Optional, Tuple, TypedDict

from duo_workflow_service.gitlab.http_client import GitlabHttpClient


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


class Checkpoint(TypedDict):
    checkpoint: str


class WorkflowConfig(TypedDict):
    agent_privileges_names: list
    pre_approved_agent_privileges_names: list
    workflow_status: str
    mcp_enabled: bool
    allow_agent_to_request_user: bool
    first_checkpoint: Optional[Checkpoint]


async def fetch_workflow_and_project_data(
    client: GitlabHttpClient, workflow_id: str
) -> Tuple[Project, WorkflowConfig]:
    query = """
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

    variables = {"workflowId": f"gid://gitlab/Ai::DuoWorkflows::Workflow/{workflow_id}"}

    response = await client.graphql(query, variables)

    workflows = response.get("duoWorkflowWorkflows", {}).get("nodes", [])

    if not workflows:
        raise Exception(f"No workflow found for workflow ID: {workflow_id}")

    # Get the first workflow (assuming there's at least one)
    workflow = workflows[0]

    # Extract project data
    project_data = workflow.get("project", {})

    # Convert GraphQL response to expected Project format
    project = Project(
        id=extract_project_id_from_workflow(workflow),
        name=project_data.get("name", ""),
        http_url_to_repo=project_data.get("httpUrlToRepo", ""),
        web_url=project_data.get("webUrl", ""),
        description=project_data.get("description", ""),
        languages=project_data.get("languages", []),
        default_branch=extract_default_branch_from_project_repository(workflow),
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
    )

    return project, workflow_config


def extract_project_id_from_workflow(workflow: dict):
    project_id_str = workflow.get("projectId", "0")
    project_id = 0
    if (
        project_id_str
        and isinstance(project_id_str, str)
        and "gid://" in project_id_str
    ):
        project_id = int(project_id_str.split("/")[-1])
    else:
        project_id = int(project_id_str) if project_id_str else 0

    return project_id


def extract_default_branch_from_project_repository(workflow: dict) -> Optional[str]:
    repository_str = (
        workflow.get("project", {}).get("statisticsDetailsPaths") or {}
    ).get("repository", "")

    default_branch = None
    if repository_str and isinstance(repository_str, str):
        default_branch = str(repository_str.split("/")[-1])

    return default_branch


def empty_workflow_config() -> WorkflowConfig:
    return {
        "agent_privileges_names": [],
        "pre_approved_agent_privileges_names": [],
        "allow_agent_to_request_user": False,
        "mcp_enabled": False,
        "first_checkpoint": None,
        "workflow_status": "",
    }
