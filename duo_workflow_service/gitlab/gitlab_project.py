from typing import Any, Dict, Tuple, TypedDict

from duo_workflow_service.gitlab.gitlab_workflow_params import fetch_workflow_config
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


class Project(TypedDict):
    id: int
    description: str
    name: str
    http_url_to_repo: str
    web_url: str


async def fetch_project_data_with_workflow_id(
    client: GitlabHttpClient, workflow_id: str
) -> Tuple[Project, Dict[str, Any]]:
    workflow_config = await fetch_workflow_config(client, workflow_id)

    if not (isinstance(workflow_config, dict) and "project_id" in workflow_config):
        raise Exception("Failed to retrieve project ID from workflow config")

    project_id = workflow_config["project_id"]

    # Fetch project data using the project ID
    project = await client.aget(
        path=f"/api/v4/projects/{project_id}",
        parse_json=True,
    )

    return project, workflow_config
