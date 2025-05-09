from typing import TypedDict

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
) -> Project:
    workflow = await fetch_workflow_config(client, workflow_id)

    if not (isinstance(workflow, dict) and "project_id" in workflow):
        raise Exception("Failed to retrieve project ID")

    project_id = workflow["project_id"]

    # Fetch project data using the project ID
    project = await client.aget(
        path=f"/api/v4/projects/{project_id}",
        parse_json=True,
    )

    return project
