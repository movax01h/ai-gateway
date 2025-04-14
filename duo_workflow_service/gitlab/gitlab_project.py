from typing import TypedDict

from duo_workflow_service.gitlab.http_client import GitlabHttpClient


class Project(TypedDict):
    id: int
    description: str
    name: str
    http_url_to_repo: str
    web_url: str
    namespace: dict


async def fetch_project_data_with_workflow_id(
    client: GitlabHttpClient, workflow_id: str
) -> Project:
    # Fetch project ID using the workflow API
    result = await client.aget(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        parse_json=True,
    )

    if not (isinstance(result, dict) and "project_id" in result):
        raise Exception("Failed to retrieve project ID")

    project_id = result["project_id"]

    # Fetch project data using the project ID
    project = await client.aget(
        path=f"/api/v4/projects/{project_id}",
        parse_json=True,
    )

    return project
