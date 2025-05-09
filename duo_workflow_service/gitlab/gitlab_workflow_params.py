from typing import Any

from duo_workflow_service.gitlab.http_client import GitlabHttpClient


async def fetch_workflow_config(
    client: GitlabHttpClient, workflow_id: str
) -> dict[str, Any]:
    workflow = await client.aget(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        parse_json=True,
    )

    return workflow
