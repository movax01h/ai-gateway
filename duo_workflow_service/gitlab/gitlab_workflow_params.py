from typing import Any

import structlog

from duo_workflow_service.gitlab.http_client import GitlabHttpClient

logger = structlog.stdlib.get_logger(__name__)


async def fetch_workflow_config(
    client: GitlabHttpClient, workflow_id: str
) -> dict[str, Any]:
    response = await client.aget(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}",
        parse_json=True,
    )

    if not response.is_success():
        logger.error(
            "Failed to fetch workflow config",
            workflow_id=workflow_id,
            status_code=response.status_code,
            response_body=response.body,
        )

    return response.body
