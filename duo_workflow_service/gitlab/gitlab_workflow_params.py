from typing import Any

import structlog
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitlabHttpClient

logger = structlog.stdlib.get_logger(__name__)


class WorkflowConfigFetchError(ToolException):
    """Raised when fetching a workflow's config via the REST API fails."""


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
        raise WorkflowConfigFetchError(
            f"Failed to fetch workflow config for workflow {workflow_id}: "
            f"HTTP {response.status_code}"
        )

    if not isinstance(response.body, dict):
        logger.error(
            "Unexpected workflow config response shape",
            workflow_id=workflow_id,
            response_type=type(response.body).__name__,
        )
        raise WorkflowConfigFetchError(
            f"Unexpected workflow config response for workflow {workflow_id}: "
            f"expected object, got {type(response.body).__name__}"
        )

    return response.body
