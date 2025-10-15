import json
from typing import Callable, NoReturn

import structlog

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse

logger = structlog.stdlib.get_logger(__name__)


class UnsupportedStatusEvent(Exception):
    pass


class GitLabStatusUpdater:
    def __init__(
        self,
        client: GitlabHttpClient,
        status_update_callback: (
            Callable[[WorkflowStatusEventEnum], NoReturn] | None
        ) = None,
    ):
        self._client = client
        self.workflow_api_path = "/api/v4/ai/duo_workflows/workflows"
        self.status_update_callback = status_update_callback

    async def get_workflow_status(self, workflow_id: str) -> str:
        response = await self._client.aget(
            path=f"{self.workflow_api_path}/{workflow_id}",
            parse_json=True,
            use_http_response=True,
        )

        if not response.is_success():
            logger.error(
                "Failed to get workflow status",
                workflow_id=workflow_id,
                status_code=response.status_code,
                response_body=response.body,
            )

        return response.body.get("status")

    async def update_workflow_status(
        self, workflow_id: str, status_event: WorkflowStatusEventEnum
    ) -> None:
        """Update the status of a workflow in GitLab.

        Args:
            workflow_id (str): The ID of the workflow to update.
            status_event (WorkflowStatusEventEnum): The status event for the workflow. Can be start, finish or drop.

        Raises:
            Exception: If the update request fails.
            ToolException: If HTTP connection fails.
        """
        result = await self._client.apatch(
            path=f"{self.workflow_api_path}/{workflow_id}",
            body=json.dumps({"status_event": status_event.value}),
            parse_json=True,
            use_http_response=True,
        )

        if not isinstance(result, GitLabHttpResponse):
            raise Exception(
                f"Unexpected response from client, status update might have failed. response: {result}"
            )

        if result.status_code == 400:
            raise UnsupportedStatusEvent(
                f"Session status cannot be updated due to bad status event: {status_event}, error: {result.body}"
            )

        if result.status_code != 200:
            raise Exception(
                f"Failed to update workflow with '{status_event}' status: {result.status_code}"
            )

        if self.status_update_callback:
            self.status_update_callback(status_event)
