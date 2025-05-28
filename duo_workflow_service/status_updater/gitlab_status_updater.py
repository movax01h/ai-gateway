import json

from duo_workflow_service.gitlab.http_client import GitlabHttpClient


class GitLabStatusUpdater:
    def __init__(self, client: GitlabHttpClient):
        self._client = client
        self.workflow_api_path = "/api/v4/ai/duo_workflows/workflows"

    async def get_workflow_status(self, workflow_id: str) -> str:
        result = await self._client.aget(
            path=f"{self.workflow_api_path}/{workflow_id}",
            parse_json=True,
        )

        return result.get("status")

    async def update_workflow_status(self, workflow_id: str, status_event: str) -> None:
        """Update the status of a workflow in GitLab.

        Args:
            workflow_id (str): The ID of the workflow to update.
            status_event (str): The status event for the workflow. Can be start, finish or drop.

        Raises:
            Exception: If the update request fails.
        """
        result = await self._client.apatch(
            path=f"{self.workflow_api_path}/{workflow_id}",
            body=json.dumps({"status_event": status_event}),
            parse_json=True,
        )

        if isinstance(result, dict) and "status" in result and result["status"] != 200:
            raise Exception(
                f"Failed to update workflow with '{status_event}' status: {result}"
            )
