from typing import Any, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger(__name__)


class GetProjectInput(BaseModel):
    project_id: int = Field(description="Id of the project")


class GetProject(DuoBaseTool):
    name: str = "get_project"
    description: str = """Fetch details about the project"""
    args_schema: Type[BaseModel] = GetProjectInput  # type: ignore

    async def _arun(self, project_id: str) -> str:
        response = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}",
            parse_json=False,
            use_http_response=True,
        )

        if not response.is_success():
            log.error(
                "Get project request failed with status %s: %s",
                response.status_code,
                response.body,
            )

        return response.body

    def format_display_message(
        self, args: GetProjectInput, _tool_response: Any = None
    ) -> str:
        return f"Get project information for project {args.project_id}"
