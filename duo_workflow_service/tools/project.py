from typing import Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class GetProjectInput(BaseModel):
    project_id: int = Field(description="Id of the project")


class GetProject(DuoBaseTool):
    name: str = "get_project"
    description: str = """Fetch details about the project"""
    args_schema: Type[BaseModel] = GetProjectInput  # type: ignore

    async def _arun(self, project_id: str) -> str:
        return await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}", parse_json=False
        )

    def format_display_message(self, args: GetProjectInput) -> str:
        return f"Get project information for project {args.project_id}"
