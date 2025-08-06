import json
from typing import Any, Type

from pydantic import BaseModel

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class GetCurrentUserInput(BaseModel):
    pass


class GetCurrentUser(DuoBaseTool):
    name: str = "get_current_user"
    description: str = """
    Get the current user information from GitLab API

    Only the following information will be retrieved from the current user endpoint:
    - user name
    - job title
    - preferred languages (written in ISO 639‑1 language code)
    """
    args_schema: Type[BaseModel] = GetCurrentUserInput

    async def _arun(self) -> str:
        try:
            response = await self.gitlab_client.aget(
                path="/api/v4/user", parse_json=True
            )

            formatted_response = {
                "user_name": response.get("username"),
                "job_title": response.get("job_title"),
                "preferred_language": response.get("preferred_language"),
            }

            return json.dumps({"user": formatted_response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: GetCurrentUserInput, _tool_response: Any = None
    ) -> str:
        return "Get current user information"
