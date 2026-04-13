import json
from typing import Any, Type

import structlog
from pydantic import BaseModel

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger(__name__)


class GetCurrentUserInput(BaseModel):
    pass


class GetCurrentUser(DuoBaseTool):
    name: str = "get_current_user"
    description: str = """
    Get the current user information from GitLab API

    Only the following information will be retrieved from the current user endpoint:
    - user id
    - user name
    - job title
    - preferred languages (written in ISO 639‑1 language code)
    """
    args_schema: Type[BaseModel] = GetCurrentUserInput

    async def _execute(self) -> str:
        response = await self.gitlab_client.aget(path="/api/v4/user", parse_json=True)

        body = self._process_http_response("Get current user", response, log)

        formatted_response = {
            "user_id": body.get("id"),
            "user_name": body.get("username"),
            "job_title": body.get("job_title"),
            "preferred_language": body.get("preferred_language"),
        }

        return json.dumps({"user": formatted_response})

    def format_display_message(
        self, args: GetCurrentUserInput, _tool_response: Any = None
    ) -> str:
        return "Get current user information"
