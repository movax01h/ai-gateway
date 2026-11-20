import json
from typing import Any, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class UpdateFormPermissionsInput(BaseModel):
    select: list[str] = Field(
        default_factory=list,
        description="Permission names to select from the access token form",
    )
    clear: list[str] = Field(
        default_factory=list,
        description="Permission names to clear from the access token form",
    )


class UpdateFormPermissions(DuoBaseTool):
    name: str = "update_form_permissions"
    description: str = (
        "Update the access token form with the suggested permissions. "
        "Use this tool to select or clear fine-grained permissions on the form."
    )
    args_schema: Type[BaseModel] = UpdateFormPermissionsInput

    async def _execute(
        self,
        select: list[str] | None = None,
        clear: list[str] | None = None,
    ) -> str:
        return json.dumps({"select": select or [], "clear": clear or []})

    def format_display_message(
        self, args: UpdateFormPermissionsInput, _tool_response: Any = None
    ) -> str:
        parts = []
        if args.select:
            parts.append(f"Select: {', '.join(args.select)}")
        if args.clear:
            parts.append(f"Clear: {', '.join(args.clear)}")
        if not parts:
            return "No permission changes"
        return "Update access token permissions — " + "; ".join(parts)
