import json
from enum import StrEnum
from typing import Any, ClassVar, List, Optional, Type

from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

__all__ = ["TodoStatus", "TodoWrite"]


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# pylint: disable=line-too-long
DESCRIPTION = """Use this tool to create or update a task list to track progress on complex tasks.

- Set a few high-level milestones at the start (e.g. Investigation -> Implementation -> Verification).
- Mark a milestone completed as soon as it is done; this is the progress signal shown to the user in real time.
- Don't spend turns on updates that carry no information (e.g. re-listing unchanged milestones).
"""


class TodoItem(BaseModel):
    description: str = Field(description="Brief description of the task")
    status: TodoStatus = Field(
        description="Current status of the task: pending, in_progress, completed, cancelled"
    )


class TodoWriteInput(BaseModel):
    todos: List[TodoItem] = Field(description="The complete updated todo list")


class TodoWrite(DuoBaseTool):
    name: str = "todo_write"
    description: str = DESCRIPTION
    args_schema: Type[BaseModel] = TodoWriteInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    tool_version: ClassVar[Version] = Version("0.1.0")

    async def _execute(self, todos: List[TodoItem]) -> str:
        items = [{"description": t.description, "status": t.status} for t in todos]
        return json.dumps(items)

    def format_display_message(
        self, args: TodoWriteInput, _tool_response: Any = None
    ) -> Optional[str]:
        remaining = sum(
            1
            for t in args.todos
            if t.status not in (TodoStatus.COMPLETED, TodoStatus.CANCELLED)
        )
        return f"{remaining} todos remaining"
