from typing import Any, ClassVar, Optional, Type

from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

__all__ = ["Think"]


# pylint: disable=line-too-long
DESCRIPTION = """Use this tool to think through a problem, plan your next steps, or reason about information you've gathered.
This tool has no side effects — it simply records your thought. Use it to:

- Analyze what insights you gathered from your previous actions
- Decide on an approach before taking action
- Reflect on whether your current approach is making progress toward the goal
- Break down a complex problem before diving into code
"""


class ThinkInput(BaseModel):
    thought: str = Field(description="Your reasoning, analysis, or plan")


class Think(DuoBaseTool):
    name: str = "think"
    description: str = DESCRIPTION
    args_schema: Type[BaseModel] = ThinkInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    tool_version: ClassVar[Version] = Version("0.1.0")

    async def _execute(self, thought: str) -> str:  # pylint: disable=unused-argument
        return "ok"

    def format_display_message(
        self, args: ThinkInput, _tool_response: Any = None
    ) -> Optional[str]:
        preview = args.thought[:80]
        if len(args.thought) > 80:
            preview += "..."
        return preview
