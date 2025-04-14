from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["HandoverTool"]

# This is a no-op tool that is used to enforce model response following defined structure
# this pattern is exampled by LangGraph framework at
# https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/respond-in-format.ipynb
# Calls to this tool are capture by graph routing and used to transfer to HandoverAgent node.
TOOL_TITLE = "handover_tool"


class HandoverTool(BaseModel):
    description: str = """A final response to the user"""
    summary: str = Field(
        description="The summary of the work done based on the past conversation between human, agent and tools executions"
    )

    tool_title: ClassVar[str] = TOOL_TITLE

    model_config = ConfigDict(title=TOOL_TITLE, frozen=True)
