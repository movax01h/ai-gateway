import json
from typing import Literal, Optional, Self, TypeVar

import fastapi
from fastapi import status
from pydantic import BaseModel, model_validator

from ai_gateway.chat.context.current_page import CurrentPageContext
from ai_gateway.models.base_chat import Role

__all__ = [
    "AgentToolAction",
    "AgentFinalAnswer",
    "AgentUnknownAction",
    "AgentError",
    "AgentBaseEvent",
    "TypeAgentEvent",
    "AgentStep",
    "TypeAgentInputs",
    "CurrentFile",
    "AdditionalContext",
    "Message",
]


class AgentBaseEvent(BaseModel):
    def dump_as_response(self) -> str:
        model_dump = self.model_dump()
        type = model_dump.pop("type")
        return json.dumps({"type": type, "data": model_dump})


class AgentToolAction(AgentBaseEvent):
    type: str = "action"
    tool: str
    tool_input: str
    thought: str


class AgentFinalAnswer(AgentBaseEvent):
    type: str = "final_answer_delta"
    text: str


class AgentUnknownAction(AgentBaseEvent):
    type: str = "unknown"
    text: str


class AgentError(AgentBaseEvent):
    type: str = "error"
    message: str
    retryable: bool


TypeAgentEvent = TypeVar(
    "TypeAgentEvent", AgentToolAction, AgentFinalAnswer, AgentUnknownAction
)

TypeAgentInputs = TypeVar("TypeAgentInputs")


class AgentStep(BaseModel):
    action: Optional[AgentToolAction] = None
    observation: str


class CurrentFile(BaseModel):
    file_path: str
    data: str
    selected_code: bool


# Note: additionaL_context is an alias for injected_context
class AdditionalContext(BaseModel):
    category: Literal[
        "file",
        "snippet",
        "merge_request",
        "issue",
        "dependency",
        "local_git",
        "terminal",
    ]
    id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None


class Message(BaseModel):
    role: Role
    content: str
    context: Optional[CurrentPageContext] = None
    current_file: Optional[CurrentFile] = None
    additional_context: Optional[list[AdditionalContext]] = None
    agent_scratchpad: Optional[list[AgentStep]] = None

    @model_validator(mode="after")
    def validate_agent_scratchpad_role(self) -> Self:
        if self.agent_scratchpad is not None and self.role != Role.ASSISTANT:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="agent_scratchpad can only be present when role is ASSISTANT",
            )
        return self
