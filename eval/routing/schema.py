from enum import Enum
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, convert_to_messages
from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_LS_DATASET_INPUTS_SCHEMA = {
    "type": "object",
    "title": "dataset_input_schema",
    "required": ["messages"],
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "$ref": "https://api.smith.langchain.com/public/schemas/v1/message.json"
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "$ref": "https://api.smith.langchain.com/public/schemas/v1/tooldef.json"
            },
        },
    },
}

DEFAULT_LS_DATASET_OUTPUTS_SCHEMA = {
    "type": "object",
    "title": "dataset_output_schema",
    "properties": {
        "tool": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["name", "args"],
        }
    },
    "required": ["tool"],
}


class EvalDataset(str, Enum):
    LOCAL = "local"
    MR = "mr"
    MAIN = "main"

    @property
    def full_path(self) -> str:
        return f"routing_eval.dataset.{self.value}"


class RoutingCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[BaseMessage]
    expected_tool_input: Dict[str, Any]

    @field_validator("messages", mode="before")
    @classmethod
    def parse_messages(cls, v: Any) -> List[BaseMessage]:
        """Convert dict messages to appropriate LangChain message objects."""
        if not isinstance(v, list):
            raise ValueError("messages must be a list")

        return convert_to_messages(v)


class ToolRoutingEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    cases: List[RoutingCase]
