from abc import ABC, abstractmethod
from typing import ClassVar, Self, Type

from langchain_core.messages import AIMessage
from pydantic import BaseModel

__all__ = ["BaseAgentOutput", "BaseResponseSchemaRegistry"]


class BaseAgentOutput(BaseModel):
    tool_title: ClassVar[str]

    @classmethod
    def from_ai_message(cls, msg: AIMessage) -> Self:
        return cls(**msg.tool_calls[0]["args"])

    def to_string_output(self) -> str:
        """Return a human-readable string representation for UI display.

        The JSON is wrapped in a Markdown code fence so the UI renders it as a highlighted code block.
        """
        return f"```json\n{self.model_dump_json(indent=2)}\n```"

    def to_output(self) -> str | dict:
        """Return the canonical output value for writing to flow state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement to_output()"
        )


class BaseResponseSchemaRegistry(ABC):
    @abstractmethod
    def get(self, schema_id: str, schema_version: str) -> Type[BaseAgentOutput]:
        pass
