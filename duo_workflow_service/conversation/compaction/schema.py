from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, field_validator


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction."""

    model_config = ConfigDict(frozen=True)

    max_recent_messages: int = 10
    recent_messages_token_budget: int = 40_000
    trim_threshold: float = 0.7

    @field_validator("max_recent_messages")
    @classmethod
    def validate_max_recent_messages(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_recent_messages must be positive")
        return v

    @field_validator("recent_messages_token_budget")
    @classmethod
    def validate_recent_messages_token_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("recent_messages_token_budget must be positive")
        return v

    @field_validator("trim_threshold")
    @classmethod
    def validate_trim_threshold(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("trim_threshold must be between 0 (exclusive) and 1")
        return v


@dataclass
class MessageSlices:
    """Result of slicing messages for summarization."""

    leading_context: list[BaseMessage]
    to_summarize: list[BaseMessage]
    recent_to_keep: list[BaseMessage]


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    messages: list[BaseMessage]
    was_compacted: bool
    tokens_before: int = 0
    tokens_after: int = 0
    messages_summarized: int = 0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    error: Exception | None = field(default=None, repr=False)
    summary: AIMessage | None = None
