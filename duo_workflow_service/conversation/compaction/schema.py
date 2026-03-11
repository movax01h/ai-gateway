from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, field_validator

from duo_workflow_service.entities.state import MessageTypeEnum, UiChatLog

SUMMARIZER_SYSTEM_PROMPT = """
You are a helpful AI assistant tasked with summarizing conversations.
When asked to summarize, provide a detailed but concise summary of the conversation.
Focus on information that would be helpful for continuing the conversation, including:
- What was done
- What is currently being worked on
- Which files are being modified
- What needs to be done next
- Key user requests, constraints, or preferences that should persist
- Important technical decisions and why they were made
"""

SUMMARIZER_USER_PROMPT = """
Provide a detailed prompt for continuing our conversation above. Focus on information
that would be helpful for continuing the conversation, including what we did, what we're
doing, which files we're working on, and what we're going to do next considering new
session will not have access to our conversation.
"""


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction."""

    model_config = ConfigDict(frozen=True)

    max_recent_messages: int = 10
    recent_messages_token_budget: int = 40_000
    trim_threshold: float = 0.7
    summarizer_system_prompt: str = SUMMARIZER_SYSTEM_PROMPT
    summarizer_user_prompt: str = SUMMARIZER_USER_PROMPT

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
    error: Exception | None = field(default=None, repr=False)

    def build_ui_chat_log(self) -> UiChatLog | None:
        """Build a UI chat log message if compaction occurred."""
        if not self.was_compacted:
            return None

        return UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content="I summarized the previous conversation to fit in the context window.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=None,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
            message_id=None,
        )
