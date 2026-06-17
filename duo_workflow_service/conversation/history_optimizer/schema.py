"""Config and result types for the history optimizer abstraction.

This module is the canonical home for the optimizer config types and the
result types returned from ``HistoryOptimizer.optimize()``. The legacy
``duo_workflow_service.conversation.compaction.schema`` module re-exports
``CompactionConfig`` and ``CompactionResult`` from here for backwards
compatibility while callers migrate.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, Union

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from duo_workflow_service.entities.state import UiChatLog


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction."""

    model_config = ConfigDict(frozen=True)

    type: Literal["compaction"] = "compaction"

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


class LegacyTrimConfig(BaseModel):
    """Configuration for the legacy token-based trim optimizer.

    Defaults (token budget, threshold) live in
    ``duo_workflow_service.conversation.trimmer``; this config has no fields
    today but the discriminator key keeps the YAML/Python surfaces symmetric.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["legacy_trim"] = "legacy_trim"


class ToolResultPrunerConfig(BaseModel):
    """Stub config for the future ``ToolResultPruner`` optimizer.

    The optimizer itself is not implemented in MR 1; the builder raises
    ``NotImplementedError`` when it encounters this config. The schema entry
    exists so YAML flows can reference the future optimizer once it lands.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_result_pruner"] = "tool_result_pruner"


HistoryOptimizerConfig = Annotated[
    Union[CompactionConfig, LegacyTrimConfig, ToolResultPrunerConfig],
    Field(discriminator="type"),
]
"""Discriminated union of all supported optimizer configs.

Pydantic models that reference this alias get correct ``type``-discriminated
parsing for free.
"""


@dataclass
class MessageSlices:
    """Result of slicing messages for summarization."""

    leading_context: list[BaseMessage]
    to_summarize: list[BaseMessage]
    recent_to_keep: list[BaseMessage]


@dataclass
class OptimizationResult:
    """Base result returned from ``HistoryOptimizer.optimize()``.

    ``ui_chat_logs`` carries UI artifacts (tool cards, agent messages) the
    optimizer wants surfaced to the user. The pipeline does not interpret
    this field; callers (agents/components) decide whether and where to
    append it onto their state. Optimizers with no UI surface (e.g.
    ``LegacyTrimOptimizer``) leave it empty.
    """

    messages: list[BaseMessage] = field(default_factory=list)
    was_modified: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    duration_ms: float = 0.0
    optimizer_name: str = ""
    ui_chat_logs: list[UiChatLog] = field(default_factory=list)


@dataclass
class CompactionResult(OptimizationResult):
    """Result of a compaction (LLM summarization) operation.

    Constructed with ``was_modified``; the read property ``was_compacted``
    is kept as an alias so existing call sites reading
    ``result.was_compacted`` continue to work unchanged.
    """

    summary: AIMessage | None = None
    messages_summarized: int = 0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    error: Exception | None = field(default=None, repr=False)

    @property
    def was_compacted(self) -> bool:
        """Legacy read alias for ``was_modified``.

        Preserved so existing call sites that read ``result.was_compacted``
        keep working until they migrate. Construction must use
        ``was_modified=...``.
        """
        return self.was_modified


@dataclass
class TrimResult(OptimizationResult):
    """Result of a token-based trim operation."""

    token_budget: int = 0
    max_context_tokens: int = 0
