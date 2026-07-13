"""Optimizer that wraps the legacy token-based trim path."""

from langchain_core.messages import BaseMessage

from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.schema import TrimResult
from duo_workflow_service.conversation.trimmer import apply_token_based_trim
from duo_workflow_service.entities.state import (
    get_current_model_max_context_token_limit,
)
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum


class LegacyTrimOptimizer(HistoryOptimizer):
    """Token-based trim path adapted to the ``HistoryOptimizer`` interface.

    Wraps ``apply_token_based_trim`` and fires the
    ``LEGACY_TRIM_EXECUTED`` internal event when a trim actually occurred.
    The optimizer never produces UI artifacts (trim has no UI today), so
    ``ui_chat_logs`` is left empty.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        internal_events_client: InternalEventsClient,
    ):
        self._agent_name = agent_name
        self._internal_events_client = internal_events_client

    async def optimize(self, history: list[BaseMessage]) -> TrimResult:
        """Run token-based trim.

        Self-guarded by ``apply_token_based_trim``.
        """
        result = apply_token_based_trim(
            messages=history,
            component_name=self._agent_name,
            max_context_tokens=get_current_model_max_context_token_limit(),
        )

        trim_result = TrimResult(
            messages=result.messages,
            was_modified=result.was_trimmed,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            duration_ms=result.duration_ms,
            token_budget=result.token_budget,
            max_context_tokens=result.max_context_tokens,
        )

        if result.was_trimmed:
            self._internal_events_client.track_event(
                event_name=EventEnum.LEGACY_TRIM_EXECUTED.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=self._agent_name,
                    tokens_before=result.tokens_before,
                    tokens_after=result.tokens_after,
                    tokens_removed=result.tokens_before - result.tokens_after,
                    messages_before=result.messages_before,
                    messages_after=result.messages_after,
                    token_budget=result.token_budget,
                    max_context_tokens=result.max_context_tokens,
                    duration_ms=result.duration_ms,
                ),
                category="legacy_trimmer",
            )

        return trim_result
