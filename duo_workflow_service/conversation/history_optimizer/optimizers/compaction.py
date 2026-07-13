"""LLM-driven compaction optimizer.

This module is the canonical home for ``CompactionOptimizer`` (formerly
``ConversationCompactor`` under ``duo_workflow_service.conversation.compaction``).
The old module re-exports ``CompactionOptimizer`` as ``ConversationCompactor``
for backwards compatibility while callers migrate.
"""

import time
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, cast
from uuid import uuid4

from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from structlog import get_logger

from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.optimizers._compaction_utils import (
    slice_for_summarization,
    strip_tool_metadata,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    CompactionResult,
    MessageSlices,
)
from duo_workflow_service.conversation.token_estimator import TokenEstimator
from duo_workflow_service.entities.state import (
    TOOL_RESPONSE_MAX_DISPLAY_MSG,
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    get_current_model_max_context_token_limit,
)
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.secret_redaction import redact_secrets_for_ui
from lib.context import StarletteUser, is_gitlab_team_member
from lib.context.model import get_model_metadata
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum

log = get_logger("compaction_optimizer")

COMPACTION_PROMPT_ID = "conversation_compaction"
COMPACTION_PROMPT_MANUAL_ID = "conversation_compaction_manual"
COMPACTION_PROMPT_VERSION = "^1.0.0"

COMPACTION_CONTINUE_MESSAGE = (
    "Continue working on the task based on the conversation above."
)


class CompactionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class CompactionOptimizer(HistoryOptimizer):
    """Handles conversation history compaction via summarization.

    Implements the ``HistoryOptimizer`` interface. ``optimize()`` runs the
    automatic, threshold-gated path; ``optimize_manual()`` runs the
    user-forced ``/compact`` path. The previous ``compact()`` method is
    preserved as the internal worker that both wrappers delegate to.

    Note on naming: the class was previously ``ConversationCompactor`` and
    the internal Snowplow event category string is intentionally kept as
    that legacy name for downstream analytics compatibility (see
    ``_fire_compaction_event``).
    """

    def __init__(
        self,
        prompt_registry: BasePromptRegistry,
        user: StarletteUser | CloudConnectorUser,
        config: CompactionConfig,
        agent_name: str,
        workflow_id: str,
        workflow_type: str,
        internal_events_client: InternalEventsClient | None = None,
    ):
        # Use the same (default) model as the parent workflow rather than a
        # smaller variant.  The compaction trigger threshold is based on the
        # workflow model's context window, so the messages handed to the
        # summarizer can exceed a smaller model's context limit and cause
        # API errors.  Using the default model guarantees the summarizer
        # can always handle the payload.
        # is_graph_node=True is a safety net: if no gRPC model metadata
        # context is set (tests, edge cases), it falls back to
        # duo_agent_platform instead of failing on an unknown feature
        # setting derived from the prompt_id.
        self._prompt_registry = prompt_registry
        self._user = user
        self._agent_name = agent_name
        self._workflow_id = workflow_id
        self._workflow_type = workflow_type
        self._model_name = getattr(get_model_metadata(), "name", "unknown")
        self._internal_events_client = internal_events_client
        self._config = config
        self._token_estimator = TokenEstimator()

        # Per-mode lazy prompt cache; populated on first compact() call for
        # each mode. Keys: False = auto prompt, True = manual prompt.
        self._prompts: dict[bool, Prompt] = {}

    async def optimize(self, history: list[BaseMessage]) -> CompactionResult:
        """Self-guarded automatic compaction.

        Always callable by the ``HistoryOptimizerPipeline``; internally
        short-circuits when ``should_compact(history)`` is False. When a
        summary is produced, populates ``result.ui_chat_logs`` with a single
        ``compaction`` tool card.
        """
        result = await self.compact(messages=history)
        entry = _maybe_build_auto_compaction_entry(result)
        if entry is not None:
            result.ui_chat_logs = [entry]
        return result

    async def optimize_manual(
        self,
        history: list[BaseMessage],
        *,
        user_instruction: str | None = None,
    ) -> CompactionResult:
        """User-forced compaction (``/compact``).

        Bypasses the auto-threshold and slicing fall-throughs; always
        attempts to produce a summary. On success, ``result.ui_chat_logs``
        carries a single tool card describing the compaction. On failure or
        no-op, ``ui_chat_logs`` carries a tool-card + agent-message pair
        matching the shape ``ChatAgent`` used to emit directly.
        """
        result = await self.compact(
            messages=history,
            is_manual=True,
            user_instruction=user_instruction,
        )
        result.ui_chat_logs = _build_manual_compaction_ui_logs(result)
        return result

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        """Determine if messages should be summarized.

        Summarization is recommended when:
        - Message count exceeds max_recent_messages, AND
        - Total token count exceeds trim_threshold of the model's max context limit

        Args:
            messages: List of messages to evaluate

        Returns:
            True if messages should be summarized, False otherwise
        """
        if len(messages) <= self._config.max_recent_messages:
            return False

        token_count = self._token_estimator.count(messages, is_complete_history=True)
        threshold = (
            self._config.trim_threshold * get_current_model_max_context_token_limit()
        )
        log.info(
            "Checking if compact is needed.",
            message_token=token_count,
            threshold=threshold,
            is_gitlab_team_member=is_gitlab_team_member.get(),
        )
        return token_count > threshold

    def _get_prompt(self, is_manual: bool) -> Prompt:
        if is_manual not in self._prompts:
            prompt_id = (
                COMPACTION_PROMPT_MANUAL_ID if is_manual else COMPACTION_PROMPT_ID
            )
            self._prompts[is_manual] = self._prompt_registry.get_on_behalf(
                self._user,
                prompt_id,
                COMPACTION_PROMPT_VERSION,
                model_metadata=get_model_metadata(),
                is_graph_node=True,
                internal_event_extra={
                    "agent_name": self._agent_name,
                    "workflow_id": self._workflow_id,
                    "workflow_type": self._workflow_type,
                    "is_compaction_call": True,
                },
            )
        return self._prompts[is_manual]

    async def compact(
        self,
        messages: list[BaseMessage],
        is_manual: bool = False,
        user_instruction: str | None = None,
    ) -> CompactionResult:
        """Compact conversation history by summarizing older messages.

        Keeps:
        - First consecutive HumanMessages (initial context)
        - Recent messages within token and count limits
        - Summarizes middle messages into a single AIMessage

        Args:
            messages: List of messages to compact
            is_manual: When True, bypass should_compact thresholds, force-summarize
                the entire history if normal slicing yields no to_summarize,
                skip the synthetic continuation HumanMessage, and let summary
                tokens stream to the UI.
            user_instruction: Optional user-supplied focus directive forwarded
                to the manual prompt template.

        Returns:
            CompactionResult with compacted messages and metadata
        """
        if not messages:
            return CompactionResult(messages=messages, was_modified=False)

        if not is_manual and not self.should_compact(messages):
            return CompactionResult(messages=messages, was_modified=False)

        slices = slice_for_summarization(messages, self._config, self._token_estimator)

        if not slices.to_summarize:
            if is_manual:
                slices = MessageSlices(
                    leading_context=[],
                    to_summarize=messages,
                    recent_to_keep=[],
                )
            else:
                log.warning(
                    "No messages to summarize",
                    n_msgs=len(messages),
                    n_leading=len(slices.leading_context),
                    n_recent=len(slices.recent_to_keep),
                )
                return CompactionResult(messages=messages, was_modified=False)

        original_tokens = self._token_estimator.count(
            messages, is_complete_history=True
        )
        start_time = time.time()

        try:
            with duo_workflow_metrics.time_compaction_llm(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
            ):
                summary = await self._invoke_summarizer(
                    slices.to_summarize,
                    is_manual=is_manual,
                    user_instruction=user_instruction,
                )
            duration = time.time() - start_time

            duo_workflow_metrics.count_compaction_execution(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
                status=CompactionStatus.SUCCESS,
            )
        except Exception as e:
            duration = time.time() - start_time

            duo_workflow_metrics.count_compaction_execution(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
                status=CompactionStatus.ERROR,
            )

            self._fire_compaction_event(
                is_manual=is_manual,
                status=CompactionStatus.ERROR,
                model_name=self._model_name,
                tokens_before=original_tokens,
                duration_seconds=duration,
                error_type=type(e).__name__,
            )

            log.error(
                "Failed to summarize messages, keeping original",
                error=str(e),
                error_type=type(e).__name__,
                n_msgs=len(messages),
                n_to_summarize=len(slices.to_summarize),
                is_gitlab_team_member=is_gitlab_team_member.get(),
                exc_info=True,
            )
            return CompactionResult(
                messages=messages,
                was_modified=False,
                error=e,
            )

        compacted_messages = slices.leading_context + [summary] + slices.recent_to_keep

        if not is_manual and isinstance(compacted_messages[-1], AIMessage):
            # Vertex AI / Gemini requires conversations to end with a user
            # message (strict role alternation: user <-> model). After
            # compaction, the message list may end with an AIMessage (summary
            # or recent agent turn), which the Gemini API rejects with HTTP
            # 400. Append a synthetic HumanMessage to satisfy this constraint.
            # Note: ToolMessages are not affected -- the Gemini API converts
            # them to role-less ContentType objects that don't participate in
            # role alternation.
            #
            # In manual mode, control returns to the user; their next message
            # naturally satisfies the constraint without a synthetic marker.
            compacted_messages.append(HumanMessage(content=COMPACTION_CONTINUE_MESSAGE))

        usage = summary.usage_metadata
        compaction_input_tokens = usage.get("input_tokens", 0) if usage else 0
        compaction_output_tokens = usage.get("output_tokens", 0) if usage else 0

        compacted_tokens = self._calculate_compacted_tokens(
            original_tokens,
            compaction_input_tokens,
            compaction_output_tokens,
            is_manual,
            user_instruction,
        )

        max_context_tokens = get_current_model_max_context_token_limit()
        self._fire_compaction_event(
            is_manual=is_manual,
            status=CompactionStatus.SUCCESS,
            model_name=self._model_name,
            compaction_input_tokens=compaction_input_tokens,
            compaction_output_tokens=compaction_output_tokens,
            tokens_before=original_tokens,
            tokens_after=compacted_tokens,
            tokens_saved=original_tokens - compacted_tokens,
            compaction_ratio=(
                round(1 - compacted_tokens / original_tokens, 4)
                if original_tokens > 0
                else 0
            ),
            messages_summarized=len(slices.to_summarize),
            token_budget=int(self._config.trim_threshold * max_context_tokens),
            max_context_tokens=max_context_tokens,
            duration_seconds=round(duration, 3),
        )

        log.info(
            "Finish context compaction",
            is_manual=is_manual,
            num_msg_before=len(messages),
            num_msg_after=len(compacted_messages),
            msg_tokens_before=original_tokens,
            msg_tokens_after=compacted_tokens,
            compaction_ratio=1 - compacted_tokens / original_tokens,
            compaction_usage_data=summary.usage_metadata,
            compaction_model_details=summary.response_metadata,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            agent_name=self._agent_name,
            model_name=self._model_name,
            compaction_config=self._config.model_dump(),
            is_gitlab_team_member=is_gitlab_team_member.get(),
        )

        return CompactionResult(
            messages=compacted_messages,
            was_modified=True,
            tokens_before=original_tokens,
            tokens_after=compacted_tokens,
            messages_summarized=len(slices.to_summarize),
            compaction_input_tokens=compaction_input_tokens,
            compaction_output_tokens=compaction_output_tokens,
            summary=summary,
        )

    async def _invoke_summarizer(
        self,
        messages: list[BaseMessage],
        is_manual: bool = False,
        user_instruction: str | None = None,
    ) -> AIMessage:
        """Invoke the LLM to summarize messages.

        The "nostream" tag suppresses summarization tokens from LangGraph's stream_mode="messages" channel for both auto
        and manual modes; the compaction outcome is surfaced as a single terminal UI entry instead. See
        langgraph.constants.TAG_NOSTREAM.
        """
        log.info(
            "Start compaction summarization llm call.",
            is_manual=is_manual,
            is_gitlab_team_member=is_gitlab_team_member.get(),
        )
        messages = strip_tool_metadata(messages)

        config: RunnableConfig = {
            "tags": [TAG_NOSTREAM],
            "run_name": "Compaction Summarization",
        }

        inputs: dict[str, object] = {"history": messages}
        if is_manual:
            inputs["user_instruction"] = user_instruction

        prompt = self._get_prompt(is_manual)
        result = await prompt.ainvoke(inputs, config=config)
        return cast(AIMessage, result)

    def _calculate_compacted_tokens(
        self,
        original_tokens: int,
        input_tokens: int,
        output_tokens: int,
        is_manual: bool,
        user_instruction: str | None = None,
    ) -> int:
        """Calculate the token count after compaction.

        The prompt-template overhead is rendered from the same inputs ``_invoke_summarizer`` uses, so the estimate
        stays consistent with the actual LLM call. ``user_instruction`` only affects rendering in manual mode.
        """
        overhead_inputs: dict[str, object] = {"history": []}
        if is_manual:
            overhead_inputs["user_instruction"] = user_instruction
        prompt_tpl = cast(ChatPromptTemplate, self._get_prompt(is_manual).prompt_tpl)
        overhead = self._token_estimator.count(
            prompt_tpl.format_messages(**overhead_inputs),
            is_complete_history=False,
        )

        return original_tokens - (input_tokens - overhead) + output_tokens

    def _fire_compaction_event(self, is_manual: bool, **kwargs: Any) -> None:
        """Fire a compaction_executed Snowplow event."""
        if self._internal_events_client is None:
            return
        # operation_type is authoritative from the prompt YAML; override any
        # caller-supplied value in kwargs.
        prompt = self._prompts.get(is_manual)
        if prompt is None:
            return
        # NOTE: category string is intentionally kept as
        # ``"ConversationCompactor"`` despite the class rename to
        # ``CompactionOptimizer``. Downstream analytics dashboards depend on
        # the exact legacy string; renaming would require a coordinated
        # analytics migration. The category change is tracked separately and
        # is out of scope for this refactor.
        self._internal_events_client.track_event(
            event_name=EventEnum.COMPACTION_EXECUTED.value,
            additional_properties=InternalEventAdditionalProperties(
                label=self._agent_name,
                property="workflow_id",
                value=self._workflow_id,
                workflow_type=self._workflow_type,
                **{**kwargs, "operation_type": prompt.operation_type},
            ),
            category="ConversationCompactor",
        )


def build_compaction_tool_card(
    trigger: Literal["auto", "manual"],
    result: CompactionResult | None,
    status: ToolStatus,
    content: str | None = None,
) -> UiChatLog:
    """Build a ``UiChatLog`` tool card for any compaction outcome.

    Mirrors the shape produced by ``ToolsExecutor._create_tool_ui_chat_log``
    so the client renders it uniformly: success cards carry the summary in
    ``tool_info.tool_response`` (a ``ToolMessage`` to match the FE
    deserialization contract); no-op / failure cards still populate
    ``tool_info`` with the args metadata but omit ``tool_response``,
    matching how real tool failures are rendered.

    ``content`` defaults to the ``"Summarized N message(s)"`` summary line
    and should be overridden for no-op / failure entries.
    """
    n = result.messages_summarized if result is not None else 0
    if content is None:
        plural = "s" if n != 1 else ""
        content = f"Summarized {n} message{plural}"

    tool_call_id = f"compaction-{uuid4()}"
    args: dict[str, Any] = {
        "trigger": trigger,
        "messages_summarized": n,
        "compaction_input_tokens": (
            result.compaction_input_tokens if result is not None else 0
        ),
        "compaction_output_tokens": (
            result.compaction_output_tokens if result is not None else 0
        ),
    }

    tool_info = ToolInfo(name="compaction", args=args)
    summary = result.summary if result is not None else None
    if summary is not None:
        redacted = redact_secrets_for_ui(summary.text(), tool_name="compaction")
        tool_info["tool_response"] = ToolMessage(
            content=redacted[:TOOL_RESPONSE_MAX_DISPLAY_MSG],
            name="compaction",
            tool_call_id=tool_call_id,
            status="success",
        )

    return UiChatLog(
        message_type=MessageTypeEnum.TOOL,
        message_sub_type="compaction",
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
        correlation_id=None,
        tool_info=tool_info,
        additional_context=None,
        message_id=tool_call_id,
    )


def build_compaction_agent_message(content: str) -> UiChatLog:
    """Build an AGENT-typed ``UiChatLog`` carrying a user-facing compaction notice.

    Front end ignores ``content`` on failed tool cards, so non-success
    compaction paths emit this entry alongside the tool card to surface the
    explanation to the user.
    """
    return UiChatLog(
        message_type=MessageTypeEnum.AGENT,
        message_sub_type=None,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=ToolStatus.SUCCESS,
        correlation_id=None,
        tool_info=None,
        additional_context=None,
        message_id=f"compaction-{uuid4()}",
    )


def _maybe_build_auto_compaction_entry(
    result: CompactionResult | None,
) -> UiChatLog | None:
    if result is None or not result.succeeded:
        return None
    return build_compaction_tool_card(
        trigger="auto",
        result=result,
        status=ToolStatus.SUCCESS,
    )


def _build_manual_compaction_ui_logs(result: CompactionResult) -> list[UiChatLog]:
    """Build the UI entries surfaced after a manual ``/compact`` invocation.

    Matches the pre-refactor ``ChatAgent`` behavior:
    - success: single tool card carrying the summary
    - failure (error or missing summary): tool card + agent message pair
    """
    if result.succeeded:
        return [
            build_compaction_tool_card(
                trigger="manual",
                result=result,
                status=ToolStatus.SUCCESS,
            )
        ]

    return [
        build_compaction_tool_card(
            trigger="manual",
            result=result,
            content="Compaction failed",
            status=ToolStatus.FAILURE,
        ),
        build_compaction_agent_message(
            "Failed to compact conversation. Please try again."
        ),
    ]


def create_conversation_compactor(
    config: CompactionConfig,
    prompt_registry: BasePromptRegistry,
    user: StarletteUser | CloudConnectorUser,
    agent_name: str,
    workflow_id: str,
    workflow_type: str,
    internal_events_client: InternalEventsClient | None = None,
) -> CompactionOptimizer:
    """Build a ``CompactionOptimizer``.

    Preserved with its legacy name and signature so existing callers continue to work unchanged through MR 1. Migration
    happens in MR 2 / MR 3.
    """
    return CompactionOptimizer(
        prompt_registry=prompt_registry,
        user=user,
        config=config,
        agent_name=agent_name,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        internal_events_client=internal_events_client,
    )
