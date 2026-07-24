from datetime import datetime
from http import HTTPStatus
from typing import Any, ClassVar, Optional, Sequence, Type, cast

import structlog
from anthropic import APIStatusError
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field
from pydantic_core import ValidationError

from ai_gateway.prompts import Prompt
from ai_gateway.response_schemas.base import BaseAgentOutput
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    RuntimeIOKey,
    get_vars_from_state,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.conversation.compaction import (
    ConversationCompactor,
    maybe_compact_history,
)
from duo_workflow_service.conversation.trimmer import restore_message_consistency
from duo_workflow_service.errors.error_handler import (
    ModelError,
    ModelErrorHandler,
    ModelErrorType,
)
from lib.context import LLMFinishReason, extract_finish_reason
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient

__all__ = ["AgentFinalOutput", "AgentNode", "AgentStuckError"]

log = structlog.stdlib.get_logger("agent_node")

# LiteLLM injects this placeholder into empty text blocks when formatting messages
# for the Anthropic API. Filter it out so it never reaches the UI chat log.
_LITELLM_EMPTY_CONTENT_PLACEHOLDER = (
    "[System: Empty message content sanitised to satisfy protocol]"
)


class AgentStuckError(Exception):
    """Exception raised when an agent exceeds the maximum number of truncation retries.

    This indicates the agent is stuck in an unrecoverable loop where the LLM repeatedly produces truncated responses
    despite recovery attempts.
    """


class AgentFinalOutput(BaseAgentOutput):
    """
    MANDATORY COMPLETION TOOL: You MUST use this tool to provide your final answer
    when you have completed the user's request. This is the ONLY way to properly
    end the conversation and deliver your response to the user.
    Use this tool when:
    1. You have gathered all necessary information
    2. You have completed the requested task or analysis
    3. You are ready to give your final answer
    CRITICAL: Always use this tool to conclude your work - do not continue
    using other tools once you have the information needed to answer the user.
    The final_response must contain your complete answer.
    """

    model_config = ConfigDict(title="final_response_tool", frozen=True)

    tool_title: ClassVar[str] = "final_response_tool"

    final_response: str = Field(
        description="The final response to the user to communicate work completion"
    )

    def to_string_output(self) -> str:
        return self.final_response

    def to_output(self) -> str:
        return self.final_response


class AgentNode:  # pylint: disable=too-many-instance-attributes
    """LangGraph node that invokes an LLM and appends the completion to conversation history.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary
    access.

    The conversation-history ``IOKey`` is resolved dynamically at runtime via
    ``conversation_history_key``.  This supports both the common case (static key
    wrapped in a ``RuntimeIOKey`` by the caller) and the supervisor case where
    the key is only known at runtime.

    When ``ui_history`` is provided and ``ON_AGENT_REASONING`` is included in its
    event list, a ``MessageTypeEnum.AGENT`` entry is emitted for every ``AIMessage``
    whose text content is non-empty (even when the message also contains tool calls).
    This surfaces the LLM's reasoning commentary in the session view.

    Args:
        flow_id: Identifier of the current flow execution.
        flow_type: Reporting context used for internal events and metrics.
        name: LangGraph node name (used when registering with ``StateGraph``).
        prompt: Bound ``Prompt`` instance used to invoke the LLM.
        inputs: ``IOKey`` list describing which state values to pass as prompt
            variables.
        conversation_history_key: ``RuntimeIOKey`` that resolves the
            conversation-history ``IOKey`` at runtime.
        internal_event_client: Client for tracking internal telemetry events.
        ui_history: Optional UI log history writer.  When provided, reasoning
            text from mid-loop ``AIMessage``s is emitted as
            ``ON_AGENT_REASONING`` entries (if that event is enabled).
        invoke_config: ``RunnableConfig`` dict passed to every ``ainvoke``
            call on the prompt.  Callers must always provide this explicitly —
            pass ``{}`` for streaming-eligible components (no suppression) or
            ``{"tags": [TAG_NOSTREAM]}`` to suppress LLM chunks from the
            LangGraph ``messages`` stream.  Making this required ensures every
            ``AgentNode`` construction site makes a deliberate streaming
            decision rather than silently inheriting a default.
        max_context_tokens: Context-window limit of the model this agent runs on.
            When set, it is stamped into ``agent_context_limits`` (keyed by the
            agent's conversation-history slot) so checkpoints can report per-agent
            context utilisation.  When ``None``, no limit is stamped and the
            notifier falls back to the global context-window limit.
        iteration_warning_offset: Number of cycles before ``max_cycles`` at which a
            one-time warning ``HumanMessage`` is injected, telling the agent it is
            approaching the soft limit. ``None`` disables the warning. Ignored when
            ``max_cycles`` is ``None``.
    """

    name: str
    _prompt: Prompt

    _inputs: Sequence[IOKey | RuntimeIOKey]

    _conversation_history_key: RuntimeIOKey

    _internal_event_client: InternalEventsClient

    _flow_id: str
    _flow_type: GLReportingEventContext
    _error_handler: ModelErrorHandler
    _compactor: ConversationCompactor | None
    _ui_history: Optional[UIHistory[UILogWriterAgentTools, UILogEventsAgent]]
    _max_context_tokens: Optional[int]
    _invoke_config: RunnableConfig
    _max_cycles: Optional[int]
    _cycle_count_key: Optional[RuntimeIOKey]
    _max_wrap_up_retries: int
    _iteration_warning_offset: Optional[int]

    def __init__(
        self,
        flow_id: str,
        flow_type: GLReportingEventContext,
        name: str,
        prompt: Prompt,
        inputs: Sequence[IOKey | RuntimeIOKey],
        internal_event_client: InternalEventsClient,
        conversation_history_key: RuntimeIOKey,
        invoke_config: RunnableConfig,
        compactor: ConversationCompactor | None = None,
        response_schema: Optional[Type[BaseAgentOutput]] = None,
        ui_history: Optional[UIHistory[UILogWriterAgentTools, UILogEventsAgent]] = None,
        max_context_tokens: Optional[int] = None,
        max_cycles: Optional[int] = None,
        cycle_count_key: Optional[RuntimeIOKey] = None,
        max_wrap_up_retries: int = 3,
        iteration_warning_offset: Optional[int] = None,
        prompt_template_inputs: Optional[dict[str, Any]] = None,
    ):
        self._flow_id = flow_id
        self._flow_type = flow_type
        self.name = name
        self._prompt = prompt
        self._inputs = inputs
        self._internal_event_client = internal_event_client
        self._error_handler = ModelErrorHandler()
        self._compactor = compactor
        self._conversation_history_key = conversation_history_key
        self._response_schema = response_schema
        self._ui_history = ui_history
        self._max_context_tokens = max_context_tokens
        self._invoke_config = invoke_config
        self._max_cycles = max_cycles
        self._cycle_count_key = cycle_count_key
        self._max_wrap_up_retries = max_wrap_up_retries
        self._iteration_warning_offset = iteration_warning_offset
        # Build-time template variables (e.g. which optional tools/capabilities are
        # active) that the prompt can branch on. Merged into every prompt invocation
        # alongside the runtime variables below.
        self._prompt_template_inputs = prompt_template_inputs or {}

    _TRUNCATION_RECOVERY_MESSAGE = (
        "Your response was too long and got cut off. "
        "Be more concise and use smaller, incremental steps."
    )

    _MAX_CYCLES_REACHED_MESSAGE = (
        "You have reached the maximum number of iterations for this task. "
        "You must now {instruction}."
    )

    _ITERATION_WARNING_MESSAGE = (
        "You are approaching the maximum number of iterations for this task "
        "({cycles_remaining} remaining). Once you reach that limit, you will "
        "no longer be able to make any tool calls and must provide your final "
        "answer immediately. Start wrapping up your work now."
    )

    _MAX_TRUNCATION_RETRIES: int = 5

    @staticmethod
    def _extract_text(completion: AIMessage) -> str:
        """Extract plain text from an ``AIMessage``, handling both string and list content.

        LiteLLM placeholder strings (injected when sanitising empty text blocks for
        the Anthropic API) are filtered out and never included in the returned text.

        Args:
            completion: The ``AIMessage`` to extract text from.

        Returns:
            The text content of the message, or an empty string if none.
        """
        content = completion.content
        if isinstance(content, str):
            return "" if content == _LITELLM_EMPTY_CONTENT_PLACEHOLDER else content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    if block != _LITELLM_EMPTY_CONTENT_PLACEHOLDER:
                        parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text != _LITELLM_EMPTY_CONTENT_PLACEHOLDER:
                        parts.append(text)
            return "".join(parts)
        return ""

    def _emit_reasoning(self, completion: AIMessage) -> None:
        """Emit an ``ON_AGENT_REASONING`` log entry if the completion has non-empty text.

        Only emits when ``ui_history`` is set and ``ON_AGENT_REASONING`` is in its
        event list.  Empty or whitespace-only content is silently skipped.

        Args:
            completion: The ``AIMessage`` returned by the LLM.
        """
        if self._ui_history is None:
            return
        if UILogEventsAgent.ON_AGENT_REASONING not in self._ui_history.events:
            return
        text = self._extract_text(completion).strip()
        if not text:
            return
        self._ui_history.log.warning(
            text,
            event=UILogEventsAgent.ON_AGENT_REASONING,
            message_id=completion.id,
        )

    @staticmethod
    def _predefined_runtime_variables() -> dict[str, str]:
        now = datetime.now()

        return {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "current_timezone": now.tzname() or "",
        }

    def _check_and_increment_cycle_count(self, state: FlowState) -> tuple[int, dict]:
        """Read current cycle count from state, increment it, and return updated state fragment.

        Returns:
            A tuple of (new_cycle_count, state_update_dict).
            state_update_dict is empty when max_cycles is not configured.
        """
        if self._max_cycles is None or self._cycle_count_key is None:
            return 0, {}

        cycle_count_iokey = self._cycle_count_key.to_iokey(state)
        current_count: int = cycle_count_iokey.value_from_state(state) or 0
        new_count = current_count + 1
        state_update = cycle_count_iokey.to_nested_dict(new_count)
        return new_count, state_update

    def _wrap_up_message(self) -> str:
        """Return the wrap-up instruction message, tailored to the response schema if configured."""
        if self._response_schema is not None:
            instruction = (
                f"call the {self._response_schema.tool_title} tool to provide your "
                "final answer without making any other tool calls"
            )
        else:
            instruction = (
                "provide your final answer without making any further tool calls"
            )
        return self._MAX_CYCLES_REACHED_MESSAGE.format(instruction=instruction)

    def _iteration_warning_message(self, cycles_remaining: int) -> str:
        """Return the approaching-soft-limit warning message."""
        return self._ITERATION_WARNING_MESSAGE.format(cycles_remaining=cycles_remaining)

    def _completion_has_non_final_tool_calls(self, completion: AIMessage) -> bool:
        """Return True if the completion contains tool calls other than the final response tool.

        When a response schema is configured, only the schema's tool call counts as a final answer. Without a schema,
        any text-only response (no tool calls) is a final answer.
        """
        if self._response_schema is None and not completion.tool_calls:
            return False
        if self._response_schema is not None:
            return not all(
                tc["name"] == self._response_schema.tool_title
                for tc in completion.tool_calls
            )
        return True

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key.to_iokey(state)
        history = history_iokey.value_from_state(state) or []
        variables = get_vars_from_state(self._inputs, state)

        history, _ = await maybe_compact_history(
            compactor=self._compactor, history=history, agent_name=self.name
        )
        history = restore_message_consistency(history)

        cycle_count, cycle_count_state_update = self._check_and_increment_cycle_count(
            state
        )
        wrap_up_active = (
            self._max_cycles is not None and cycle_count >= self._max_cycles
        )
        if wrap_up_active:
            log.warning(
                "Agent reached max_cycles soft limit; injecting wrap-up instruction",
                agent=self.name,
                cycle_count=cycle_count,
                max_cycles=self._max_cycles,
            )
            history = [*history, HumanMessage(content=self._wrap_up_message())]
        elif (
            self._max_cycles is not None
            and self._iteration_warning_offset is not None
            and cycle_count == self._max_cycles - self._iteration_warning_offset
        ):
            cycles_remaining = self._max_cycles - cycle_count
            log.warning(
                "Agent approaching max_cycles soft limit; injecting warning",
                agent=self.name,
                cycle_count=cycle_count,
                max_cycles=self._max_cycles,
                cycles_remaining=cycles_remaining,
            )
            history = [
                *history,
                HumanMessage(content=self._iteration_warning_message(cycles_remaining)),
            ]

        wrap_up_retries: int = 0
        truncation_retries: int = 0

        while True:
            try:
                completion: AIMessage = cast(
                    AIMessage,
                    await self._prompt.ainvoke(
                        input={
                            **self._prompt_template_inputs,
                            **variables,
                            "history": history,
                            **self._predefined_runtime_variables(),
                        },
                        config=self._invoke_config,
                    ),
                )
                finish_reason = extract_finish_reason(completion.response_metadata)
                if finish_reason in LLMFinishReason.truncation_values():
                    truncation_retries += 1
                    log.warning(
                        "LLM response was truncated due to token limit; "
                        "injecting recovery message and retrying within AgentNode",
                        finish_reason=finish_reason,
                        truncation_retries=truncation_retries,
                        max_truncation_retries=self._MAX_TRUNCATION_RETRIES,
                    )
                    if truncation_retries >= self._MAX_TRUNCATION_RETRIES:
                        raise AgentStuckError(
                            f"Agent '{self.name}' is stuck in an unrecoverable loop: "
                            f"LLM response was truncated {truncation_retries} times in a row, "
                            f"exceeding the maximum of {self._MAX_TRUNCATION_RETRIES} retries."
                        )
                    history = restore_message_consistency(
                        [
                            *history,
                            completion,
                            HumanMessage(content=self._TRUNCATION_RECOVERY_MESSAGE),
                        ]
                    )
                    continue

                if finish_reason in LLMFinishReason.abnormal_values():
                    log.warning(f"LLM stopped abnormally with reason: {finish_reason}")

                if len(updates := self._final_answer_validate(completion)) > 0:
                    history = [*history, *updates]
                    continue

                if wrap_up_active and self._completion_has_non_final_tool_calls(
                    completion
                ):
                    wrap_up_retries += 1
                    log.warning(
                        "Agent ignored wrap-up instruction and made non-final tool calls; "
                        "re-injecting wrap-up instruction",
                        agent=self.name,
                        wrap_up_retries=wrap_up_retries,
                        max_wrap_up_retries=self._max_wrap_up_retries,
                    )
                    if wrap_up_retries >= self._max_wrap_up_retries:
                        raise AgentStuckError(
                            f"Agent '{self.name}' is stuck: "
                            f"ignored the wrap-up instruction {wrap_up_retries} times in a row, "
                            f"exceeding the maximum of {self._max_wrap_up_retries} retries."
                        )
                    history = restore_message_consistency(
                        [
                            *history,
                            completion,
                            HumanMessage(content=self._wrap_up_message()),
                        ]
                    )
                    continue

                # Only emit reasoning if there are tool calls (i.e. omit for text-only messages)
                if completion.tool_calls:
                    self._emit_reasoning(completion)

                # Append new completion to existing history for replace-based reducer.
                # The reducer will replace this component's conversation history with
                # the complete list returned here.
                ui_updates = (
                    self._ui_history.pop_state_updates() if self._ui_history else {}
                )
                state_update = merge_nested_dict(
                    ui_updates,
                    history_iokey.to_nested_dict(history + [completion]),
                )
                result = merge_nested_dict(
                    state_update,
                    self._agent_context_limits_update(history_iokey),
                )
                return merge_nested_dict(result, cycle_count_state_update)
            except ContextOverflowError as e:
                model_error = ModelError(
                    error_type=ModelErrorType.REQUEST_TOO_LARGE,
                    status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    message=str(e),
                )

                await self._error_handler.handle_error(model_error)
            except APIStatusError as e:
                error_message = str(e)
                status_code = e.response.status_code
                model_error = ModelError(
                    error_type=self._error_handler.get_error_type(status_code),
                    status_code=status_code,
                    message=error_message,
                )

                await self._error_handler.handle_error(model_error)

    def _agent_context_limits_update(self, history_iokey: IOKey) -> dict:
        """Stamp ``{agent_key: max_context_tokens}`` keyed off ``history_iokey``.

        Keying off the conversation_history slot (not ``self.name``) keeps the
        limit aligned with the notifier's token-total map, including supervisor
        subagents. Returns an empty dict when no limit was supplied.
        """
        if self._max_context_tokens is None:
            return {}

        limits_iokey = IOKey(
            target="agent_context_limits",
            subkeys=history_iokey.subkeys,
        )
        return limits_iokey.to_nested_dict(self._max_context_tokens)

    def _final_answer_validate(self, completion: AIMessage) -> list:
        if self._response_schema is None:
            return []

        tool_title = self._response_schema.tool_title

        final_answer = next(
            (
                tool_call
                for tool_call in completion.tool_calls
                if tool_call["name"] == tool_title
            ),
            None,
        )

        if not final_answer:
            return []

        if len(completion.tool_calls) > 1:
            return [completion] + [
                ToolMessage(
                    content=f"{tool_title} mustn't be combined with other tool calls",
                    tool_call_id=tool_call["id"],
                )
                for tool_call in completion.tool_calls
            ]

        try:
            self._response_schema.from_ai_message(completion)
            return []
        except ValidationError as ve:
            return [
                completion,
                ToolMessage(
                    content=f"{tool_title} raised validation error: {ve}",
                    tool_call_id=final_answer["id"],
                ),
            ]
