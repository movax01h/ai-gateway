from typing import ClassVar, Optional, Sequence, Type, cast

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorHandler
from lib.context import LLMFinishReason, extract_finish_reason
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient

__all__ = ["AgentNode", "AgentFinalOutput", "AgentStuckError"]

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

    def __init__(
        self,
        flow_id: str,
        flow_type: GLReportingEventContext,
        name: str,
        prompt: Prompt,
        inputs: Sequence[IOKey | RuntimeIOKey],
        internal_event_client: InternalEventsClient,
        conversation_history_key: RuntimeIOKey,
        compactor: ConversationCompactor | None = None,
        response_schema: Optional[Type[BaseAgentOutput]] = None,
        ui_history: Optional[UIHistory[UILogWriterAgentTools, UILogEventsAgent]] = None,
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

    _TRUNCATION_RECOVERY_MESSAGE = (
        "Your response was too long and got cut off. "
        "Be more concise and use smaller, incremental steps."
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
        )

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key.to_iokey(state)
        history = history_iokey.value_from_state(state) or []
        variables = get_vars_from_state(self._inputs, state)

        history = await maybe_compact_history(
            compactor=self._compactor, history=history, agent_name=self.name
        )

        truncation_retries: int = 0

        while True:
            try:
                completion: AIMessage = cast(
                    AIMessage,
                    await self._prompt.ainvoke(input={**variables, "history": history}),
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

                # Only emit reasoning if there are tool calls (i.e. omit for text-only messages)
                if completion.tool_calls:
                    self._emit_reasoning(completion)

                # Append new completion to existing history for replace-based reducer.
                # The reducer will replace this component's conversation history with
                # the complete list returned here.
                ui_updates = (
                    self._ui_history.pop_state_updates() if self._ui_history else {}
                )
                return merge_nested_dict(
                    ui_updates,
                    history_iokey.to_nested_dict(history + [completion]),
                )
            except APIStatusError as e:
                error_message = str(e)
                status_code = e.response.status_code
                model_error = ModelError(
                    error_type=self._error_handler.get_error_type(status_code),
                    status_code=status_code,
                    message=error_message,
                )

                await self._error_handler.handle_error(model_error)

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
