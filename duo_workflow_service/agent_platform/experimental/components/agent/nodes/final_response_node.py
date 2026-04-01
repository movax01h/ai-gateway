from typing import Optional, Type

from langchain_core.messages import AIMessage, ToolMessage

from ai_gateway.response_schemas.base import BaseAgentOutput
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState
from duo_workflow_service.agent_platform.experimental.state.base import (
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.experimental.ui_log import (
    DefaultUILogWriter,
    UIHistory,
)

__all__ = ["FinalResponseNode"]


class FinalResponseNode:
    """LangGraph node that handles the final response and writes the result to state.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary
    access.

    Both the conversation-history ``IOKey`` and the output ``IOKey`` are
    ``RuntimeIOKey`` instances that resolve the concrete key at runtime.  This
    supports both the common case (static key wrapped via ``RuntimeIOKey``) and
    the supervisor case where the key depends on runtime state such as the active
    subsession ID.

    Args:
        name: LangGraph node name.
        ui_history: UI log history writer for final-answer events.
        conversation_history_key: ``RuntimeIOKey`` that resolves the
            conversation-history ``IOKey`` at runtime.
        output_key: ``RuntimeIOKey`` that resolves the output ``IOKey`` at runtime.
    """

    def __init__(
        self,
        *,
        name: str,
        ui_history: UIHistory[DefaultUILogWriter, UILogEventsAgent],
        response_schema: Optional[Type[BaseAgentOutput]] = None,
        conversation_history_key: RuntimeIOKey,
        output_key: RuntimeIOKey,
    ):
        self.name = name
        self._ui_history = ui_history
        self._response_schema = response_schema
        self._conversation_history_key = conversation_history_key
        self._output_key = output_key

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key.to_iokey(state)
        output_iokey = self._output_key.to_iokey(state)

        last_message, history = self._get_last_ai_message(state, history_iokey)

        if self._response_schema is not None:
            final_response_text, updates = self._extract_structured_response(
                last_message, history, history_iokey, output_iokey
            )
        else:
            final_response_text, updates = self._extract_text_response(
                last_message, output_iokey
            )

        self._ui_history.log.success(
            final_response_text,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

        return {**self._ui_history.pop_state_updates(), **updates}

    def _get_last_ai_message(
        self, state: FlowState, history_iokey: IOKey
    ) -> tuple[AIMessage, list]:
        history = history_iokey.value_from_state(state) or []

        if not history:
            raise ValueError(
                f"No messages found for conversation history key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"The last message for conversation history key "
                f"{history_iokey.target}:{history_iokey.subkeys} "
                f"is not of type AIMessage"
            )

        return last_message, history

    def _extract_structured_response(
        self,
        last_message: AIMessage,
        history: list,
        history_iokey: IOKey,
        output_iokey: IOKey,
    ) -> tuple[str, dict]:
        if not last_message.tool_calls:
            raise ValueError(
                f"Response schema requires a tool call but got a text-only response "
                f"for conversation history key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

        if len(last_message.tool_calls) > 1:
            raise ValueError(
                f"Too many tool calls found in the last message for conversation "
                f"history key {history_iokey.target}:{history_iokey.subkeys}"
            )

        final_response_call = last_message.tool_calls[0]

        if self._response_schema is None:
            raise ValueError(
                "Response schema is required for structured response extraction"
            )
        if final_response_call["name"] != self._response_schema.tool_title:
            raise ValueError(
                f"Final response tool call not found in the conversation history "
                f"for key {history_iokey.target}:{history_iokey.subkeys}"
            )

        parsed_response = self._response_schema(**final_response_call["args"])

        # Append ToolMessage completion response to existing history for replace-based reducer.
        # The reducer will replace this component's conversation history with the complete list.
        updates: dict = {
            **history_iokey.to_nested_dict(
                history
                + [ToolMessage(content="", tool_call_id=final_response_call["id"])]
            ),
        }

        output_data = parsed_response.to_output()
        updates.update(output_iokey.to_nested_dict(output_data))

        return parsed_response.to_string_output(), updates

    def _extract_text_response(
        self, last_message: AIMessage, output_iokey: IOKey
    ) -> tuple[str, dict]:
        final_response_text = last_message.text

        return final_response_text, output_iokey.to_nested_dict(final_response_text)
