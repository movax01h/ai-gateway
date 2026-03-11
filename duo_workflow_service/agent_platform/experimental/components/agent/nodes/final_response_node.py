from typing import Callable, Optional

from langchain_core.messages import AIMessage, ToolMessage

from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
    ConversationHistoryKeyFactory,
)
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
)
from duo_workflow_service.agent_platform.experimental.ui_log import (
    DefaultUILogWriter,
    UIHistory,
)

__all__ = ["FinalResponseNode"]

OutputKeyFactory = Callable[[FlowState], Optional[IOKey]]


class FinalResponseNode:
    """LangGraph node that handles the final response tool call and writes the result to state.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary
    access.

    Both the conversation-history ``IOKey`` and the output ``IOKey`` are resolved
    dynamically at runtime via factory callables.  This supports both the common
    case (static key wrapped in a lambda by the caller) and the supervisor case
    where the key is only known at runtime.

    Args:
        name: LangGraph node name.
        ui_history: UI log history writer for final-answer events.
        conversation_history_key_factory: Callable ``(state) -> IOKey`` that
            resolves the conversation-history ``IOKey`` at runtime.
        output_key_factory: Callable ``(state) -> Optional[IOKey]`` that resolves
            the output ``IOKey`` at runtime.  Pass ``lambda _: None`` when no
            persistent output is required.
    """

    def __init__(
        self,
        *,
        name: str,
        ui_history: UIHistory[DefaultUILogWriter, UILogEventsAgent],
        conversation_history_key_factory: ConversationHistoryKeyFactory,
        output_key_factory: OutputKeyFactory,
    ):
        self.name = name
        self._ui_history = ui_history
        self._conversation_history_key_factory = conversation_history_key_factory
        self._output_key_factory = output_key_factory

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key_factory(state)
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

        if not last_message.tool_calls:
            raise ValueError(
                f"No tool calls found in the last message for conversation history key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

        if len(last_message.tool_calls) > 1:
            raise ValueError(
                f"Too many tool calls found in the last message for conversation history key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

        final_response_call = last_message.tool_calls[0]

        if final_response_call["name"] != AgentFinalOutput.tool_title:
            raise ValueError(
                f"Final response tool call not found in the conversation history for key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

        parsed_response = AgentFinalOutput(**final_response_call["args"])
        self._ui_history.log.success(
            parsed_response.final_response,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

        # Append ToolMessage completion response to existing history for replace-based reducer.
        # The reducer will replace this component's conversation history with the complete list.
        updates: dict = {
            **self._ui_history.pop_state_updates(),
            **history_iokey.to_nested_dict(
                history
                + [ToolMessage(content="", tool_call_id=final_response_call["id"])]
            ),
        }

        output = self._output_key_factory(state)
        if output:
            updates = {
                **updates,
                **output.to_nested_dict(parsed_response.final_response),
            }

        return updates
