from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    AgentFinalOutput,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    create_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import DefaultUILogWriter, UIHistory

__all__ = ["FinalResponseNode"]


class FinalResponseNode:
    def __init__(
        self,
        *,
        component_name: str,
        name: str,
        output: Optional[IOKey],
        ui_history: UIHistory[DefaultUILogWriter, UILogEventsAgent],
    ):
        self._component_name = component_name
        self.name = name
        self._output = output
        self._ui_history = ui_history

    async def run(self, state: FlowState) -> dict:
        history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )

        if not history:
            raise ValueError(f"No messages found for {self._component_name}")

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"The last message of {self._component_name} is not of type AIMessage"
            )

        if not last_message.tool_calls:
            raise ValueError(
                f"No tool calls found in the last message of {self._component_name}"
            )

        if len(last_message.tool_calls) > 1:
            raise ValueError(
                f"Too many tool calls found in the last message of {self._component_name}"
            )

        final_response_call = last_message.tool_calls[0]

        # Check if no final response tool call found
        if final_response_call["name"] != AgentFinalOutput.tool_title:
            raise ValueError(
                f"Final response tool call not found in the conversation history of {self._component_name}"
            )

        parsed_response = AgentFinalOutput(**final_response_call["args"])
        self._ui_history.log.success(
            parsed_response.final_response,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

        updates: dict = {
            **self._ui_history.pop_state_updates(),
            FlowStateKeys.CONVERSATION_HISTORY: {
                self._component_name: [
                    ToolMessage(content="", tool_call_id=final_response_call["id"])
                ]
            },
        }

        if self._output:
            if self._output.subkeys is not None:
                updates[self._output.target] = create_nested_dict(
                    self._output.subkeys, parsed_response.final_response
                )
            else:
                updates[self._output.target] = parsed_response.final_response

        return updates
