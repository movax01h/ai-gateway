from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowEvent,
    FlowEventType,
    FlowState,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = ["FetchNode"]


class FetchNode:
    """Node that fetches user input via interrupt() and creates HumanMessage.

    This node interrupts workflow execution to wait for user input, then processes
    the response based on the event type received.

    Supported event types:
        - APPROVE: User approves, stores approval value in output location.
                    No conversation history update.
        - REJECT: User rejects, stores rejection value in output location.
        - MODIFY: User requests modifications, adds message to conversation history.
                    Sets status to EXECUTION to continue workflow.
        - RESPONSE: User provides input/response, adds message to conversation history.
                    Sets status to EXECUTION to continue workflow.

    Args:
        name: The name of this node
        component_name: The name of the parent component
        output: IOKey for storing approval/rejection values
        conversation_history_key: IOKey for accessing conversation history (includes target agent)
        ui_history: UI logging history for tracking events
        status_key: IOKey for setting workflow status
    """

    def __init__(
        self,
        *,
        name: str,
        component_name: str,
        output: IOKey,
        conversation_history_key: IOKey,
        ui_history: UIHistory,
        status_key: IOKey,
    ):
        self.name = name
        self._component_name = component_name
        self._output = output
        self._conversation_history_key = conversation_history_key
        self._ui_history = ui_history
        self._status_key = status_key

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute the fetch node - interrupt for user input and create HumanMessage."""
        # Interrupt workflow to wait for user input
        event: FlowEvent = interrupt("Workflow interrupted; waiting for user input.")

        existing_history = self._conversation_history_key.value_from_state(state) or []
        # Handle different event types
        if event["event_type"] in (FlowEventType.APPROVE, FlowEventType.REJECT):
            # Handle approval/rejection events
            # Store the user decision in the specified output location
            approval_value = event["event_type"].value  # "approve" or "reject"
            result = merge_nested_dict(
                self._status_key.to_nested_dict(WorkflowStatusEnum.EXECUTION.value),
                self._output.to_nested_dict(approval_value),
            )

            # For REJECT events, also add HumanMessage to conversation history
            if event["event_type"] == FlowEventType.REJECT:
                rejection_message = "User rejected this action. Do not proceed and stop any tool execution in progress."

                self._ui_history.log.success(
                    content="Action rejected.",
                    event=UILogEventsHumanInput.ON_USER_RESPONSE,
                )

                # Append user rejection message to target component's history for replace-based reducer.
                # The reducer will replace the target component's conversation history with this list.
                human_message = HumanMessage(content=rejection_message)
                conversation_history_dict = (
                    self._conversation_history_key.to_nested_dict(
                        existing_history + [human_message]
                    )
                )
                result = merge_nested_dict(result, conversation_history_dict)

            result.update(self._ui_history.pop_state_updates())

            return result

        if event["event_type"] in (FlowEventType.MODIFY, FlowEventType.RESPONSE):
            if "message" not in event or not event["message"]:
                raise ValueError(
                    f"{event['event_type'].value.upper()} event must include a message with user feedback"
                )

            # Extract user message from event
            user_message = event["message"]

            self._ui_history.log.success(
                content=user_message,
                event=UILogEventsHumanInput.ON_USER_RESPONSE,
            )

            # Create HumanMessage for conversation history
            human_message = HumanMessage(content=user_message)

            result = merge_nested_dict(
                self._ui_history.pop_state_updates(),
                self._status_key.to_nested_dict(WorkflowStatusEnum.EXECUTION.value),
            )

            if event["event_type"] == FlowEventType.MODIFY:
                result = merge_nested_dict(
                    result, self._output.to_nested_dict("modify")
                )

            conversation_history_dict = self._conversation_history_key.to_nested_dict(
                existing_history + [human_message]
            )
            result = merge_nested_dict(result, conversation_history_dict)

            return result

        # For any other event type, raise error as this should not happen
        raise ValueError(
            f"Unknown event type: {event['event_type']}. Expected one of: {list(FlowEventType)}"
        )
