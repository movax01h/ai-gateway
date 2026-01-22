from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowEvent,
    FlowEventType,
    FlowState,
    FlowStateKeys,
    IOKey,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = ["FetchNode"]


class FetchNode(BaseModel):
    """Node that fetches user input via interrupt() and creates HumanMessage."""

    name: str
    component_name: str
    sends_response_to: str
    output: IOKey
    ui_history: UIHistory

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute the fetch node - interrupt for user input and create HumanMessage."""
        # Interrupt workflow to wait for user input
        event: FlowEvent = interrupt("Workflow interrupted; waiting for user input.")

        existing_history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self.sends_response_to, []
        )
        # Handle different event types
        if event["event_type"] in (FlowEventType.APPROVE, FlowEventType.REJECT):
            # Handle approval/rejection events
            # Store the user decision in the specified output location
            approval_value = event["event_type"].value  # "approve" or "reject"
            result = {
                FlowStateKeys.STATUS: WorkflowStatusEnum.EXECUTION.value,
                **self.output.to_nested_dict(approval_value),
            }

            # For REJECT events, also add HumanMessage to conversation history
            if event["event_type"] == FlowEventType.REJECT and "message" in event:
                # Log user response
                self.ui_history.log.success(
                    content=event["message"],
                    event=UILogEventsHumanInput.ON_USER_RESPONSE,
                )

                # Append user rejection message to target component's history for replace-based reducer.
                # The reducer will replace the target component's conversation history with this list.
                human_message = HumanMessage(content=event["message"])
                result[FlowStateKeys.CONVERSATION_HISTORY] = {
                    self.sends_response_to: existing_history + [human_message]
                }

            result.update(self.ui_history.pop_state_updates())

            return result

        if event["event_type"] == FlowEventType.RESPONSE and "message" in event:
            # Extract user message from event
            user_message = event["message"]

            # Log user response
            self.ui_history.log.success(
                content=user_message,
                event=UILogEventsHumanInput.ON_USER_RESPONSE,
            )

            # Create HumanMessage for conversation history
            human_message = HumanMessage(content=user_message)

            return {
                **self.ui_history.pop_state_updates(),
                FlowStateKeys.STATUS: WorkflowStatusEnum.EXECUTION.value,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    self.sends_response_to: existing_history + [human_message]
                },
            }

        # For any other event type, raise error as this should not happen
        raise ValueError(
            f"Unknown event type: {event['event_type']}. Expected one of: {list(FlowEventType)}"
        )
