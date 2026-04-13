from typing import Any, Literal

from ai_gateway.prompts import jinja2_formatter
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    get_vars_from_state,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = ["RequestNode"]


class RequestNode:
    """Node that requests user input and transitions workflow to INPUT_REQUIRED status.

    This node marks the workflow as requiring input from the user and displays
    a prompt to guide the user's response.

    Request types:
        - approval: Requests user approval/rejection (APPROVE/REJECT events)
        - input: Requests freeform text input (RESPONSE/MODIFY events)

    Args:
        name: The name of this node
        component_name: The name of the parent component
        message_template: Jinja2 template string to display to the user
        inputs: List of IOKeys for extracting variables from state for message template rendering
        ui_history: UI logging history for tracking events
        request_type: Type of request - "approval" or "input"
        status_key: IOKey for setting workflow status
    """

    def __init__(
        self,
        *,
        name: str,
        component_name: str,
        message_template: str,
        inputs: list[IOKey],
        ui_history: UIHistory[AgentLogWriter, UILogEventsHumanInput],
        request_type: Literal["approval", "input"] = "approval",
        status_key: IOKey,
    ):
        self.name = name
        self._component_name = component_name
        self._message_template = message_template
        self._inputs = inputs
        self._ui_history = ui_history
        self._request_type = request_type
        self._status_key = status_key

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute the request node - emit user prompt and transition to INPUT_REQUIRED status."""
        result: dict[str, Any] = self._status_key.to_nested_dict(
            WorkflowStatusEnum.INPUT_REQUIRED.value
        )

        # Get input variables from state for message template rendering
        input_vars = get_vars_from_state(self._inputs, state)

        # Render the message template with input variables
        prompt_content = jinja2_formatter(self._message_template, **input_vars)

        # Log the prompt to UI
        self._ui_history.log.success(
            content=prompt_content,
            event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            request_type=self._request_type,
        )

        ui_updates = self._ui_history.pop_state_updates()
        return {**result, **ui_updates}
