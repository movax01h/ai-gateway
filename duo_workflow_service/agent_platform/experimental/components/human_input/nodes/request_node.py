from typing import Any, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, model_validator

from ai_gateway.prompts import Prompt
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    get_vars_from_state,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = ["RequestNode"]


class RequestNode(BaseModel):
    """Node that requests user input and transitions workflow to INPUT_REQUIRED status."""

    name: str
    component_name: str
    prompt: Optional[Prompt]
    inputs: list[IOKey]
    ui_history: Optional[UIHistory[AgentLogWriter, UILogEventsHumanInput]] = None

    @model_validator(mode="after")
    def validate_prompt_with_ui_history(self):
        """Ensure prompt and ui_history are either both present or both missing."""
        if bool(self.ui_history) != bool(self.prompt):
            raise ValueError(
                "prompt and ui_history must be either both present or both missing"
            )
        return self

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute the request node - emit user prompt and transition to INPUT_REQUIRED."""
        result: dict[str, Any] = {
            FlowStateKeys.STATUS: WorkflowStatusEnum.INPUT_REQUIRED.value
        }

        # Emit user_input_prompt event if enabled and prompt is available
        if self.prompt:
            # Get input variables from state for prompt rendering
            input_vars = get_vars_from_state(self.inputs, state)

            # Render the prompt with input variables
            prompt_tmpl = self.prompt.prompt_tpl
            prompt_messages = prompt_tmpl.invoke(input_vars).messages  # type: ignore[attr-defined]

            prompt_content = next(
                (
                    msg.content
                    for msg in prompt_messages
                    if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
                ),
                "",
            )

            # Use the UI history log writer if ui_history is available
            if self.ui_history:
                self.ui_history.log.success(
                    content=prompt_content,
                    event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                )

        ui_updates = self.ui_history.pop_state_updates() if self.ui_history else {}
        return {**result, **ui_updates}
