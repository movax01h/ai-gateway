from typing import Annotated, Any, ClassVar, Literal, Self, override

from langgraph.graph import StateGraph
from pydantic import Field, model_validator

from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.components.human_input.nodes import (
    FetchNode,
    RequestNode,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
    agent_log_writer_class,
    user_log_writer_class,
)
from duo_workflow_service.agent_platform.v1.components.registry import (
    register_component_factory,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory

__all__ = [
    "HumanInputComponent",
    "ChatHumanInputComponent",
]


class HumanInputComponent(BaseComponent):
    """Base component for requesting and fetching user input during workflow execution.

    This component enables human-in-the-loop interactions by:
    - Requesting user input with a prompt message
    - Interrupting workflow execution to wait for user response
    - Processing user responses (text input, approval/rejection decisions)
    - Routing responses to specified target components

    The component consists of two nodes:
    - RequestNode: Transitions workflow to INPUT_REQUIRED status and displays a prompt
    - FetchNode: Waits for user input via interrupt() and processes the response

    Supports different event types:
    - RESPONSE: Regular text input from user
    - APPROVE/REJECT: User approval decisions that are stored in context

    UI Log Events (both enabled by default):
    - on_user_input_prompt: Agent's prompt/question to user, includes information about request type in message_sub_type
    - on_user_response: User's response/input
    """

    _sends_response_to_component: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="conversation_history",
        subkeys=[IOKeyTemplate.SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE],
    )

    _user_approval: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "approval"],
    )

    _status: ClassVar[IOKeyTemplate] = IOKeyTemplate(target="status")

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        _sends_response_to_component,
        _user_approval,
        _status,
    )

    required_ui_log_events: ClassVar[set[UILogEventsHumanInput]] = {
        UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
        UILogEventsHumanInput.ON_USER_RESPONSE,
    }

    supported_environments: ClassVar[tuple[str, ...]] = (
        "ambient",
        "chat",
        "chat-partial",
    )

    interaction_type: Literal["approval", "input"] = "approval"

    sends_response_to: str
    message_template: str

    ui_log_events: list[UILogEventsHumanInput] = Field(
        default=[
            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            UILogEventsHumanInput.ON_USER_RESPONSE,
        ]
    )

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    @model_validator(mode="after")
    def validate_required_ui_log_events(self) -> Self:
        """Validate that required UI log events are present.

        Both ON_USER_INPUT_PROMPT and ON_USER_RESPONSE are required:
        - ON_USER_INPUT_PROMPT: Needed to transport interaction type (approval / input) to UI
        - ON_USER_RESPONSE: Needed to display user's messages in the UI

        Raises:
            ValueError: If any required events are missing from ui_log_events.
        """

        missing = self.required_ui_log_events - set(self.ui_log_events)
        if missing:
            raise ValueError(
                f"ui_log_events must include: {', '.join(e.value for e in missing)}"
            )
        return self

    @override
    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return f"{self.name}#request"

    @property
    @override
    def outputs(self) -> tuple[IOKey, ...]:
        replacements = {
            IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name,
            IOKeyTemplate.SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE: self.sends_response_to,
        }
        return tuple(output.to_iokey(replacements) for output in self._outputs)

    @property
    def _approval_output(self) -> IOKey:
        return self._user_approval.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def _status_output(self) -> IOKey:
        return self._status.to_iokey({})

    @property
    def _conversation_history_input(self) -> IOKey:
        return self._sends_response_to_component.to_iokey(
            {
                IOKeyTemplate.SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE: self.sends_response_to
            }
        )

    @override
    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        ui_history = UIHistory(
            events=self.ui_log_events,
            writer_class=agent_log_writer_class(component_name=self.name),
        )

        # Create UI history for user responses using UserLogWriter
        user_response_ui_history = UIHistory(
            events=self.ui_log_events,
            writer_class=user_log_writer_class(component_name=self.name),
        )

        # Create request node
        request_node = RequestNode(
            name=f"{self.name}#request",
            component_name=self.name,
            message_template=self.message_template,
            inputs=self.inputs,
            request_type=self.interaction_type,
            ui_history=ui_history,
            status_key=self._status_output,
        )

        # Create fetch node with approval output
        fetch_node = FetchNode(
            name=f"{self.name}#fetch",
            component_name=self.name,
            output=self._approval_output,
            conversation_history_key=self._conversation_history_input,
            ui_history=user_response_ui_history,
            status_key=self._status_output,
        )

        # Add nodes to graph
        graph.add_node(request_node.name, request_node.run)
        graph.add_node(fetch_node.name, fetch_node.run)

        # Add edges
        graph.add_edge(request_node.name, fetch_node.name)
        graph.add_conditional_edges(fetch_node.name, router.route)


class ChatHumanInputComponent(HumanInputComponent):
    """HumanInputComponent variant for ``chat`` environment.

    Only enforces that ``on_user_response`` is present in ``ui_log_events``.
    Opting out of ``on_user_input_prompt`` is allowed because chat flows send
    that event via ``RequestNode`` at a different point, so including it here
    would produce duplicate messages in the Agentic Chat UI.
    """

    required_ui_log_events: ClassVar[set[UILogEventsHumanInput]] = {
        UILogEventsHumanInput.ON_USER_RESPONSE,
    }


@register_component_factory("HumanInputComponent")
def human_input_component_factory(**kwargs: Any) -> HumanInputComponent:
    """Dispatch to HumanInputComponent or ChatHumanInputComponent.

    Args:
        **kwargs: Component constructor arguments from the flow YAML, plus
            ``environment`` injected by the flow builder.

    Returns:
        An :class:`HumanInputComponent` or :class:`ChatHumanInputComponent`
        instance depending on the flow environment.
    """
    environment: str | None = kwargs.get("environment")

    if environment == "chat":
        return ChatHumanInputComponent(**kwargs)

    return HumanInputComponent(**kwargs)
