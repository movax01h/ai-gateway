from typing import Annotated, ClassVar, Optional

from dependency_injector.wiring import Provide, inject
from langgraph.graph import StateGraph
from pydantic import Field

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import LocalPromptRegistry
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.nodes import (
    FetchNode,
    RequestNode,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
    UserLogWriter,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    register_component,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory

__all__ = ["HumanInputComponent"]


@register_component(decorators=[inject])
class HumanInputComponent(BaseComponent):
    """Component for requesting and fetching user input during workflow execution.

    This component enables human-in-the-loop interactions by:
    - Requesting user input with optional prompts
    - Interrupting workflow execution to wait for user response
    - Processing user responses (text input, approval/rejection decisions)
    - Routing responses to specified target components

    The component consists of two nodes:
    - RequestNode: Transitions workflow to INPUT_REQUIRED status and optionally displays prompts
    - FetchNode: Waits for user input via interrupt() and processes the response

    Supports different event types:
    - RESPONSE: Regular text input from user
    - APPROVE/REJECT: User approval decisions that are stored in context

    Available UI Log Events:
    - on_user_input_prompt: Agent's prompt/question to user (OPTIONAL)
    - on_user_response: User's response/input (OPTIONAL)
    """

    _sends_response_to_component: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="conversation_history",
        subkeys=[IOKeyTemplate.SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE],
    )

    _user_approval: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "approval"],
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        _sends_response_to_component,
        _user_approval,
    )

    supported_environments: ClassVar[tuple[str, ...]] = ("ide",)

    sends_response_to: str
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None

    prompt_registry: LocalPromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ]

    ui_log_events: list[UILogEventsHumanInput] = Field(default_factory=list)

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
        return f"{self.name}#request"

    @property
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

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        # Prepare prompt if provided
        prompt = None
        if self.prompt_id and self.prompt_version:
            prompt = self.prompt_registry.get_on_behalf(
                self.user,
                self.prompt_id,
                self.prompt_version,
            )

        ui_history = None
        if self.ui_log_events:
            ui_history = UIHistory(
                events=self.ui_log_events, writer_class=AgentLogWriter
            )

        # Create UI history for user responses using UserLogWriter
        user_response_ui_history = UIHistory(
            events=self.ui_log_events,
            writer_class=UserLogWriter,
        )

        # Create request node
        request_node = RequestNode(
            name=f"{self.name}#request",
            component_name=self.name,
            prompt=prompt,
            inputs=self.inputs,
            ui_history=ui_history,
        )

        # Create fetch node with approval output
        fetch_node = FetchNode(
            name=f"{self.name}#fetch",
            component_name=self.name,
            sends_response_to=self.sends_response_to,
            output=self._approval_output,
            ui_history=user_response_ui_history,
        )

        # Add nodes to graph
        graph.add_node(request_node.name, request_node.run)
        graph.add_node(fetch_node.name, fetch_node.run)

        # Add edges
        graph.add_edge(request_node.name, fetch_node.name)
        graph.add_conditional_edges(fetch_node.name, router.route)
