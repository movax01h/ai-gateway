from typing import Annotated, ClassVar, Optional

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import LocalPromptRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.nodes import (
    AgentFinalOutput,
    AgentNode,
    FinalResponseNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events import InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum

__all__ = ["AgentComponent"]


class RoutingError(Exception):
    """Exception raised when edge routers encounter unexpected conditions."""


class AgentComponent(BaseComponent):
    _final_answer_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "final_answer"],
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="conversation_history",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE],
        ),
        IOKeyTemplate(target="status"),
        _final_answer_key,
    )

    supported_environments: ClassVar[tuple[str, ...]] = ("platform",)

    prompt_id: str
    prompt_version: str
    toolset: Toolset

    prompt_registry: LocalPromptRegistry
    internal_event_client: InternalEventsClient

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    @inject
    def __init__(
        self,
        name: str,
        flow_id: str,
        flow_type: CategoryEnum,
        inputs: list[IOKey],
        prompt_id: str,
        prompt_version: str,
        toolset: Toolset,
        output: Optional[IOKey] = None,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        **kwargs,
    ):
        super().__init__(
            name=name,
            flow_id=flow_id,
            flow_type=flow_type,
            inputs=inputs,
            output=output,
            prompt_id=prompt_id,  # type: ignore[call-arg]
            prompt_version=prompt_version,  # type: ignore[call-arg]
            toolset=toolset,  # type: ignore[call-arg]
            prompt_registry=prompt_registry,  # type: ignore[call-arg]
            internal_event_client=internal_event_client,  # type: ignore[call-arg]
            **kwargs,
        )

    def _agent_node_router(self, state: FlowState) -> str:
        history: list[BaseMessage] = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self.name,
            [],
        )
        if not history:
            raise RoutingError(f"Conversation history not found for {self.name}")

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise RoutingError(
                f"Last message is not AIMessage for component {self.name}"
            )

        if not last_message.tool_calls:
            raise RoutingError(f"Tool calls not found for component {self.name}")

        if any(
            tool_call["name"] == AgentFinalOutput.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}#final_response"
        return f"{self.name}#tools"

    def __entry_hook__(self) -> Annotated[str, "Entry node name"]:
        return f"{self.name}#agent"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        tools = self.toolset.bindable + [AgentFinalOutput]
        tool_choice = "any"  # make sure the LLM always uses a tool to respond.

        prompt = self.prompt_registry.get(
            self.prompt_id, self.prompt_version, tools=tools, tool_choice=tool_choice  # type: ignore[arg-type]
        )

        node_agent = AgentNode(
            name=self.__entry_hook__(),
            component_name=self.name,
            prompt=prompt,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )
        node_tools = ToolNode(
            name=f"{self.name}#tools",
            component_name=self.name,
            toolset=self.toolset,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}#final_response",
            component_name=self.name,
            output=self._final_answer_key.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
        )

        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)

        graph.add_conditional_edges(
            node_agent.name,
            self._agent_node_router,
        )
        graph.add_edge(node_tools.name, node_agent.name)

        graph.add_conditional_edges(
            node_final_response.name,
            router.route,
        )
