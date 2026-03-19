from typing import ClassVar, Literal, Optional

from dependency_injector.wiring import inject
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from pydantic import Field

from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponentBase,
    RoutingError,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes import (
    AgentNode,
    FinalResponseNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    ConversationHistoryKeyFactory,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.final_response_node import (
    OutputKeyFactory,
)
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    register_component,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.experimental.state.base import IOKey
from duo_workflow_service.agent_platform.experimental.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)

__all__ = ["SubagentComponent", "SUBAGENT_COMPONENT_MARKER"]

# Marker attribute name used by SupervisorAgentComponent to identify
# SubagentComponent instances without relying on class name string comparison,
# which is fragile under dependency-injector's @inject decorator wrapping.
SUBAGENT_COMPONENT_MARKER = "_is_subagent_component"


@register_component(decorators=[inject])
class SubagentComponent(AgentComponentBase):
    """AgentComponent variant for supervisor-managed subagents.

    Inherits from AgentComponentBase to avoid the inject-transforms-class-to-function
    issue that occurs when Pydantic models with Provide[] fields are decorated with @inject.

    Overrides the final_answer output key to be session-scoped, enabling
    multiple sessions of the same subagent type to coexist without
    overwriting each other's results.

    Before calling ``attach``, the owning supervisor must call
    ``bind_to_supervisor`` to inject the subsession-scoped key factories.
    This makes the supervisor–subagent relationship explicit at graph-build
    time and safe when multiple supervisors share the same flow — the subagent
    never needs to scan state to discover which supervisor owns it.

    The ``description`` field is surfaced to the supervisor's delegate_task
    tool so the LLM knows what each subagent specialises in.
    """

    # Marker checked by SupervisorAgentComponent to identify subagent instances
    # without relying on class-name string comparison, which is fragile under
    # dependency-injector's @inject decorator wrapping.
    _is_subagent_component: ClassVar[bool] = True

    description: str
    ui_log_events: list[UILogEventsAgent] = Field(default_factory=list)
    ui_role_as: Literal["agent", "tool"] = "agent"

    _final_answer_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[
            IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE,
            IOKeyTemplate.COMPONENT_NAME_TEMPLATE,
            IOKeyTemplate.SUBSESSION_ID_TEMPLATE,
            "final_answer",
        ],
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="conversation_history",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE],
        ),
        IOKeyTemplate(target="status"),
        _final_answer_key,
    )

    @property
    def outputs(self) -> tuple[IOKey, ...]:
        """Return an empty tuple.

        SubagentComponent's final_answer key requires three runtime substitutions
        (supervisor name, subagent name, subsession ID) that are only known at
        graph execution time.  The key is injected by the owning supervisor via
        ``bind_to_supervisor`` and resolved dynamically in ``FinalResponseNode``.
        Declaring it statically via the base ``outputs`` property would produce
        keys with unresolved template placeholders.
        """
        return ()

    # Set by bind_to_supervisor; None until bound.
    _conversation_history_key_factory: Optional[ConversationHistoryKeyFactory] = None
    _output_key_factory: Optional[OutputKeyFactory] = None

    def bind_to_supervisor(
        self,
        *,
        conversation_history_key_factory: ConversationHistoryKeyFactory,
        output_key_factory: OutputKeyFactory,
    ) -> None:
        """Bind this subagent to its owning supervisor.

        Must be called before ``attach``.  The supervisor passes subsession-scoped
        key factories so the subagent never needs to scan state to discover which
        supervisor owns it or what the active subsession ID is.

        Args:
            conversation_history_key_factory: Callable ``(state) -> IOKey`` that
                resolves the subsession-scoped conversation-history key at runtime.
            output_key_factory: Callable ``(state) -> IOKey`` that resolves the
                subsession-scoped final_answer key at runtime.
        """
        self._conversation_history_key_factory = conversation_history_key_factory
        self._output_key_factory = output_key_factory

    def _agent_node_router(self, state: FlowState) -> str:
        """Route based on the subagent's subsession-scoped conversation history.

        Relies on ``_conversation_history_key_factory`` set by ``bind_to_supervisor``.
        The bound factory resolves the correct subsession key from state at runtime,
        so no closure or dynamic router creation is needed.
        """
        history_iokey = self._conversation_history_key_factory(state)  # type: ignore[misc]
        history: list[BaseMessage] = history_iokey.value_from_state(state) or []
        if not history:
            raise RoutingError(
                f"Conversation history not found for subsession key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )
        last_message = history[-1]
        if not isinstance(last_message, AIMessage):
            raise RoutingError(
                f"Last message is not AIMessage for subsession key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )
        if not last_message.tool_calls:
            raise RoutingError(
                f"Tool calls not found for subsession key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )
        if any(
            tool_call["name"] == self._response_schema.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}#final_response"
        return f"{self.name}#tools"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        """Attach the subagent's ReAct subgraph to the parent graph.

        ``bind_to_supervisor`` must be called before this method.  Raises
        ``RuntimeError`` if the subagent has not been bound.

        Creates the standard 3-node ReAct loop (agent ↔ tools, final_response)
        with subsession-scoped conversation history.  The final_response node
        routes via the provided router (which in the supervisor case always
        routes to the supervisor's subagent_return node).
        """
        if (
            self._conversation_history_key_factory is None
            or self._output_key_factory is None
        ):
            raise RuntimeError(
                f"SubagentComponent '{self.name}' must be bound to a supervisor before "
                f"attach() is called. Call bind_to_supervisor() first."
            )

        tools = self.toolset.bindable + [self._response_schema]

        prompt = self.prompt_registry.get_on_behalf(
            self.user,
            self.prompt_id,
            self.prompt_version,
            tools=tools,  # type: ignore[arg-type]
            tool_choice="any",
            internal_event_extra={
                "agent_name": self.name,
                "workflow_id": self.flow_id,
                "workflow_type": self.flow_type.value,
            },
        )

        node_agent = AgentNode(
            name=self.__entry_hook__(),
            prompt=prompt,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            conversation_history_key_factory=self._conversation_history_key_factory,
            compactor=(
                create_conversation_compactor(
                    config=(
                        self.compaction
                        if isinstance(self.compaction, CompactionConfig)
                        else CompactionConfig()
                    ),
                    llm_model=prompt.model,
                )
                if self.compaction
                else None
            ),
            response_schema=self._response_schema,
        )
        node_tools = ToolNode(
            name=f"{self.name}#tools",
            toolset=self.toolset,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            ui_history=UIHistory(
                events=self.ui_log_events, writer_class=UILogWriterAgentTools
            ),
            conversation_history_key_factory=self._conversation_history_key_factory,
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}#final_response",
            ui_history=UIHistory(
                events=self.ui_log_events,
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsAgent, ui_role_as=self.ui_role_as
                ),
            ),
            conversation_history_key_factory=self._conversation_history_key_factory,
            output_key_factory=self._output_key_factory,
            response_schema=self._response_schema,
        )

        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)

        graph.add_conditional_edges(node_agent.name, self._agent_node_router)
        graph.add_edge(node_tools.name, node_agent.name)

        graph.add_conditional_edges(
            node_final_response.name,
            router.route,
        )
