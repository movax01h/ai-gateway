from typing import Annotated, Any, ClassVar, Self

from dependency_injector.wiring import inject
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from pydantic import Field, PrivateAttr, field_validator, model_validator

from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponentBase,
    RoutingError,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes import (
    AgentFinalOutput,
    AgentNode,
    FinalResponseNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    ConversationHistoryKeyFactory,
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
from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    ManagedAgentConfig,
    build_delegate_task_model,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.nodes import (
    SUBSESSION_KEY_SEPARATOR,
    DelegationNode,
    SubagentReturnNode,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.subagent_component import (
    SUBAGENT_COMPONENT_MARKER,
    SubagentComponent,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.ui_log import (
    UILogEventsSupervisor,
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

__all__ = ["SupervisorAgentComponent"]


class _SubagentRouter:
    """Internal router for subagent components within a supervisor.

    Routes subagent final_response to the subagent_return node
    instead of to an external router.

    ``attach`` is intentionally a no-op: subagents are attached directly by
    ``SupervisorAgentComponent.attach``, which wires the subagent_return edge
    itself.  ``_SubagentRouter`` is passed only to satisfy the ``RouterProtocol``
    interface required by ``SubagentComponent.attach``; ``route`` is the only
    method that LangGraph's conditional edge machinery will ever call on it.
    """

    def __init__(self, supervisor_name: str):
        self._supervisor_name = supervisor_name

    def attach(self, graph: StateGraph) -> None:
        """No-op: edge wiring for subagents is handled by SupervisorAgentComponent.attach."""

    def route(self, _state: FlowState) -> Annotated[str, "Next node"]:
        """Always route to the subagent_return node."""
        return f"{self._supervisor_name}#subagent_return"


@register_component(decorators=[inject])
class SupervisorAgentComponent(AgentComponentBase):
    """Supervisor component that orchestrates subagents via delegate_task tool.

    The SupervisorAgentComponent acts as a container that manages dedicated
    ReAct subgraphs for each of its managed subagents.

    Key capabilities:
    1. Explicit delegation via delegate_task tool
    2. Contextual handoffs via prompt injection as HumanMessage
    3. Session management (new sessions or resume existing ones by ID)
    4. Sub-agent results injected as ToolMessage responses to delegate_task calls

    Graph topology:
        supervisor#agent ↔ supervisor#tools (regular tools)
        supervisor#agent → supervisor#delegation (delegate_task)
        supervisor#agent → supervisor#final_response (final_response_tool)
        supervisor#delegation → <subagent>#agent (routes by subagent_type)
        <subagent>#final_response → supervisor#subagent_return
        supervisor#subagent_return → supervisor#agent (loop)
    """

    # Orchestration metadata written by delegation/return nodes.
    # All are optional=True so value_from_state returns None (instead of raising
    # KeyError) before the supervisor's context sub-dict is first populated.
    _delegation_count_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "delegation_count"],
        optional=True,
    )
    _active_subsession_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "active_subsession"],
        optional=True,
    )
    _active_subagent_type_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "active_subagent_type"],
        optional=True,
    )
    _max_subsession_id_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "max_subsession_id"],
        optional=True,
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        # Supervisor's own conversation history and final answer
        IOKeyTemplate(
            target="conversation_history",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE],
        ),
        IOKeyTemplate(target="status"),
        AgentComponentBase._final_answer_key,
        # Orchestration metadata
        _delegation_count_key,
        _active_subsession_key,
        _active_subagent_type_key,
        _max_subsession_id_key,
    )

    supported_environments: ClassVar[tuple[str, ...]] = ("ambient",)

    max_delegations: int

    ui_log_events: list[UILogEventsSupervisor] = Field(default_factory=list)
    ui_role_as: str = "agent"

    subagent_components: dict[str, Any] = Field(
        description="Resolved subagent component instances, injected by the flow builder at construction time.",
        exclude=True,
    )

    # Built once in the model_validator(mode="after") and reused in
    # _agent_node_router / attach.  Must NOT be ClassVar — it is instance-specific
    # because every SupervisorAgentComponent manages a different subagent set and
    # therefore owns a distinct DelegateTask subclass.  PrivateAttr keeps it out of
    # Pydantic's field schema while allowing normal instance assignment.
    _delegate_task_cls: type = PrivateAttr()

    @field_validator("max_delegations")
    @classmethod
    def validate_max_delegations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_delegations must be at least 1.")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_and_consume_managed_agents(cls, data: Any) -> Any:
        """Validate managed_agents against subagent_components and consume it.

        The YAML config declares `managed_agents: [...]` as the human-facing
        list of subagent names. The flow builder resolves these to actual
        component instances and passes them as `subagent_components`.

        This validator:
        1. Validates that every name in managed_agents has a matching entry
            in subagent_components
        2. Validates that every subagent is of type SubagentComponent
        3. Removes managed_agents from the data (not stored on the model)

        managed_agents is consumed here and never stored — the runtime model
        uses subagent_components.keys() as the source of truth.
        """
        if isinstance(data, dict):
            managed_agents = data.pop("managed_agents", None)
            sub_agents = data.get("subagent_components", {})

            if managed_agents is not None:
                if not managed_agents:
                    raise ValueError(
                        "SupervisorAgentComponent requires at least one managed agent."
                    )
                for agent_name in managed_agents:
                    if sub_agents and agent_name not in sub_agents:
                        raise ValueError(
                            f"Managed agent '{agent_name}' not found in subagent_components. "
                            f"Available: {list(sub_agents.keys())}"
                        )

            if sub_agents:
                for name, component in sub_agents.items():
                    if getattr(component, SUBAGENT_COMPONENT_MARKER, False) is not True:
                        raise ValueError(
                            f"Subagent '{name}' is of type '{type(component).__name__}'. "
                            f"Managed agents must be of type {SubagentComponent.__name__}."
                        )

        return data

    @model_validator(mode="after")
    def validate_and_build_delegate_task_cls(self) -> Self:
        """Validate subagent_components and build the dynamic DelegateTask model."""
        if not self.subagent_components:
            raise ValueError(
                "SupervisorAgentComponent requires at least one subagent component."
            )
        self._delegate_task_cls = build_delegate_task_model(self.managed_agents_config)
        return self

    @property
    def managed_agent_names(self) -> list[str]:
        """Derive managed agent names from subagent_components keys."""
        return list(self.subagent_components.keys())

    @property
    def managed_agents_config(self) -> list[ManagedAgentConfig]:
        """Derive name+description config for each managed subagent."""
        return [
            ManagedAgentConfig(name=name, description=component.description)
            for name, component in self.subagent_components.items()
        ]

    @property
    def _resolved_delegation_count_key(self) -> IOKey:
        """Resolve the delegation_count ``IOKey`` for this supervisor instance."""
        return self._delegation_count_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def _resolved_active_subsession_key(self) -> IOKey:
        """Resolve the active_subsession ``IOKey`` for this supervisor instance."""
        return self._active_subsession_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def _resolved_active_subagent_type_key(self) -> IOKey:
        """Resolve the active_subagent_type ``IOKey`` for this supervisor instance."""
        return self._active_subagent_type_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def _resolved_max_subsession_id_key(self) -> IOKey:
        """Resolve the max_subsession_id ``IOKey`` for this supervisor instance."""
        return self._max_subsession_id_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def outputs(self) -> tuple[IOKey, ...]:
        replacements = {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        return tuple(output.to_iokey(replacements) for output in self._outputs)

    def _delegation_router(self, state: FlowState) -> str:
        """Router for the delegation node conditional edge.

        Reads active_subagent_type written by DelegationNode.run() and routes to the appropriate subagent entry node.
        Falls back to the supervisor's own agent node when delegation failed (error ToolMessage was injected instead of
        setting active_subagent_type).
        """
        active_subagent_type = self._resolved_active_subagent_type_key.value_from_state(
            state
        )

        if active_subagent_type and active_subagent_type in self.managed_agent_names:
            return f"{active_subagent_type}#agent"

        # Delegation failed — route back to supervisor so LLM can react
        return f"{self.name}#agent"

    def _agent_node_router(self, state: FlowState) -> str:
        """3-way router for the supervisor's agent node.

        Routes based on the tool call type:
        - delegate_task → supervisor#delegation
        - final_response_tool → supervisor#final_response
        - other tools → supervisor#tools
        """
        history_iokey = self._conversation_history_key(state)
        history: list[BaseMessage] = history_iokey.value_from_state(state) or []

        if not history:
            raise RoutingError(f"Conversation history not found for {self.name}")

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise RoutingError(
                f"Last message is not AIMessage for component {self.name}"
            )

        if not last_message.tool_calls:
            raise RoutingError(f"Tool calls not found for component {self.name}")

        # Check for delegate_task
        delegate_title: str = self._delegate_task_cls.tool_title  # type: ignore[attr-defined]
        if any(
            tool_call["name"] == delegate_title for tool_call in last_message.tool_calls
        ):
            return f"{self.name}#delegation"

        # Check for final_response_tool
        if any(
            tool_call["name"] == AgentFinalOutput.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}#final_response"

        # Regular tools
        return f"{self.name}#tools"

    def _subsession_history_key_factory(
        self, subagent_type: str, subsession_id: int
    ) -> IOKey:
        """Build the subsession-scoped conversation-history IOKey.

        Encapsulates the key naming convention so nodes never need to know it.
        Matches the ``SubsessionHistoryKeyFactory`` signature.
        """
        key = f"{self.name}{SUBSESSION_KEY_SEPARATOR}{subagent_type}{SUBSESSION_KEY_SEPARATOR}{subsession_id}"
        return IOKey(
            target="conversation_history",
            subkeys=[key],
            optional=True,
        )

    def _subsession_history_key_factory_for(
        self, subagent_name: str
    ) -> ConversationHistoryKeyFactory:
        """Return a ``ConversationHistoryKeyFactory`` scoped to a specific subagent.

        The returned factory reads the active subsession ID from state at runtime
        and delegates to ``_subsession_history_key_factory``.  Passed into
        ``SubagentComponent.bind_to_supervisor`` so the subagent never needs to
        scan state to discover its supervisor or subsession.

        Args:
            subagent_name: The name of the subagent this factory is scoped to.
        """
        return lambda state: self._subsession_history_key_factory(
            subagent_name, self._resolved_active_subsession_key.value_from_state(state)
        )

    def _subagent_final_answer_key_factory_for(
        self, subagent_name: str
    ) -> ConversationHistoryKeyFactory:
        """Return a factory that resolves the final_answer IOKey for a specific subagent.

        The returned factory reads the active subsession ID from state at runtime
        and builds the IOKey for the given subagent_name.  Passed into
        ``SubagentComponent.bind_to_supervisor`` so the subagent knows where to
        write its final answer.

        Args:
            subagent_name: The name of the subagent this factory is scoped to.
        """
        return lambda state: SubagentComponent._final_answer_key.to_iokey(
            {
                IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE: self.name,
                IOKeyTemplate.COMPONENT_NAME_TEMPLATE: subagent_name,
                IOKeyTemplate.SUBSESSION_ID_TEMPLATE: str(
                    self._resolved_active_subsession_key.value_from_state(state)
                ),
            }
        )

    def _active_subagent_final_answer_key_factory(self, state: FlowState) -> IOKey:
        """Resolve the final_answer IOKey for whichever subagent is currently active.

        Reads active_subagent_type from state, then delegates to the per-subagent factory.  Used by SubagentReturnNode
        to read the result from whichever subagent just completed.
        """
        active_type = self._resolved_active_subagent_type_key.value_from_state(state)

        if not active_type or active_type not in self.subagent_components:
            raise ValueError(
                f"Cannot resolve final_answer key: no active subagent type or "
                f"'{active_type}' not in managed agents {self.managed_agent_names}"
            )

        return self._subagent_final_answer_key_factory_for(active_type)(state)

    def _build_prompt(self, supervisor_tools: list) -> Any:
        """Build the supervisor prompt with the given tool list."""
        return self.prompt_registry.get_on_behalf(
            self.user,
            self.prompt_id,
            self.prompt_version,
            tools=supervisor_tools,
            tool_choice="any",
            internal_event_extra={
                "agent_name": self.name,
                "workflow_id": self.flow_id,
                "workflow_type": self.flow_type.value,
            },
        )

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        """Attach the supervisor and all subagent subgraphs to the graph.

        Builds the complete graph topology:
        - Supervisor's 3-node ReAct loop (agent ↔ tools, final_response)
        - Delegation node with routing to subagent subgraphs
        - One 3-node ReAct subgraph per managed subagent
        - Subagent return node that injects results back to supervisor
        """
        # Supervisor tools = user-specified tools + delegate_task + final_response_tool
        supervisor_tools = self.toolset.bindable + [
            self._delegate_task_cls,
            AgentFinalOutput,
        ]
        prompt = self._build_prompt(supervisor_tools)

        output_key = self._final_answer_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

        node_agent = AgentNode(
            name=self.__entry_hook__(),
            conversation_history_key_factory=self._conversation_history_key,
            prompt=prompt,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
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
        # Filter supervisor events to only those compatible with UILogEventsAgent
        # (ToolNode uses UILogWriterAgentTools which requires UILogEventsAgent)
        tool_events = [
            UILogEventsAgent[e.name]
            for e in self.ui_log_events
            if e.name in UILogEventsAgent.__members__
        ]
        node_tools = ToolNode(
            name=f"{self.name}#tools",
            conversation_history_key_factory=self._conversation_history_key,
            toolset=self.toolset,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            ui_history=UIHistory(
                events=tool_events, writer_class=UILogWriterAgentTools
            ),
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}#final_response",
            conversation_history_key_factory=self._conversation_history_key,
            output_key_factory=lambda _: output_key,
            ui_history=UIHistory(
                events=self.ui_log_events,  # type: ignore[arg-type]
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsSupervisor,
                    ui_role_as=self.ui_role_as,  # type: ignore[arg-type]
                ),
            ),
            response_schema=self._response_schema,
        )

        # --- Delegation node ---
        node_delegation = DelegationNode(
            name=f"{self.name}#delegation",
            max_delegations=self.max_delegations,
            delegate_task_cls=self._delegate_task_cls,
            delegation_count_key=self._resolved_delegation_count_key,
            active_subsession_key=self._resolved_active_subsession_key,
            active_subagent_type_key=self._resolved_active_subagent_type_key,
            max_subsession_id_key=self._resolved_max_subsession_id_key,
            supervisor_history_key_factory=self._conversation_history_key,
            subsession_history_key_factory=self._subsession_history_key_factory,
        )

        # --- Subagent return node ---
        node_subagent_return = SubagentReturnNode(
            name=f"{self.name}#subagent_return",
            delegate_task_cls=self._delegate_task_cls,
            active_subsession_key=self._resolved_active_subsession_key,
            active_subagent_type_key=self._resolved_active_subagent_type_key,
            final_answer_key_factory=self._active_subagent_final_answer_key_factory,
            supervisor_history_key_factory=self._conversation_history_key,
        )

        # --- Add supervisor nodes to graph ---
        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)
        graph.add_node(node_delegation.name, node_delegation.run)
        graph.add_node(node_subagent_return.name, node_subagent_return.run)

        # --- Supervisor edges ---
        # 3-way conditional routing from agent node
        graph.add_conditional_edges(
            node_agent.name,
            self._agent_node_router,
        )

        # Tools → back to agent
        graph.add_edge(node_tools.name, node_agent.name)

        # Final response → external router
        graph.add_conditional_edges(
            node_final_response.name,
            router.route,
        )

        # Delegation → conditional routing to subagent subgraphs
        graph.add_conditional_edges(
            node_delegation.name,
            self._delegation_router,
        )

        # Subagent return → back to supervisor agent
        graph.add_edge(node_subagent_return.name, node_agent.name)

        # --- Attach subagent subgraphs ---
        subagent_router = _SubagentRouter(supervisor_name=self.name)

        for agent_name, subagent in self.subagent_components.items():
            subagent.bind_to_supervisor(
                conversation_history_key_factory=self._subsession_history_key_factory_for(
                    agent_name
                ),
                output_key_factory=self._subagent_final_answer_key_factory_for(
                    agent_name
                ),
            )
            subagent.attach(graph, subagent_router)
