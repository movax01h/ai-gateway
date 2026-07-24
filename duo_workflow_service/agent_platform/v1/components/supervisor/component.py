from typing import Annotated, Any, ClassVar, Optional, Self, TypedDict, override

from dependency_injector.wiring import inject
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import Field, PrivateAttr, field_validator, model_validator

from duo_workflow_service.agent_platform.constants import NODE_ROLE_SEPARATOR
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponentBase,
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes import (
    AgentNode,
    FinalResponseNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    agent_tools_ui_log_writer_class,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.delegate_task import (
    SubagentDescriptor,
    build_delegate_task_model,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.nodes import (
    SUBSESSION_KEY_SEPARATOR,
    DelegationNode,
    SubagentReturnNode,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.v1.state.base import (
    IOKey,
    NoneIOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)
from duo_workflow_service.entities.state import MessageTypeEnum

__all__ = ["SubagentConfig", "SupervisorAgentComponent", "extract_subagent_names"]


class SubagentConfig(TypedDict):
    """Descriptor for a subagent entry in the YAML flow configuration.

    Represents a single entry in the ``subagents`` list of a
    ``SupervisorAgentComponent`` config block.  Only ``name`` is required.
    In case optional fields are introduced in the future, the ``NotRequired``
    type qualifier can be applied to those fields.

    Example YAML entry::

        subagents:
            - name: "developer"
            - name: "tester"

    Attributes:
        name: The agent name.  Must match a component defined in the same
            flow config.  **Required.**
    """

    name: str


def extract_subagent_names(subagents: list[SubagentConfig]) -> list[str]:
    """Extract agent names from a subagents list of configs.

    Each entry in ``subagents`` is a :class:`SubagentConfig`
    that must contain at least a ``"name"`` key.

    Args:
        subagents: List of :class:`SubagentConfig` entries from the YAML config.

    Returns:
        Ordered list of agent name strings.

    Raises:
        ValueError: If any entry is not a dict or is missing the ``"name"`` key,
            or if any name appears more than once in the list.
    """
    names: list[str] = []
    for entry in subagents:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                f"Each subagents entry must be a dict with a 'name' key, got: {entry!r}"
            )
        name = entry["name"]
        if name in names:
            raise ValueError(
                f"Duplicate subagent name '{name}' found in subagents list. "
                f"Each subagent name must be unique."
            )
        names.append(name)

    return names


class _SubagentRouter:
    """Internal router for subagent components within a supervisor.

    Routes subagent final_response to the subagent_return node
    instead of to an external router.

    ``attach`` is intentionally a no-op: subagents are attached directly by
    ``SupervisorAgentComponent.attach``, which wires the subagent_return edge
    itself.  ``_SubagentRouter`` is passed only to satisfy the ``RouterProtocol``
    interface required by ``AgentComponent.attach``; ``route`` is the only
    method that LangGraph's conditional edge machinery will ever call on it.
    """

    def __init__(self, supervisor_name: str):
        self._supervisor_name = supervisor_name

    def attach(self, graph: StateGraph) -> None:
        """No-op: edge wiring for subagents is handled by SupervisorAgentComponent.attach."""

    def route(self, _state: FlowState) -> Annotated[str, "Next node"]:
        """Always route to the subagent_return node."""
        return f"{self._supervisor_name}{NODE_ROLE_SEPARATOR}subagent_return"


@inject
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
        supervisor#delegation → <subagent>#agent (routes by subagent_name)
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
    _active_subagent_name_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "active_subagent_name"],
        optional=True,
    )
    _max_subsession_id_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "max_subsession_id"],
        optional=True,
    )
    _subsession_final_answer_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[
            IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE,
            IOKeyTemplate.COMPONENT_NAME_TEMPLATE,
            IOKeyTemplate.SUBSESSION_ID_TEMPLATE,
            "final_answer",
        ],
    )
    _subsession_tool_approval_decision_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[
            IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE,
            IOKeyTemplate.COMPONENT_NAME_TEMPLATE,
            IOKeyTemplate.SUBSESSION_ID_TEMPLATE,
            "tool_approval_decision",
        ],
        optional=True,
    )
    _subsession_cycle_count_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[
            IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE,
            IOKeyTemplate.COMPONENT_NAME_TEMPLATE,
            IOKeyTemplate.SUBSESSION_ID_TEMPLATE,
            "cycle_count",
        ],
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
        _active_subagent_name_key,
        _max_subsession_id_key,
    )

    supported_environments: ClassVar[tuple[str, ...]] = ("ambient",)

    max_delegations: Optional[int] = None

    ui_log_events: list[UILogEventsSupervisor] = Field(default_factory=list)
    ui_role_as: str = "agent"

    @override
    def _agent_node_invoke_config(self) -> RunnableConfig:
        """Return TAG_NOSTREAM config unless both LLM output event types are declared.

        Both ON_AGENT_FINAL_ANSWER and ON_AGENT_REASONING must be present because AgentNode tokens may become either —
        they are indistinguishable at chunk time.
        """
        if (
            UILogEventsSupervisor.ON_AGENT_FINAL_ANSWER in self.ui_log_events
            and UILogEventsSupervisor.ON_AGENT_REASONING in self.ui_log_events
        ):
            return self.STREAMING_ENABLED_CONFIG
        return self.STREAMING_DISABLED_CONFIG

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
    def validate_max_delegations(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("max_delegations must be at least 1.")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_and_consume_subagents(cls, data: Any) -> Any:
        """Validate subagents, select subagents from the pool, and consume subagents.

        The YAML config declares ``subagents: [...]`` as the human-facing list of
        :class:`SubagentConfig` entries.  Each entry must contain at least a
        ``"name"`` key.  The factory passes the full pool of already-built
        components as ``subagent_components``.

        Example YAML::

            subagents:
                - name: "developer"
                - name: "tester"

        This validator centralises all subagent-selection logic.
        It validates that ``subagents`` is non-empty (raising ``ValueError`` for
        an empty or absent list), extracts agent names from each
        :class:`SubagentConfig` entry, selects only the named agents from the
        ``subagent_components`` pool (raising ``ValueError`` for any missing
        name), validates that every selected subagent exposes
        ``bind_to_supervisor``, replaces ``subagent_components`` with the
        filtered dict, and removes ``subagents`` so the runtime model uses
        ``subagent_components.keys()`` as the source of truth.
        """
        if isinstance(data, dict):
            subagents: list[SubagentConfig] = data.pop("subagents", [])
            all_components = data.pop("subagent_components", {})

            if not subagents:
                raise ValueError(
                    "SupervisorAgentComponent requires at least one managed agent."
                )

            selected_components: dict[str, Any] = {}
            for agent_name in extract_subagent_names(subagents):
                if agent_name not in all_components:
                    raise ValueError(
                        f"Managed agent '{agent_name}' not found in subagent_components. "
                        f"Available: {list(all_components.keys())}"
                    )

                component = all_components[agent_name]

                if not hasattr(component, "bind_to_supervisor") or not callable(
                    getattr(component, "bind_to_supervisor")
                ):
                    raise ValueError(
                        f"Managed agent '{agent_name}' of type '{type(component).__name__}' "
                        f"does not have a bind_to_supervisor method. "
                        f"Managed agents must have a bind_to_supervisor method."
                    )

                selected_components[agent_name] = component

            # Replace the full pool with only the selected subagents
            data["subagent_components"] = selected_components

        return data

    @model_validator(mode="after")
    def validate_and_build_delegate_task_cls(self) -> Self:
        """Validate subagent_components and build the dynamic DelegateTask model."""
        if not self.subagent_components:
            raise ValueError(
                "SupervisorAgentComponent requires at least one subagent component."
            )
        self._delegate_task_cls = build_delegate_task_model(self.subagents_config)
        return self

    @property
    def managed_agent_names(self) -> list[str]:
        """Derive managed agent names from subagent_components keys."""
        return list(self.subagent_components.keys())

    @property
    def subagents_config(self) -> list[SubagentDescriptor]:
        """Derive name+description config for each managed subagent."""
        return [
            SubagentDescriptor(name=name, description=component.description)
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
    def _resolved_active_subagent_name_key(self) -> IOKey:
        """Resolve the active_subagent_name ``IOKey`` for this supervisor instance."""
        return self._active_subagent_name_key.to_iokey(
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

        Reads active_subagent_name written by DelegationNode.run() and routes to the appropriate subagent entry node.
        Falls back to the supervisor's own agent node when delegation failed (error ToolMessage was injected instead of
        setting active_subagent_name).
        """
        active_subagent_name = self._resolved_active_subagent_name_key.value_from_state(
            state
        )

        if active_subagent_name and active_subagent_name in self.managed_agent_names:
            return f"{active_subagent_name}{NODE_ROLE_SEPARATOR}agent"

        # Delegation failed — route back to supervisor so LLM can react
        return f"{self.name}{NODE_ROLE_SEPARATOR}agent"

    def _agent_node_router(self, state: FlowState) -> str:
        """Router for the supervisor's agent node.

        Routes based on the last message:
        - text-only (no tool calls) → supervisor#final_response (implicit final answer)
        - delegate_task → supervisor#delegation
        - schema tool call → supervisor#final_response
        - other tools → supervisor#tools
        """
        history_iokey = self._default_conversation_history_key.to_iokey(state)
        history: list[BaseMessage] = history_iokey.value_from_state(state) or []

        if not history:
            raise RoutingError(f"Conversation history not found for {self.name}")

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise RoutingError(
                f"Last message is not AIMessage for component {self.name}"
            )

        if not last_message.tool_calls:
            if self._response_schema is not None:
                raise RoutingError(
                    f"Schema mode requires a tool call but got a text-only response "
                    f"for component {self.name}"
                )
            return f"{self.name}{NODE_ROLE_SEPARATOR}final_response"

        # Check for delegate_task
        delegate_title: str = self._delegate_task_cls.tool_title  # type: ignore[attr-defined]
        if any(
            tool_call["name"] == delegate_title for tool_call in last_message.tool_calls
        ):
            return f"{self.name}{NODE_ROLE_SEPARATOR}delegation"

        # Check for schema tool (final response)
        if self._response_schema is not None and any(
            tool_call["name"] == self._response_schema.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}{NODE_ROLE_SEPARATOR}final_response"

        # Regular tools — optionally gated by tool approval
        if self.require_tool_approval:
            return f"{self.name}{NODE_ROLE_SEPARATOR}tool_approval_request"

        return f"{self.name}{NODE_ROLE_SEPARATOR}tools"

    def _subsession_history_key_factory(
        self, subagent_name: str, subsession_id: int
    ) -> IOKey:
        """Build the subsession-scoped conversation-history IOKey.

        Encapsulates the key naming convention so nodes never need to know it.
        Matches the ``SubsessionHistoryKeyFactory`` signature.
        """
        key = f"{self.name}{SUBSESSION_KEY_SEPARATOR}{subagent_name}{SUBSESSION_KEY_SEPARATOR}{subsession_id}"
        return IOKey(
            target="conversation_history",
            subkeys=[key],
            optional=True,
        )

    def _subsession_history_key_for(self, subagent_name: str) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` scoped to a specific subagent's conversation history.

        The returned ``RuntimeIOKey`` reads the active subsession ID from state at runtime
        and delegates to ``_subsession_history_key_factory``.  Passed into
        ``AgentComponent.bind_to_supervisor`` so the subagent never needs to
        scan state to discover its supervisor or subsession.

        Args:
            subagent_name: The name of the subagent this key is scoped to.
        """
        return RuntimeIOKey(
            alias="conversation_history",
            factory=lambda state: self._subsession_history_key_factory(
                subagent_name,
                self._resolved_active_subsession_key.value_from_state(state),
            ),
        )

    def _subsession_goal_key_factory(
        self, subagent_name: str, subsession_id: int
    ) -> IOKey:
        """Build the subsession-scoped goal IOKey.

        Encapsulates the key naming convention so nodes never need to know it.

        Args:
            subagent_name: The name of the subagent.
            subsession_id: The numeric subsession ID.
        """
        key = f"{self.name}{SUBSESSION_KEY_SEPARATOR}{subagent_name}{SUBSESSION_KEY_SEPARATOR}{subsession_id}"
        return IOKey(
            target="context",
            subkeys=[key, "goal"],
            optional=True,
        )

    def _subsession_goal_key_for(self, subagent_name: str) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` scoped to a specific subagent's goal.

        The returned ``RuntimeIOKey`` reads the active subsession ID from state at runtime
        and delegates to ``_subsession_goal_key``.  Passed into
        ``AgentComponent.bind_to_supervisor`` so the subagent's ``AgentNode``
        can read the delegation prompt from the correct subsession-scoped location.

        Args:
            subagent_name: The name of the subagent this key is scoped to.
        """
        return RuntimeIOKey(
            alias="goal",
            factory=lambda state: self._subsession_goal_key_factory(
                subagent_name,
                self._resolved_active_subsession_key.value_from_state(state),
            ),
        )

    def _subagent_final_answer_key_for(self, subagent_name: str) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` that resolves the final_answer IOKey for a specific subagent.

        The returned ``RuntimeIOKey`` reads the active subsession ID from state at runtime
        and builds the IOKey for the given subagent_name.  Passed into
        ``AgentComponent.bind_to_supervisor`` so the subagent knows where to
        write its final answer.

        Args:
            subagent_name: The name of the subagent this key is scoped to.
        """
        return RuntimeIOKey(
            alias="final_answer",
            factory=lambda state: self._subsession_final_answer_key.to_iokey(
                {
                    IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE: self.name,
                    IOKeyTemplate.COMPONENT_NAME_TEMPLATE: subagent_name,
                    IOKeyTemplate.SUBSESSION_ID_TEMPLATE: str(
                        self._resolved_active_subsession_key.value_from_state(state)
                    ),
                }
            ),
        )

    def _subagent_tool_approval_decision_key_for(
        self, subagent_name: str
    ) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` that resolves the tool_approval_decision IOKey for a specific subagent.

        The returned ``RuntimeIOKey`` reads the active subsession ID from state at runtime
        and builds the IOKey for the given subagent_name.  Passed into
        ``AgentComponent.bind_to_supervisor`` so the subagent's tool approval nodes
        store and read the decision under a subsession-scoped key, preventing race
        conditions when the same subagent runs in multiple subsessions.

        Args:
            subagent_name: The name of the subagent this key is scoped to.
        """
        return RuntimeIOKey(
            alias="tool_approval_decision",
            factory=lambda state: self._subsession_tool_approval_decision_key.to_iokey(
                {
                    IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE: self.name,
                    IOKeyTemplate.COMPONENT_NAME_TEMPLATE: subagent_name,
                    IOKeyTemplate.SUBSESSION_ID_TEMPLATE: str(
                        self._resolved_active_subsession_key.value_from_state(state)
                    ),
                }
            ),
        )

    def _subagent_cycle_count_key_for(self, subagent_name: str) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` that resolves the cycle_count IOKey for a specific subagent.

        Reads the active subsession ID from state at runtime so parallel subsessions of the same
        subagent each maintain an independent cycle counter, preventing one runaway subsession
        from exhausting the cycle budget for its siblings.

        Args:
            subagent_name: The name of the subagent this key is scoped to.
        """
        return RuntimeIOKey(
            alias="cycle_count",
            factory=lambda state: self._subsession_cycle_count_key.to_iokey(
                {
                    IOKeyTemplate.SUPERVISOR_NAME_TEMPLATE: self.name,
                    IOKeyTemplate.COMPONENT_NAME_TEMPLATE: subagent_name,
                    IOKeyTemplate.SUBSESSION_ID_TEMPLATE: str(
                        self._resolved_active_subsession_key.value_from_state(state)
                    ),
                }
            ),
        )

    @property
    def _active_subagent_final_answer_key(self) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` that resolves the final_answer IOKey for the currently active subagent.

        Reads active_subagent_name from state at runtime, then resolves the per-subagent final_answer key.  Used by
        SubagentReturnNode to read the result from whichever subagent just completed.
        """

        def _factory(state: FlowState) -> IOKey:
            active_name = self._resolved_active_subagent_name_key.value_from_state(
                state
            )

            if not active_name or active_name not in self.subagent_components:
                raise ValueError(
                    f"Cannot resolve final_answer key: no active subagent name or "
                    f"'{active_name}' not in managed agents {self.managed_agent_names}"
                )

            return self._subagent_final_answer_key_for(active_name).to_iokey(state)

        return RuntimeIOKey(alias="final_answer", factory=_factory)

    def _build_prompt(self, tools: list, tool_choice: str = "auto") -> Any:
        """Build the supervisor prompt with the given tool list."""
        tool_choice = "any" if self._response_schema is not None else "auto"
        return super()._build_prompt(tools=tools, tool_choice=tool_choice)

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        """Attach the supervisor and all subagent subgraphs to the graph.

        Builds the complete graph topology:
        - Supervisor's 3-node ReAct loop (agent ↔ tools, final_response)
        - Delegation node with routing to subagent subgraphs
        - One 3-node ReAct subgraph per managed subagent
        - Subagent return node that injects results back to supervisor
        """
        # Supervisor tools = user-specified tools + delegate_task + (schema tool if any)
        supervisor_tools = self.toolset.bindable + [self._delegate_task_cls]
        if self._response_schema is not None:
            supervisor_tools = supervisor_tools + [self._response_schema]
        prompt = self._build_prompt(tools=supervisor_tools)

        static_output_key = self._final_answer_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )
        supervisor_history_key = self._default_conversation_history_key

        # Filter supervisor events to only those compatible with UILogEventsAgent
        # (AgentNode and ToolNode use UILogWriterAgentTools which requires UILogEventsAgent)
        agent_events = [
            UILogEventsAgent[e.name]
            for e in self.ui_log_events
            if e.name in UILogEventsAgent.__members__
        ]
        node_agent = AgentNode(
            name=self.__entry_hook__(),
            conversation_history_key=supervisor_history_key,
            prompt=prompt,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            invoke_config=self._agent_node_invoke_config(),
            compactor=(
                create_conversation_compactor(
                    config=(
                        self.compaction
                        if isinstance(self.compaction, CompactionConfig)
                        else CompactionConfig()
                    ),
                    prompt_registry=self.prompt_registry,
                    user=self.user,
                    agent_name=self.name,
                    workflow_id=self.flow_id,
                    workflow_type=self.flow_type.value,
                )
                if self.compaction
                else None
            ),
            response_schema=self._response_schema,
            ui_history=UIHistory(
                events=agent_events,
                writer_class=agent_tools_ui_log_writer_class(
                    component_name=self.name,
                ),
            ),
            max_cycles=self._max_cycles_threshold,
            cycle_count_key=self._cycle_count_key,
            max_wrap_up_retries=self.max_wrap_up_retries,
            iteration_warning_offset=self._iteration_warning_offset,
        )
        tracker = ToolEventTracker(
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )
        node_tools = ToolNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}tools",
            conversation_history_key=supervisor_history_key,
            toolset=self.toolset,
            ui_history=UIHistory(
                events=agent_events,
                writer_class=agent_tools_ui_log_writer_class(
                    component_name=self.name,
                ),
            ),
            tracker=tracker,
            # Supervisor is never a subagent — session_id is always None for its own nodes
            session_id_key=NoneIOKey(alias="session_id"),
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}final_response",
            conversation_history_key=supervisor_history_key,
            output_key=RuntimeIOKey(
                alias="final_answer", factory=lambda _: static_output_key
            ),
            ui_history=UIHistory(
                events=self.ui_log_events,  # type: ignore[arg-type]
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsSupervisor,
                    ui_role_as=self.ui_role_as,  # type: ignore[arg-type]
                    component_name=self.name,
                ),
            ),
            response_schema=self._response_schema,
            component_name=self.name,
            # Supervisor is never a subagent — session_id is always None for its own nodes
            session_id_key=NoneIOKey(alias="session_id"),
        )

        # --- Delegation node ---
        node_delegation = DelegationNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}delegation",
            max_delegations=self.max_delegations,
            delegate_task_cls=self._delegate_task_cls,
            delegation_count_key=self._resolved_delegation_count_key,
            active_subsession_key=self._resolved_active_subsession_key,
            active_subagent_name_key=self._resolved_active_subagent_name_key,
            max_subsession_id_key=self._resolved_max_subsession_id_key,
            supervisor_history_key=supervisor_history_key,
            subsession_history_key_factory=self._subsession_history_key_factory,
            subsession_goal_key_factory=self._subsession_goal_key_factory,
            ui_history=UIHistory(
                events=[
                    UILogEventsSupervisor.ON_DELEGATION,
                    UILogEventsSupervisor.ON_DELEGATION_ERROR,
                ],
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsSupervisor,
                    ui_role_as=MessageTypeEnum.TOOL.value,
                    component_name=self.name,
                ),
            ),
        )

        # --- Subagent return node ---
        node_subagent_return = SubagentReturnNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}subagent_return",
            delegate_task_cls=self._delegate_task_cls,
            active_subsession_key=self._resolved_active_subsession_key,
            active_subagent_name_key=self._resolved_active_subagent_name_key,
            final_answer_key=self._active_subagent_final_answer_key,
            supervisor_history_key=supervisor_history_key,
            ui_history=UIHistory(
                events=[UILogEventsSupervisor.ON_DELEGATION_RETURNS],
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsSupervisor,
                    ui_role_as=MessageTypeEnum.TOOL.value,
                    component_name=self.name,
                ),
            ),
        )

        # --- Add supervisor nodes to graph ---
        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)
        graph.add_node(node_delegation.name, node_delegation.run)
        graph.add_node(node_subagent_return.name, node_subagent_return.run)

        self._attach_tool_approval_nodes(
            graph,
            conversation_history_key=supervisor_history_key,
            ui_log_events=agent_events,
        )

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
                conversation_history_key=self._subsession_history_key_for(agent_name),
                output_key=self._subagent_final_answer_key_for(agent_name),
                goal_key=self._subsession_goal_key_for(agent_name),
                session_id_key=self._resolved_active_subsession_key,
                tool_approval_decision_key=self._subagent_tool_approval_decision_key_for(
                    agent_name
                ),
                cycle_count_key=self._subagent_cycle_count_key_for(agent_name),
            )
            subagent.attach(graph, subagent_router)
