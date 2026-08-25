from typing import Any, ClassVar, Optional, Self, TypedDict, override

from dependency_injector.wiring import inject
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
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
    DEFAULT_SESSION_ID_KEY,
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
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    SubagentDescriptor,
    build_delegate_task_model,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes import (
    DelegationCollectNode,
    DelegationPrepareNode,
    SubagentDispatchNode,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    SUBSESSION_RUNS_CONTEXT_KEY,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.ui_log import (
    UILogEventsSupervisor,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.v1.state.base import (
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    get_model_max_context_token_limit,
)
from duo_workflow_service.tracking.subagent_delegation import (
    SubagentDelegationTracker,
)

__all__ = ["SubagentConfig", "SupervisorAgentComponentV2", "extract_subagent_names"]


class SubagentConfig(TypedDict):
    """Descriptor for a subagent entry in the YAML flow configuration.

    Represents a single entry in the ``subagents`` list of a
    ``SupervisorAgentComponentV2`` config block.  Only ``name`` is required.
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


@inject
class SupervisorAgentComponentV2(AgentComponentBase):
    """Supervisor component that orchestrates subagents via delegate_task tool.

    The SupervisorAgentComponentV2 acts as a container that manages dedicated
    ReAct subgraphs for each of its managed subagents.

    Key capabilities:
    1. Explicit delegation via delegate_task tool
    2. Contextual handoffs via prompt injection as HumanMessage
    3. Session management (every delegation runs as its own new subsession)
    4. Sub-agent results injected as ToolMessage responses to delegate_task calls

    Graph topology:
        supervisor#agent ↔ supervisor#tools (regular tools)
        supervisor#agent → supervisor#delegation_prepare (delegate_task)
        supervisor#agent → supervisor#final_response (final_response_tool)
        supervisor#delegation_prepare → <subagent_name> (native Send, one per
            delegate_task call ready to dispatch — concurrently, including
            multiple concurrent dispatches to the same subagent type)
        supervisor#delegation_prepare → supervisor#agent (if nothing is left to
            dispatch — every call failed validation, or a mixed
            delegate_task/other-tool turn; the prepare node answers those
            calls itself)
        <subagent_name> → supervisor#delegation_collect (every managed
            subagent type has an edge here; the collect node only actually
            runs once all of this turn's dispatches have completed)
        supervisor#delegation_collect → supervisor#agent (loop)

    Each managed subagent is compiled once (``AgentComponent.compile_as_subagent``)
    and attached as a real node of *this* graph (see ``attach``), so
    ``delegate_task`` calls are dispatched as native LangGraph ``Send`` tasks —
    run concurrently by LangGraph's own Pregel scheduler, each with its own
    checkpoint namespace and, if it pauses for tool approval, its own
    independently-resumable interrupt ID.
    """

    # Orchestration metadata written by `DelegationPrepareNode`.
    # All are optional=True so value_from_state returns None (instead of raising
    # KeyError) before the supervisor's context sub-dict is first populated.
    _delegation_count_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "delegation_count"],
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
    # because every SupervisorAgentComponentV2 manages a different subagent set and
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
        ``compile_as_subagent``, replaces ``subagent_components`` with the
        filtered dict, and removes ``subagents`` so the runtime model uses
        ``subagent_components.keys()`` as the source of truth.
        """
        if isinstance(data, dict):
            subagents: list[SubagentConfig] = data.pop("subagents", [])
            all_components = data.pop("subagent_components", {})

            if not subagents:
                raise ValueError(
                    "SupervisorAgentComponentV2 requires at least one managed agent."
                )

            selected_components: dict[str, Any] = {}
            for agent_name in extract_subagent_names(subagents):
                if agent_name not in all_components:
                    raise ValueError(
                        f"Managed agent '{agent_name}' not found in subagent_components. "
                        f"Available: {list(all_components.keys())}"
                    )

                component = all_components[agent_name]

                if not hasattr(component, "compile_as_subagent") or not callable(
                    getattr(component, "compile_as_subagent")
                ):
                    raise ValueError(
                        f"Managed agent '{agent_name}' of type '{type(component).__name__}' "
                        f"does not have a compile_as_subagent method. "
                        f"Managed agents must have a compile_as_subagent method."
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
                "SupervisorAgentComponentV2 requires at least one subagent component."
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
    def _resolved_max_subsession_id_key(self) -> IOKey:
        """Resolve the max_subsession_id ``IOKey`` for this supervisor instance."""
        return self._max_subsession_id_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    @property
    def outputs(self) -> tuple[IOKey, ...]:
        replacements = {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        return tuple(output.to_iokey(replacements) for output in self._outputs)

    def _agent_node_router(self, state: FlowState) -> str:
        """Router for the supervisor's agent node.

        Routes based on the last message:
        - text-only (no tool calls) → supervisor#final_response (implicit final answer)
        - delegate_task → supervisor#delegation_prepare
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
            return f"{self.name}{NODE_ROLE_SEPARATOR}delegation_prepare"

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

    def _subsession_run_key_factory(self, call_id: str) -> IOKey:
        """Build the IOKey holding one dispatched delegation's ``SubsessionRun`` record.

        Encapsulates the key naming convention so nodes never need to know it.
        Matches the ``SubsessionRunKeyFactory`` signature. Written by
        ``SubagentDispatchNode`` once a dispatched subagent finishes (or fails),
        and read back by ``DelegationCollectNode`` when it builds that call's
        ToolMessage. Scoped to this supervisor's own name — so two supervisors
        in one flow can't overwrite each other's records — and never part of
        ``_outputs``: it is purely an internal wiring detail.

        Keyed by ``delegate_task`` call ID rather than by subsession, so that a
        record can only ever answer the one call that produced it.

        Records are not cleared once consumed: ``context`` deep-merges, so
        there is no update that can remove a key from it, only ones that
        overwrite leaves. They accumulate for the lifetime of the flow, at one
        small record (a subsession ID, a status, and the answer already stored
        in the supervisor's own conversation history) per delegation.

        Args:
            call_id: The ID of the ``delegate_task`` call that was dispatched.
        """
        return IOKey(
            target="context",
            subkeys=[self.name, SUBSESSION_RUNS_CONTEXT_KEY, call_id],
            optional=True,
        )

    def _build_prompt(self, tools: list, tool_choice: str = "auto") -> Any:
        """Build the supervisor prompt with the given tool list."""
        tool_choice = "any" if self._response_schema is not None else "auto"
        return super()._build_prompt(tools=tools, tool_choice=tool_choice)

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        """Attach the supervisor to the graph, wiring subagents as native dispatch targets.

        Builds the complete graph topology:
        - Supervisor's 3-node ReAct loop (agent ↔ tools, final_response)
        - A delegation-prepare node, which validates every ``delegate_task``
            call from the supervisor's last turn, answers the invalid ones
            itself, and dispatches the valid ones as native ``Send`` tasks —
            targeting one ``SubagentDispatchNode`` per managed subagent type,
            run concurrently by LangGraph's own Pregel scheduler (including
            multiple concurrent dispatches to the *same* subagent type).
        - A delegation-collect node, which runs once every dispatch this turn
            has completed, and turns each dispatched call's run record into a
            ``ToolMessage`` response back to the supervisor.

        Unlike the sequential ``SupervisorAgentComponent`` (whose
        ``DelegationNode``/``SubagentReturnNode`` pair drives one subagent
        flattened into the supervisor's own graph via ``bind_to_supervisor``),
        managed subagents *are* added as real nodes of this graph (see
        ``AgentComponent.compile_as_subagent``) — this is what lets
        ``delegate_task`` calls be dispatched via ``Send`` and run truly
        concurrently under LangGraph's own scheduler, with automatic
        per-dispatch checkpoint namespacing and interrupt-ID assignment.
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
            max_context_tokens=get_model_max_context_token_limit(self.model_tags),
            optimizer_pipeline=self._build_optimizer_pipeline(),
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
            session_id_key=DEFAULT_SESSION_ID_KEY,
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
            session_id_key=DEFAULT_SESSION_ID_KEY,
        )

        # --- Compile every managed subagent, and register it as a real node of
        # --- this graph, so it's dispatchable via native Send(...) tasks.
        #
        # Compiled once per subagent *type*, so two concurrent dispatches of the
        # same type run through the same node objects. Their `FlowState`s stay
        # isolated (each `Send` carries its own), but any state a node keeps on
        # itself is shared: today that is `UIHistory._logs`, which accumulates
        # entries and is drained wholesale by `pop_state_updates()`, so the run
        # that drains first can carry off a sibling's entries. Tracked as a known
        # limitation; the fix is to scope that accumulation per invocation (or to
        # compile per dispatch).
        subagent_graphs: dict[str, CompiledStateGraph] = {
            agent_name: subagent.compile_as_subagent()
            for agent_name, subagent in self.subagent_components.items()
        }
        dispatch_nodes = {
            agent_name: SubagentDispatchNode(
                subagent_name=agent_name,
                subsession_run_key_factory=self._subsession_run_key_factory,
                compiled_graph=compiled_graph,
            )
            for agent_name, compiled_graph in subagent_graphs.items()
        }

        delegation_tracker = SubagentDelegationTracker(
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            supervisor_name=self.name,
            parallel=True,
        )

        # --- Delegation prepare/collect nodes ---
        node_delegation_prepare = DelegationPrepareNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}delegation_prepare",
            supervisor_name=self.name,
            max_delegations=self.max_delegations,
            delegate_task_cls=self._delegate_task_cls,
            delegation_count_key=self._resolved_delegation_count_key,
            max_subsession_id_key=self._resolved_max_subsession_id_key,
            supervisor_history_key=supervisor_history_key,
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
            tracker=delegation_tracker,
        )
        node_delegation_collect = DelegationCollectNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}delegation_collect",
            delegate_task_cls=self._delegate_task_cls,
            subsession_run_key_factory=self._subsession_run_key_factory,
            supervisor_history_key=supervisor_history_key,
            ui_history=UIHistory(
                events=[
                    UILogEventsSupervisor.ON_DELEGATION_RETURNS,
                ],
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsSupervisor,
                    ui_role_as=MessageTypeEnum.TOOL.value,
                    component_name=self.name,
                ),
            ),
            tracker=delegation_tracker,
        )

        # --- Add supervisor nodes to graph ---
        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)
        # `DelegationPrepareNode.run` dispatches by returning a `Command`, whose
        # targets are the managed subagents (as `Send` tasks), or -- when there
        # is nothing left to dispatch, every call having been answered with a
        # validation error by the prepare node itself -- this supervisor's agent
        # node. Those names come from the flow config at runtime, so they are
        # declared here rather than as a `Command[Literal[...]]` annotation;
        # without them the graph has no edges out of this node to render or
        # validate.
        graph.add_node(
            node_delegation_prepare.name,
            node_delegation_prepare.run,
            destinations=(*dispatch_nodes, node_agent.name),
        )
        graph.add_node(node_delegation_collect.name, node_delegation_collect.run)

        # --- Add every managed subagent as a real, dispatchable node ---
        for agent_name, dispatch_node in dispatch_nodes.items():
            graph.add_node(agent_name, dispatch_node.run)
            graph.add_edge(agent_name, node_delegation_collect.name)

        self._attach_tool_approval_nodes(
            graph,
            conversation_history_key=supervisor_history_key,
            # Supervisor is never a subagent -- session_id is always None for
            # its own nodes. A subagent's own approval prompts are attributed
            # to its subsession by `AgentComponent.compile_as_subagent`, which
            # points `_session_id_key` at the dispatched subsession ID.
            session_id_key=DEFAULT_SESSION_ID_KEY,
            tracker=tracker,
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

        # Delegation prepare needs no edges: it dispatches every valid
        # delegate_task call as a native Send task (concurrently), or names the
        # agent node directly, from the `Command` it returns -- see the
        # `destinations` declared with its node above.

        # Delegation collect → always back to supervisor agent (every
        # dispatched subagent this turn — success or failure — has already
        # completed by the time this node runs).
        graph.add_edge(node_delegation_collect.name, node_agent.name)
