from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Optional,
    Self,
    Type,
    Union,
    override,
)

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import StateGraph
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from ai_gateway.response_schemas.registry import BaseAgentOutput
from duo_workflow_service.agent_platform.constants import (
    NODE_ROLE_SEPARATOR,
    RECURSION_LIMIT,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes import (
    AgentNode,
    FinalResponseNode,
    ToolApprovalFetchNode,
    ToolApprovalRequestNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
    agent_tools_ui_log_writer_class,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowEventType,
    FlowState,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.experimental.state.base import (
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.experimental.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)

# Re-export RoutingError from v1 to prevent code duplication.
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey, NoneIOKey
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.tools.toolset import Toolset
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events import InternalEventsClient

__all__ = ["AgentComponent", "AgentComponentBase", "RoutingError"]


class AgentComponentBase(BaseComponent):
    """Shared base for agent-style components (AgentComponent, SupervisorAgentComponent).

    Holds the common field declarations and class-level metadata shared by all
    agent variants.  Subclasses must override ``_agent_node_router`` and
    ``attach`` with their own graph-topology logic.

    Do NOT use this class directly in flow configs — use AgentComponent instead.
    """

    STREAMING_ENABLED_CONFIG: ClassVar[RunnableConfig] = {}
    STREAMING_DISABLED_CONFIG: ClassVar[RunnableConfig] = {"tags": [TAG_NOSTREAM]}

    _final_answer_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "final_answer"],
    )

    _cycle_count_key_template: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "cycle_count"],
        optional=True,
    )

    _tool_approval_decision_key_template: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "tool_approval_decision"],
        optional=True,
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(
            target="conversation_history",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE],
        ),
        IOKeyTemplate(target="status"),
        _final_answer_key,
    )

    supported_environments: ClassVar[tuple[str, ...]] = (
        "platform",
        "ide",
        "local",
        "remote",
    )

    prompt_id: str
    prompt_version: Optional[str] = None
    toolset: Toolset
    compaction: Union[CompactionConfig, bool] = False
    response_schema_id: Optional[str] = None
    response_schema_version: Optional[str] = None
    response_schema_tracking: bool = False
    _DEFAULT_SOFT_LIMIT_OFFSET: ClassVar[int] = 20
    _DEFAULT_MAX_CYCLES: ClassVar[int] = max(
        1, RECURSION_LIMIT - _DEFAULT_SOFT_LIMIT_OFFSET
    )

    max_cycles: int = _DEFAULT_MAX_CYCLES
    max_wrap_up_retries: int = 3
    # Opt-in (per flow config): bind the provider-native web-search tool so the
    # agent can look up e.g. changelogs / migration guides. Still gated at runtime
    # by the dependency_bump_web_search flag and the client's "web_search"
    # capability, and only honored by native-Anthropic models (LiteLLM discards it).
    enable_web_search: bool = False

    @field_validator("max_cycles")
    @classmethod
    def validate_max_cycles(cls, v: int) -> int:
        """Validate that max_cycles is at least 1."""
        if v < 1:
            raise ValueError("max_cycles must be at least 1.")
        return v

    @field_validator("max_wrap_up_retries")
    @classmethod
    def validate_max_wrap_up_retries(cls, v: int) -> int:
        """Validate that max_wrap_up_retries is at least 1."""
        if v < 1:
            raise ValueError("max_wrap_up_retries must be at least 1.")
        return v

    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ]
    schema_registry: BaseResponseSchemaRegistry = Provide[
        ContainerApplication.pkg_schemas.schema_registry
    ]
    internal_event_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ]

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    _response_schema: Optional[Type[BaseAgentOutput]] = PrivateAttr()
    _cycle_count_key: RuntimeIOKey = PrivateAttr()

    @model_validator(mode="after")
    def initialize_cycle_count_key(self) -> Self:
        """Initialize the cycle count key with a component-scoped default RuntimeIOKey.

        As with ``tool_approval_decision_key``, subagents may override this key via
        ``bind_to_supervisor`` to use a subsession-scoped key, preventing race conditions
        when the same subagent runs in multiple subsessions in parallel.
        """
        static_key = self._cycle_count_key_template.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )
        self._cycle_count_key = RuntimeIOKey(
            alias="cycle_count",
            factory=lambda _: static_key,
        )
        return self

    @model_validator(mode="after")
    def validate_and_resolve_response_schema(self) -> Self:
        """Validate response schema params and resolve the schema.

        Supports three mutually exclusive modes, mirroring how prompts work:

        1. Registry mode: both ``response_schema_id`` and ``response_schema_version``
            are set — the schema is loaded from the shared file-based registry.
        2. Inline mode: ``response_schema_id`` is set but ``response_schema_version``
            is omitted — the schema is looked up in the flow's
            ``InlineResponseSchemaRegistry`` (keyed by ``response_schema_id``).
        3. Default text-only mode: ``response_schema_id`` is not set.

        Raises:
            ValueError: If ``response_schema_version`` is set without
                ``response_schema_id``, or if other validation constraints are
                violated.
        """
        if self.response_schema_version and not self.response_schema_id:
            raise ValueError(
                "response_schema_version requires response_schema_id to be set."
            )

        if self.response_schema_tracking and not self.response_schema_id:
            raise ValueError(
                "response_schema_tracking requires response_schema_id to be set."
            )

        if self.response_schema_id:
            response_schema = self.schema_registry.get(
                schema_id=self.response_schema_id,
                schema_version=self.response_schema_version or "",
            )
            if response_schema.tool_title in self.toolset:
                raise ValueError(
                    f"Response schema tool title '{response_schema.tool_title}' "
                    f"collides with existing tool: '{response_schema.tool_title}'"
                )
        else:
            response_schema = None

        self._response_schema = response_schema
        return self

    @property
    def _default_conversation_history_key(self) -> RuntimeIOKey:
        """Return a ``RuntimeIOKey`` for this component's conversation history.

        The key is static (component-scoped) and does not depend on runtime state,
        but is wrapped in a ``RuntimeIOKey`` so it can be passed uniformly to nodes
        that accept ``RuntimeIOKey`` for their conversation-history parameter.

        Returns:
            A ``RuntimeIOKey`` wrapping the static ``IOKey`` for this component's
            conversation history slot.
        """
        static_key = IOKey(
            target="conversation_history",
            subkeys=[self.name],
            optional=True,
        )
        return RuntimeIOKey(alias="conversation_history", factory=lambda _: static_key)

    def __entry_hook__(self) -> Annotated[str, "Entry node name"]:
        return f"{self.name}{NODE_ROLE_SEPARATOR}agent"

    def _agent_node_invoke_config(self) -> RunnableConfig:
        """Return the ``RunnableConfig`` to pass to every ``AgentNode`` ``ainvoke`` call.

        Subclasses must override this method to express their own streaming policy
        in terms of their own typed ``ui_log_events`` enum.

        Return ``STREAMING_ENABLED_CONFIG`` to allow LLM chunks to stream to the
        UI, or ``STREAMING_DISABLED_CONFIG`` to suppress them.
        """
        raise NotImplementedError

    def _web_search_enabled(self) -> bool:
        """Whether the provider-native web-search tool is active for this build.

        Requires the per-flow opt-in AND the runtime feature flag AND the client's
        `web_search` capability. Used both to bind the tool and to tell the prompt
        (via `tools_enabled`) so it can branch its web-search guidance.
        """
        return (
            self.enable_web_search
            and is_feature_enabled(FeatureFlag.DEPENDENCY_BUMP_WEB_SEARCH)
            and is_client_capable("web_search")
        )

    def _tools_enabled(self) -> dict[str, bool]:
        """Map of optional tool/capability -> active, exposed to the prompt template."""
        return {"web_search": self._web_search_enabled()}

    def _build_prompt(self, tools: list, tool_choice: str) -> Any:
        """Build the agent prompt with the given tool list and tool choice."""
        extra_params: dict[str, Any] = {}
        if self._web_search_enabled():
            extra_params["bind_tools_params"] = {"web_search_options": {}}
        return self.prompt_registry.get_on_behalf(
            self.user,
            self.prompt_id,
            self.prompt_version,
            tools=tools,
            tool_choice=tool_choice,
            is_graph_node=True,
            internal_event_extra={
                "agent_name": self.name,
                "workflow_id": self.flow_id,
                "workflow_type": self.flow_type.value,
            },
            **extra_params,
        )

    def _agent_node_router(self, state: FlowState) -> str:
        raise NotImplementedError

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        raise NotImplementedError


@inject
class AgentComponent(AgentComponentBase):
    """AgentComponent for use in flow configs.

    Provides the standard single-agent ReAct loop (agent ↔ tools, final_response) with dependency injection applied.

    Can be used standalone or bound to a SupervisorAgentComponent via bind_to_supervisor.
    When bound to a supervisor:
    - The description field becomes required (used by supervisor's delegate_task tool)
    - The component uses supervisor-provided key factories for conversation history and output
    - Output keys are resolved at runtime based on the active subsession

    When used standalone:
    - The description field is optional
    - Uses default key factories that resolve to component-scoped keys
    - Output keys are static and component-scoped
    """

    description: Optional[str] = None
    ui_log_events: list[UILogEventsAgent] = Field(default_factory=list)
    ui_role_as: Literal["agent", "tool"] = "agent"
    require_tool_approval: bool = False
    pre_approved_tools: list[str] = Field(default_factory=list)

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    @override
    def _agent_node_invoke_config(self) -> RunnableConfig:
        """Return TAG_NOSTREAM config unless both LLM output event types are declared."""
        if (
            UILogEventsAgent.ON_AGENT_FINAL_ANSWER in self.ui_log_events
            and UILogEventsAgent.ON_AGENT_REASONING in self.ui_log_events
        ):
            return self.STREAMING_ENABLED_CONFIG
        return self.STREAMING_DISABLED_CONFIG

    # Private attributes for RuntimeIOKey instances with default values.
    # Overridden by bind_to_supervisor when used as a subagent.
    _conversation_history_key: RuntimeIOKey = PrivateAttr()
    _output_key: RuntimeIOKey = PrivateAttr()
    _session_id_key: BaseIOKey = PrivateAttr()
    _tool_approval_decision_key: RuntimeIOKey = PrivateAttr()
    _is_bound_to_supervisor: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def initialize_key_factories(self) -> Self:
        """Initialize RuntimeIOKey instances with default values for standalone use.

        These defaults are overridden by bind_to_supervisor when the component is used as a subagent.
        """
        # Default conversation history key uses component-scoped static key
        self._conversation_history_key = self._default_conversation_history_key

        # Default output key resolves to a static component-scoped final_answer key
        static_output_key = self._final_answer_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )
        self._output_key = RuntimeIOKey(
            alias="final_answer",
            factory=lambda _: static_output_key,
        )

        # Default session_id_key is a no-op key; overridden by bind_to_supervisor
        # when used as a subagent to attribute tool calls to subsessions in UI logs.
        self._session_id_key = NoneIOKey(alias="session_id")

        # Default tool approval decision key is component-scoped (no subsession namespace).
        # Overridden by bind_to_supervisor to be subsession-scoped when used as a subagent,
        # preventing race conditions when the same subagent runs in multiple subsessions.
        static_tool_approval_decision_key = (
            self._tool_approval_decision_key_template.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            )
        )
        self._tool_approval_decision_key = RuntimeIOKey(
            alias="tool_approval_decision",
            factory=lambda _: static_tool_approval_decision_key,
        )

        return self

    def bind_to_supervisor(
        self,
        *,
        conversation_history_key: RuntimeIOKey,
        output_key: RuntimeIOKey,
        goal_key: RuntimeIOKey,
        session_id_key: BaseIOKey = NoneIOKey(alias="session_id"),
        tool_approval_decision_key: RuntimeIOKey,
        cycle_count_key: RuntimeIOKey,
    ) -> None:
        """Bind this agent to a supervisor.

        Must be called before ``attach`` when using this component as a subagent.
        The supervisor passes subsession-scoped ``RuntimeIOKey`` instances so the
        agent never needs to scan state to discover which supervisor owns it or
        what the active subsession ID is.

        When bound to a supervisor, the description field must be set.

        Args:
            conversation_history_key: ``RuntimeIOKey`` that resolves the
                subsession-scoped conversation-history key at runtime.
            output_key: ``RuntimeIOKey`` that resolves the subsession-scoped
                final_answer key at runtime.
            goal_key: ``RuntimeIOKey`` that resolves the subsession-scoped goal
                key at runtime.  The resolved IOKey replaces the static
                ``context:goal`` input so the subagent reads the delegation
                prompt written by ``DelegationNode`` rather than the shared
                flow goal.
            session_id_key: ``BaseIOKey`` that resolves the subsession ID at
                runtime.  Used to attribute tool calls to subsessions in UI
                logs.  Defaults to a no-op key when not provided.
            tool_approval_decision_key: ``RuntimeIOKey`` that resolves the
                subsession-scoped tool approval decision key at runtime.  The
                tool approval decision is stored under a subsession-scoped key,
                preventing race conditions when the same subagent runs in
                multiple subsessions in parallel.
            cycle_count_key: ``RuntimeIOKey`` that resolves the subsession-scoped
                cycle count key at runtime.  The cycle count is stored under a
                subsession-scoped key, preventing race conditions when the same
                subagent runs in multiple subsessions in parallel.

        Raises:
            ValueError: If description is not set when binding to supervisor.
        """
        if not self.description:
            raise ValueError(
                f"AgentComponent '{self.name}' must have a description when bound to a supervisor."
            )
        self._conversation_history_key = conversation_history_key
        self._output_key = output_key
        self._session_id_key = session_id_key
        self._tool_approval_decision_key = tool_approval_decision_key
        self._cycle_count_key = cycle_count_key
        self._is_bound_to_supervisor = True

        # Ensure subagent does not read shared `context:goal` directly
        self.inputs = [
            inp
            for inp in self.inputs
            if inp.template_variable_name != goal_key.template_variable_name
        ]
        self.inputs.append(goal_key)

    def _agent_node_router(self, state: FlowState) -> str:
        history_iokey = self._conversation_history_key.to_iokey(state)
        history: list[BaseMessage] = history_iokey.value_from_state(state) or []
        if not history:
            raise NotifiableAgentException(
                "An internal error occurred: no conversation history was found.",
                internal_detail=(
                    f"Conversation history not found for key "
                    f"{history_iokey.target}:{history_iokey.subkeys}"
                ),
            )

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise NotifiableAgentException(
                "An internal error occurred: the agent produced an unexpected message type.",
                internal_detail=f"Last message is not AIMessage for component {self.name}",
            )

        if not last_message.tool_calls:
            if self._response_schema is not None:
                raise NotifiableAgentException(
                    "An internal error occurred: the agent did not produce the expected tool call.",
                    internal_detail=(
                        f"Schema mode requires a tool call but got a text-only response "
                        f"for component {self.name}"
                    ),
                )
            return f"{self.name}{NODE_ROLE_SEPARATOR}final_response"

        if self._response_schema is not None and any(
            tool_call["name"] == self._response_schema.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}{NODE_ROLE_SEPARATOR}final_response"

        if self.require_tool_approval:
            return f"{self.name}{NODE_ROLE_SEPARATOR}tool_approval_request"

        return f"{self.name}{NODE_ROLE_SEPARATOR}tools"

    def _tool_approval_request_router(self, state: FlowState) -> str:
        """Route from tool approval request node.

        Routes to:
            - fetch: If approval is required (status=TOOL_CALL_APPROVAL_REQUIRED)
            - tools: If all tools pre-approved (status=EXECUTION)
        """
        # Check workflow status to determine routing
        status_iokey = RuntimeIOKey(
            alias="status",
            factory=lambda _: IOKey(target="status"),
        ).to_iokey(state)
        status = status_iokey.value_from_state(state)

        # Route based on explicit status set by request node
        needs_approval = status == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED

        if needs_approval:
            target = f"{self.name}{NODE_ROLE_SEPARATOR}tool_approval_fetch"
        else:
            target = f"{self.name}{NODE_ROLE_SEPARATOR}tools"

        return target

    def _tool_approval_fetch_router(self, state: FlowState) -> str:
        """Route from tool approval fetch node.

        Routes to:
            - tools: If approval was granted (decision=APPROVE)
            - agent: If approval was rejected (decision=REJECT or MODIFY)
        """
        approval_decision_iokey = self._tool_approval_decision_key.to_iokey(state)

        decision = approval_decision_iokey.value_from_state(state)

        if not decision:
            raise RoutingError(f"No approval decision found in state for {self.name}")

        # Route based on explicit decision
        if decision == FlowEventType.APPROVE:
            return f"{self.name}{NODE_ROLE_SEPARATOR}tools"
        if decision in [FlowEventType.REJECT, FlowEventType.MODIFY]:
            return f"{self.name}{NODE_ROLE_SEPARATOR}agent"

        raise RoutingError(f"Unexpected approval decision: {decision}")

    @override
    def __entry_hook__(self) -> Annotated[str, "Entry node name"]:
        return f"{self.name}{NODE_ROLE_SEPARATOR}agent"

    @property
    def outputs(self) -> tuple[IOKey, ...]:
        """Return the outputs for this component.

        When bound to a supervisor, returns an empty tuple since the output keys
        require runtime substitutions (supervisor name, subagent name, subsession ID)
        that are only known at graph execution time.

        For standalone components with custom response schemas, this includes both
        the base final_answer output and individual field paths for each schema field,
        enabling easier discovery and access to nested data.

        Returns:
            Tuple of IOKey objects describing all output paths from this component.

        Examples:
            Default (no custom schema):
                - context:<name>.final_answer (string)

            Custom schema with fields (summary, issues_found, recommendations):
                - context:<name>.final_answer (dict)
                - context:<name>.final_answer.summary
                - context:<name>.final_answer.issues_found
                - context:<name>.final_answer.recommendations
        """
        # If bound to supervisor, return empty tuple (keys resolved at runtime)
        if self._is_bound_to_supervisor:
            return ()

        replacements = {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        base_outputs = tuple(output.to_iokey(replacements) for output in self._outputs)

        # If using custom response schema, add individual field outputs
        if self._response_schema is not None:
            # Add output keys for each field in the schema
            field_outputs = []
            for field_name in self._response_schema.model_fields.keys():
                field_outputs.append(
                    IOKey(
                        target="context",
                        subkeys=[self.name, "final_answer", field_name],
                    )
                )

            return base_outputs + tuple(field_outputs)

        return base_outputs

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        # Response schema is already resolved in validate_and_resolve_response_schema()
        if self._response_schema is not None:
            tools = self.toolset.bindable + [self._response_schema]
            tool_choice = "any"
        else:
            tools = self.toolset.bindable
            tool_choice = "auto"

        prompt = self._build_prompt(tools=tools, tool_choice=tool_choice)

        node_agent = AgentNode(
            name=self.__entry_hook__(),
            conversation_history_key=self._conversation_history_key,
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
                events=self.ui_log_events,
                writer_class=agent_tools_ui_log_writer_class(
                    component_name=self.name,
                ),
            ),
            max_cycles=self.max_cycles,
            cycle_count_key=self._cycle_count_key,
            max_wrap_up_retries=self.max_wrap_up_retries,
            prompt_template_inputs={"tools_enabled": self._tools_enabled()},
        )
        tracker = ToolEventTracker(
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )
        node_tools = ToolNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}tools",
            conversation_history_key=self._conversation_history_key,
            toolset=self.toolset,
            ui_history=UIHistory(
                events=self.ui_log_events,
                writer_class=agent_tools_ui_log_writer_class(
                    component_name=self.name,
                ),
            ),
            tracker=tracker,
            session_id_key=self._session_id_key,
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}{NODE_ROLE_SEPARATOR}final_response",
            conversation_history_key=self._conversation_history_key,
            output_key=self._output_key,
            ui_history=UIHistory(
                events=self.ui_log_events,
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsAgent, ui_role_as=self.ui_role_as
                ),
            ),
            response_schema=self._response_schema,
            response_schema_tracking=self.response_schema_tracking,
            component_name=self.name,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )

        graph.add_node(self.__entry_hook__(), node_agent.run)
        graph.add_node(node_tools.name, node_tools.run)
        graph.add_node(node_final_response.name, node_final_response.run)

        # Conditionally add tool approval nodes
        if self.require_tool_approval:
            node_tool_approval_request = ToolApprovalRequestNode(
                name=f"{self.name}{NODE_ROLE_SEPARATOR}tool_approval_request",
                conversation_history_key=self._conversation_history_key,
                toolset=self.toolset,
                pre_approved_tools=self.pre_approved_tools,
                status_key=RuntimeIOKey(
                    alias="status",
                    factory=lambda _: IOKey(target="status"),
                ),
                ui_history=UIHistory(
                    events=self.ui_log_events,
                    writer_class=UILogWriterAgentTools,
                ),
            )

            node_tool_approval_fetch = ToolApprovalFetchNode(
                name=f"{self.name}{NODE_ROLE_SEPARATOR}tool_approval_fetch",
                conversation_history_key=self._conversation_history_key,
                status_key=RuntimeIOKey(
                    alias="status",
                    factory=lambda _: IOKey(target="status"),
                ),
                approval_decision_key=self._tool_approval_decision_key,
            )

            # Add approval nodes to graph
            graph.add_node(
                node_tool_approval_request.name, node_tool_approval_request.run
            )
            graph.add_node(node_tool_approval_fetch.name, node_tool_approval_fetch.run)

            # Add edges for approval flow
            # Conditional edge from request: goes to fetch if approval needed, tools if all pre-approved
            graph.add_conditional_edges(
                node_tool_approval_request.name,
                self._tool_approval_request_router,
            )
            # Conditional edge from fetch: goes to tools if approved, agent if rejected
            graph.add_conditional_edges(
                node_tool_approval_fetch.name,
                self._tool_approval_fetch_router,
            )

        graph.add_conditional_edges(
            node_agent.name,
            self._agent_node_router,
        )
        graph.add_edge(node_tools.name, node_agent.name)

        graph.add_conditional_edges(
            node_final_response.name,
            router.route,
        )
