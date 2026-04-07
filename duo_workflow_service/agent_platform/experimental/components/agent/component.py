from typing import Annotated, ClassVar, Literal, Optional, Self, Type, Union, override

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from pydantic import Field, PrivateAttr, model_validator

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from ai_gateway.response_schemas.registry import BaseAgentOutput
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
    BaseComponent,
    RouterProtocol,
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
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events import InternalEventsClient

__all__ = ["AgentComponent", "AgentComponentBase", "RoutingError"]


class RoutingError(Exception):
    """Exception raised when edge routers encounter unexpected conditions."""


class AgentComponentBase(BaseComponent):
    """Shared base for agent-style components (AgentComponent, SupervisorAgentComponent).

    Holds the common field declarations and class-level metadata shared by all
    agent variants.  Subclasses must override ``_agent_node_router`` and
    ``attach`` with their own graph-topology logic.

    Do NOT use this class directly in flow configs — use AgentComponent instead.
    """

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
    prompt_version: Optional[str] = None
    toolset: Toolset
    compaction: Union[CompactionConfig, bool] = False
    response_schema_id: Optional[str] = None
    response_schema_version: Optional[str] = None

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

    @model_validator(mode="after")
    def validate_and_resolve_response_schema(self) -> Self:
        """Validate response schema params and resolve the schema.

        1. Validates that response_schema_id and response_schema_version are either both set or both None.
        2. Resolves the response schema from the registry, or uses None for default text-only mode.
        3. Validates that the schema tool name doesn't collide with existing tools.
        """
        if bool(self.response_schema_id) != bool(self.response_schema_version):
            raise ValueError(
                "response_schema_id and response_schema_version must be provided together. "
                "Either provide both or omit both for default text-only mode."
            )

        if self.response_schema_id and self.response_schema_version:
            response_schema = self.schema_registry.get(
                schema_id=self.response_schema_id,
                schema_version=self.response_schema_version,
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

    def _conversation_history_key(self, _state: FlowState) -> IOKey:
        """Return the ``IOKey`` for this component's conversation history.

        Matches the ``ConversationHistoryKeyFactory`` signature so it can be
        passed directly as ``conversation_history_key_factory=self._conversation_history_key``.

        Args:
            _state: Current flow state (unused; present to satisfy the factory protocol).

        Returns:
            The resolved ``IOKey`` pointing to this component's conversation history slot.
        """
        return IOKey(
            target="conversation_history",
            subkeys=[self.name],
            optional=True,
        )

    def __entry_hook__(self) -> Annotated[str, "Entry node name"]:
        return f"{self.name}#agent"

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

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())

    # Private attributes for key factories with default values.
    # Overridden by bind_to_supervisor when used as a subagent.
    _conversation_history_key_factory: ConversationHistoryKeyFactory = PrivateAttr()
    _output_key_factory: OutputKeyFactory = PrivateAttr()
    _is_bound_to_supervisor: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def initialize_key_factories(self) -> Self:
        """Initialize key factories with default values for standalone use.

        These defaults are overridden by bind_to_supervisor when the component is used as a subagent.
        """
        # Default conversation history factory uses component-scoped key
        self._conversation_history_key_factory = self._conversation_history_key

        # Default output factory returns static component-scoped final_answer key
        output_key = self._final_answer_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )
        self._output_key_factory = lambda _: output_key

        return self

    def bind_to_supervisor(
        self,
        *,
        conversation_history_key_factory: ConversationHistoryKeyFactory,
        output_key_factory: OutputKeyFactory,
    ) -> None:
        """Bind this agent to a supervisor.

        Must be called before ``attach`` when using this component as a subagent.
        The supervisor passes subsession-scoped key factories so the agent never
        needs to scan state to discover which supervisor owns it or what the active
        subsession ID is.

        When bound to a supervisor, the description field must be set.

        Args:
            conversation_history_key_factory: Callable ``(state) -> IOKey`` that
                resolves the subsession-scoped conversation-history key at runtime.
            output_key_factory: Callable ``(state) -> IOKey`` that resolves the
                subsession-scoped final_answer key at runtime.

        Raises:
            ValueError: If description is not set when binding to supervisor.
        """
        if not self.description:
            raise ValueError(
                f"AgentComponent '{self.name}' must have a description when bound to a supervisor."
            )
        self._conversation_history_key_factory = conversation_history_key_factory
        self._output_key_factory = output_key_factory
        self._is_bound_to_supervisor = True

    def _agent_node_router(self, state: FlowState) -> str:
        history_iokey = self._conversation_history_key_factory(state)
        history: list[BaseMessage] = history_iokey.value_from_state(state) or []
        if not history:
            raise RoutingError(
                f"Conversation history not found for key "
                f"{history_iokey.target}:{history_iokey.subkeys}"
            )

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
            return f"{self.name}#final_response"

        if self._response_schema is not None and any(
            tool_call["name"] == self._response_schema.tool_title
            for tool_call in last_message.tool_calls
        ):
            return f"{self.name}#final_response"

        return f"{self.name}#tools"

    @override
    def __entry_hook__(self) -> Annotated[str, "Entry node name"]:
        return f"{self.name}#agent"

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

        prompt = self.prompt_registry.get_on_behalf(
            self.user,
            self.prompt_id,
            self.prompt_version,
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,
            internal_event_extra={
                "agent_name": self.name,
                "workflow_id": self.flow_id,
                "workflow_type": self.flow_type.value,
            },
        )

        node_agent = AgentNode(
            name=self.__entry_hook__(),
            conversation_history_key_factory=self._conversation_history_key_factory,
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
        node_tools = ToolNode(
            name=f"{self.name}#tools",
            conversation_history_key_factory=self._conversation_history_key_factory,
            toolset=self.toolset,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            ui_history=UIHistory(
                events=self.ui_log_events, writer_class=UILogWriterAgentTools
            ),
        )
        node_final_response = FinalResponseNode(
            name=f"{self.name}#final_response",
            conversation_history_key_factory=self._conversation_history_key_factory,
            output_key_factory=self._output_key_factory,
            ui_history=UIHistory(
                events=self.ui_log_events,
                writer_class=default_ui_log_writer_class(
                    events_class=UILogEventsAgent, ui_role_as=self.ui_role_as
                ),
            ),
            response_schema=self._response_schema,
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
