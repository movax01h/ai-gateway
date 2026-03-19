from typing import Annotated, ClassVar, Literal, Optional, Type, override

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from pydantic import Field, PrivateAttr, model_validator

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from ai_gateway.response_schemas.registry import BaseAgentOutput
from duo_workflow_service.agent_platform.v1.components.agent.nodes import (
    AgentNode,
    FinalResponseNode,
    ToolNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.components.registry import (
    register_component,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.v1.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events import InternalEventsClient

__all__ = ["AgentComponent", "RoutingError"]


class RoutingError(Exception):
    """Exception raised when edge routers encounter unexpected conditions."""


@register_component(decorators=[inject])
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

    supported_environments: ClassVar[tuple[str, ...]] = (
        "ambient",
        "chat",
        "chat-partial",
    )

    prompt_id: str
    prompt_version: Optional[str] = None
    toolset: Toolset
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

    ui_log_events: list[UILogEventsAgent] = Field(default_factory=list)
    ui_role_as: Literal["agent", "tool"] = "agent"

    _allowed_input_targets = tuple(FlowState.__annotations__.keys())
    _response_schema: Optional[Type[BaseAgentOutput]] = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_resolve_response_schema(self):
        """Validate response schema params and resolve the schema.

        This validator:
        1. Validates that response_schema_id and response_schema_version are either both set or both None
        2. Resolves the response schema from the registry (or uses None for default text-only mode)
        3. Validates that the schema tool name doesn't collide with existing tools
        """
        if bool(self.response_schema_id) != bool(self.response_schema_version):
            raise ValueError(
                "response_schema_id and response_schema_version must be provided together. "
                "Either provide both or omit both for default text-only mode."
            )

        # Resolve the response schema
        if self.response_schema_id and self.response_schema_version:
            response_schema = self.schema_registry.get(
                schema_id=self.response_schema_id,
                schema_version=self.response_schema_version,
            )
            # Check to see if name of schema tool collides with existing tools
            if response_schema.tool_title in self.toolset:
                raise ValueError(
                    f"Response schema tool title '{response_schema.tool_title}' "
                    f"collides with existing tool: '{response_schema.tool_title}'"
                )
        else:
            response_schema = None

        self._response_schema = response_schema
        return self

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

        For custom response schemas, this includes both the base final_answer output
        and individual field paths for each schema field, enabling easier discovery
        and access to nested data.

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
            component_name=self.name,
            prompt=prompt,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            response_schema=self._response_schema,
        )
        node_tools = ToolNode(
            name=f"{self.name}#tools",
            component_name=self.name,
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
            component_name=self.name,
            output=self._final_answer_key.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
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
