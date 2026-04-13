from functools import partial
from typing import Any, ClassVar, Self, override

from dependency_injector.wiring import inject
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import Field, model_validator

from duo_workflow_service.agent_platform.experimental.components import (
    RouterProtocol,
    RoutingError,
    register_component,
)
from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponentBase,
)
from duo_workflow_service.agent_platform.experimental.components.agent.nodes import (
    AgentNode,
)
from duo_workflow_service.agent_platform.experimental.components.one_off.nodes.tool_node_with_error_correction import (
    ATTEMPTS_REMAINING_SENTINEL,
    MAX_ATTEMPTS_SENTINEL,
    SUCCESS_SENTINEL,
    ToolNodeWithErrorCorrection,
)
from duo_workflow_service.agent_platform.experimental.components.one_off.ui_log import (
    UILogEventsOneOff,
    UILogWriterOneOffTools,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKeyTemplate,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.state import IOKey
from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)


@register_component(decorators=[inject])
class OneOffComponent(AgentComponentBase):
    _tool_calls_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "tool_calls"],
    )

    _tool_responses_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "tool_responses"],
    )

    _execution_result_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "execution_result"],
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(target="ui_chat_log"),
        IOKeyTemplate(
            target="conversation_history",
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE],
        ),
        _tool_calls_key,
        _tool_responses_key,
        _execution_result_key,
    )

    max_correction_attempts: int = 3

    ui_log_events: list[UILogEventsOneOff] = Field(default_factory=list)

    _allowed_input_targets = ("context", "conversation_history")

    @model_validator(mode="before")
    @classmethod
    def set_default_inputs(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "inputs" not in data or not data["inputs"]:
            data["inputs"] = ["context:goal"]
        return data

    @model_validator(mode="after")
    @override
    def validate_and_resolve_response_schema(self) -> Self:
        """No-op: OneOffComponent does not use response schemas."""
        return self

    @override
    def __entry_hook__(self) -> str:
        return f"{self.name}#llm"

    @override
    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        tools = self.toolset.bindable
        tool_choice = "auto"

        prompt = self.prompt_registry.get_on_behalf(
            self.user,
            self.prompt_id,
            self.prompt_version,
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,
            is_graph_node=True,
            internal_event_extra={
                "agent_name": self.name,
                "workflow_id": self.flow_id,
                "workflow_type": self.flow_type.value,
            },
        )

        # reuse existing agent_node
        agent_node = AgentNode(
            name=self.__entry_hook__(),
            conversation_history_key=self._default_conversation_history_key,
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
        )

        # Use enhanced tool node with error correction
        tracker = ToolEventTracker(
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
        )
        tool_node = ToolNodeWithErrorCorrection(
            name=f"{self.name}#tools",
            component_name=self.name,
            toolset=self.toolset,
            max_correction_attempts=self.max_correction_attempts,
            ui_history=UIHistory(
                events=self.ui_log_events,
                writer_class=UILogWriterOneOffTools,
            ),
            tracker=tracker,
            tool_calls_key=self._tool_calls_key.to_iokey(  # type: ignore[arg-type]
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            tool_responses_key=self._tool_responses_key.to_iokey(  # type: ignore[arg-type]
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            execution_result_key=self._execution_result_key.to_iokey(  # type: ignore[arg-type]
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=[self.name]
            ),
        )

        # Node 1: Agent Node
        graph.add_node(self.__entry_hook__(), agent_node.run)

        # Node 2: Tool execution with error correction
        graph.add_node(f"{self.name}#tools", tool_node.run)

        # Connect LLM node to tools node
        graph.add_edge(self.__entry_hook__(), f"{self.name}#tools")

        # Connect tools node with conditional routing for error correction
        graph.add_conditional_edges(
            f"{self.name}#tools", partial(self._tools_router, router)
        )

    def _tools_router(self, outgoing_router: RouterProtocol, state: FlowState) -> str:
        """Route based on tool execution results and correction attempts."""
        conversation = state.get(FlowStateKeys.CONVERSATION_HISTORY, {}).get(
            self.name, []
        )

        if not conversation:
            raise RoutingError(
                f"No conversation history found for component {self.name}. "
                f"Tool node should have added messages."
            )

        last_message = conversation[-1]

        if not last_message:
            return outgoing_router.route(state)

        # Check if it's a success message
        if (
            isinstance(last_message, HumanMessage)
            and SUCCESS_SENTINEL in last_message.content
        ):
            return outgoing_router.route(state)  # Success - exit component

        # Check if it's an error feedback message
        if (
            isinstance(last_message, HumanMessage)
            and ATTEMPTS_REMAINING_SENTINEL in last_message.content
        ):
            # Parse remaining attempts from the message
            if MAX_ATTEMPTS_SENTINEL in last_message.content:
                return outgoing_router.route(
                    state
                )  # Max attempts reached - exit component
            return self.__entry_hook__()  # Error with attempts remaining - retry

        # If we can't parse then raise error
        raise RoutingError(
            f"Unable to route based on last message content: {last_message.content}"
        )
