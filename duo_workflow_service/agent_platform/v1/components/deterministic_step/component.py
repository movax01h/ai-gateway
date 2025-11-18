from typing import Any, ClassVar, Literal

from dependency_injector.wiring import Provide, inject
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from pydantic import Field, model_validator

from ai_gateway.container import ContainerApplication
from duo_workflow_service.agent_platform.v1.components import register_component
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.components.deterministic_step.nodes import (
    DeterministicStepNode,
)
from duo_workflow_service.agent_platform.v1.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
    UILogWriterDeterministicStep,
)
from duo_workflow_service.agent_platform.v1.state import IOKey, IOKeyTemplate
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.tools.toolset import Toolset

__all__ = ["DeterministicStepComponent"]

from lib.internal_events import InternalEventsClient


@register_component(decorators=[inject])
class DeterministicStepComponent(BaseComponent):
    _tool_responses_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "tool_responses"],
    )
    _tool_error_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "error"],
    )
    _execution_result_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "execution_result"],
    )
    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(target="ui_chat_log"),
        _tool_responses_key,
        _tool_error_key,
        _execution_result_key,
    )

    internal_event_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ]

    tool_name: str
    toolset: Toolset

    _allowed_input_targets: ClassVar[tuple[str, ...]] = (
        "context",
        "conversation_history",
    )

    ui_log_events: list[UILogEventsDeterministicStep] = Field(default_factory=list)
    ui_role_as: Literal["tool"] = "tool"

    validated_tool: BaseTool = Field(init=False)

    @model_validator(mode="before")
    @classmethod
    def validate_tool_configuration(cls, data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            tool_name = data.get("tool_name")
            toolset = data.get("toolset")

            raw_inputs = data.get("inputs", [])
            inputs = IOKey.parse_keys(raw_inputs)

            if not tool_name:
                raise ValueError("tool_name is required")

            if not toolset:
                raise ValueError("toolset is required")

            if tool_name not in toolset:
                available_tools = list(toolset.keys())
                raise KeyError(
                    f"Tool '{tool_name}' not found in toolset. "
                    f"Available tools: {available_tools}"
                )

            tool = toolset[tool_name]

            if tool.args_schema:
                error = cls._validate_tool_arguments(tool, inputs)
                if error:
                    schema = tool.args_schema.model_json_schema()
                    raise ValueError(
                        f"Tool '{tool_name}' configuration validation failed:\n"
                        f"Error: {error}\n"
                        f"Expected schema: {schema}"
                    )

            data["validated_tool"] = tool

        return data

    @classmethod
    def _validate_tool_arguments(
        cls, tool: BaseTool, inputs: list[IOKey]
    ) -> str | None:
        if not tool.args_schema:
            return None

        try:
            # Get expected parameters from schema
            schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
            expected_params = set(schema.get("properties", {}).keys())
            required_params = set(schema.get("required", []))

            # Extract configured parameter names
            configured_params = set()
            for input_key in inputs:
                if input_key.alias:
                    param_name = input_key.alias
                elif input_key.subkeys:
                    param_name = input_key.subkeys[-1]
                else:
                    param_name = str(input_key)
                configured_params.add(param_name)

            # Check for missing required parameters
            missing_required = required_params - configured_params
            if missing_required:
                return f"Missing required parameters: {sorted(missing_required)}"

            # Check for unknown parameters
            unknown_params = configured_params - expected_params
            if unknown_params:
                return f"Unknown parameters: {sorted(unknown_params)}. Valid parameters are: {sorted(expected_params)}"

            return None

        except Exception as e:
            return f"Validation error: {str(e)}"

    def __entry_hook__(self) -> str:
        return f"{self.name}#deterministic_step"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        node = DeterministicStepNode(
            name=self.__entry_hook__(),
            tool_name=self.tool_name,
            inputs=self.inputs,
            flow_id=self.flow_id,
            flow_type=self.flow_type,
            internal_event_client=self.internal_event_client,
            ui_history=UIHistory(
                events=self.ui_log_events, writer_class=UILogWriterDeterministicStep
            ),
            tool_responses_key=self._tool_responses_key.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            tool_error_key=self._tool_error_key.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            execution_result_key=self._execution_result_key.to_iokey(
                {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
            ),
            validated_tool=self.validated_tool,
        )

        graph.add_node(self.__entry_hook__(), node.run)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)
