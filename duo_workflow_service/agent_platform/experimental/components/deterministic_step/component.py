from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from dependency_injector.wiring import inject
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from duo_workflow_service.agent_platform.experimental.components import (
    register_component,
)
from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
    get_vars_from_state,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.security.prompt_security import PromptSecurity
from duo_workflow_service.tools import DuoBaseTool
from duo_workflow_service.tools.toolset import Toolset

__all__ = ["DeterministicStepComponent"]


@register_component(decorators=[inject])
class DeterministicStepComponent(BaseComponent):
    _tool_result_key: ClassVar[IOKeyTemplate] = IOKeyTemplate(
        target="context",
        subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, "tool_result"],
    )

    _outputs: ClassVar[tuple[IOKeyTemplate, ...]] = (
        IOKeyTemplate(target="ui_chat_log"),
        _tool_result_key,
    )

    tool_name: str
    toolset: Toolset

    _allowed_input_targets: ClassVar[tuple[str, ...]] = (
        "context",
        "conversation_history",
    )

    ui_log_events: list[UILogEventsAgent] = Field(default_factory=list)
    ui_role_as: Literal["tool"] = "tool"

    def __entry_hook__(self) -> str:
        return f"{self.name}#deterministic_step"

    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        graph.add_node(self.__entry_hook__(), self._execute_tool)
        graph.add_conditional_edges(self.__entry_hook__(), router.route)

    # Component outputs follow fixed pattern with dynamic naming
    # Example: context:read_config.tool_result, context:scan_files.tool_result
    def get_output_key(self) -> IOKey:
        return self._tool_result_key.to_iokey(
            {IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self.name}
        )

    async def _execute_tool(self, state: FlowState) -> dict[str, Any]:
        """Execute the appointed tool with extracted parameters from state.

        Args:
            state: Current flow state

        Returns:
            Dictionary with ui_chat_log and context updates
        """
        try:
            # Extract tool parameters from component inputs
            variables = get_vars_from_state(self.inputs, state)

            # Get appointed tool from toolset
            if self.tool_name not in self.toolset:
                raise KeyError(f"Tool '{self.tool_name}' not found in toolset")

            tool = self.toolset[self.tool_name]

            tool_response = await tool._arun(**variables)

            secure_result = PromptSecurity.apply_security_to_tool_response(
                response=tool_response, tool_name=self.tool_name
            )

            success_log = UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content=self._format_message(tool, variables, tool_response),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=ToolInfo(name=tool.name, args=variables),
                message_sub_type=tool.name,
                correlation_id=None,
                additional_context=None,
            )

            return {
                "ui_chat_log": [success_log],
                "context": {self.name: {"tool_result": secure_result}},
            }

        except Exception as e:
            error_log = UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                message_sub_type=None,
                content=f"Tool {self.tool_name} execution failed: {str(e)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.FAILURE,
                correlation_id=None,
                tool_info=ToolInfo(name=self.tool_name, args={}),
                additional_context=None,
            )

            return {
                "ui_chat_log": [error_log],
                "context": {self.name: {"tool_result": None, "error": str(e)}},
            }

    @staticmethod
    def _format_message(
        tool: BaseTool, tool_call_args: dict[str, Any], tool_response: Any = None
    ) -> str:
        if not hasattr(tool, "format_display_message"):
            args_str = ", ".join(f"{k}={str(v)}" for k, v in tool_call_args.items())
            return f"Using {tool.name}: {args_str}"

        try:
            schema = getattr(tool, "args_schema", None)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # type: ignore[arg-type]
                parsed = schema(**tool_call_args)
                return tool.format_display_message(parsed, tool_response)
        except Exception:
            return DuoBaseTool.format_display_message(
                tool, tool_call_args, tool_response  # type: ignore[arg-type]
            )  # type: ignore[return-value]

        return tool.format_display_message(tool_call_args, tool_response)
