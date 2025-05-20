"""Module containing RunToolNode class for executing tools with input and output parsing."""

from datetime import datetime, timezone
from typing import Any, Generic, Protocol, TypeVar

from langchain.tools import BaseTool

from duo_workflow_service.entities import MessageTypeEnum, ToolStatus, UiChatLog
from duo_workflow_service.entities.state import (
    SearchAndReplaceWorkflowState,
    ToolInfo,
    WorkflowState,
)
from duo_workflow_service.monitoring import duo_workflow_metrics

WorkflowStateT_contra = TypeVar(
    "WorkflowStateT_contra",
    SearchAndReplaceWorkflowState,
    WorkflowState,
    contravariant=True,
)


class InputParserProtocol(Protocol[WorkflowStateT_contra]):
    """Protocol for input parser functions that prepare tool parameters from state."""

    def __call__(self, state: WorkflowStateT_contra) -> list[dict[str, Any]]: ...


class OutputParserProtocol(Protocol[WorkflowStateT_contra]):
    """Protocol for output parser functions that process tool outputs and update state."""

    def __call__(
        self, outputs: list[Any], state: WorkflowStateT_contra
    ) -> dict[str, Any]: ...


WorkflowStateT = TypeVar("WorkflowStateT", SearchAndReplaceWorkflowState, WorkflowState)


class RunToolNode(Generic[WorkflowStateT]):
    """A node class that executes a tool with input and output parsing capabilities."""

    _input_parser: InputParserProtocol[WorkflowStateT]
    _output_parser: OutputParserProtocol[WorkflowStateT]
    _tool: BaseTool

    def __init__(
        self,
        tool: BaseTool,
        input_parser: InputParserProtocol[WorkflowStateT],
        output_parser: OutputParserProtocol[WorkflowStateT],
    ):
        """Initialize the RunToolNode.

        Args:
            tool: The tool to execute
            input_parser: Function that converts state into tool parameters
            output_parser: Function that processes tool outputs and updates state
        """
        self._tool = tool
        self._input_parser = input_parser
        self._output_parser = output_parser

    async def run(self, state: WorkflowStateT) -> dict[str, Any]:
        """Execute the tool with given state.

        Args:
            state: Current workflow state

        Returns:
            Updated state dictionary
        """
        outputs = []
        logs = []

        for tool_params in self._input_parser(state):
            with duo_workflow_metrics.time_tool_call(tool_name=self._tool.name):
                output = await self._tool._arun(**tool_params)
            outputs.append(output)
            logs.append(
                UiChatLog(
                    message_type=MessageTypeEnum.TOOL,
                    content=f"Run tool {self._tool.name} with params {tool_params}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=ToolInfo(name=self._tool.name, args=tool_params),
                    context_elements=None,
                )
            )

        return {"ui_chat_log": logs, **self._output_parser(outputs, state)}
