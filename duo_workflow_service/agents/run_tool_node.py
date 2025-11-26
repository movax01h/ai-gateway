"""Module containing RunToolNode class for executing tools with input and output parsing."""

from datetime import datetime, timezone
from typing import Any, Generic, Protocol, TypeVar

import structlog
from langchain.tools import BaseTool

from duo_workflow_service.entities import MessageTypeEnum, ToolStatus, UiChatLog
from duo_workflow_service.entities.state import ToolInfo, WorkflowState
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import (
    PromptSecurity,
    SecurityException,
)
from duo_workflow_service.tracking.errors import log_exception
from lib.internal_events.event_enum import CategoryEnum

WorkflowStateT_contra = TypeVar(
    "WorkflowStateT_contra",
    bound=WorkflowState,
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


WorkflowStateT = TypeVar("WorkflowStateT", bound=WorkflowState)


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
        flow_type: CategoryEnum,
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
        self._logger = structlog.stdlib.get_logger("workflow")
        self._flow_type = flow_type

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
            with duo_workflow_metrics.time_tool_call(
                tool_name=self._tool.name, flow_type=self._flow_type.value
            ):
                if output := await self._tool._arun(**tool_params):
                    try:
                        secure_output = PromptSecurity.apply_security_to_tool_response(
                            response=output,
                            tool_name=self._tool.name,
                        )
                        output = secure_output
                    except SecurityException as e:
                        log_exception(
                            e,
                            extra={
                                "context": "Security validation failed for tool",
                                "tool_name": self._tool.name,
                            },
                        )
                        raise

            outputs.append(output)
            logs.append(
                UiChatLog(
                    message_type=MessageTypeEnum.TOOL,
                    message_sub_type=None,
                    content=f"Run tool {self._tool.name} with params {tool_params}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=ToolInfo(name=self._tool.name, args=tool_params),
                    additional_context=None,
                    message_id=None,
                )
            )

        return {"ui_chat_log": logs, **self._output_parser(outputs, state)}
