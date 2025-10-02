import copy
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import ValidationError

from ai_gateway.container import ContainerApplication
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    Plan,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import (
    PromptSecurity,
    SecurityException,
)
from duo_workflow_service.tools import RunCommand, Toolset, format_tool_display_message
from duo_workflow_service.tools.planner import PlannerTool
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventLabelEnum

_HIDDEN_TOOLS = ["get_plan"]

_ACTION_HANDLERS = [
    "add_new_task",
    "remove_task",
    "update_task_description",
    "set_task_status",
    "create_plan",
]

_COMMAND_OUTPUT_TOOLS = {
    "run_command": RunCommand,
}

# Display only first 4KB of a tool response on UI to avoid duplicating large responses twice in a checkpoint
TOOL_RESPONSE_MAX_DISPLAY_MSG = 4 * 1024


class IncompleteToolCallDueToMaxTokens(ToolException):
    """Raised when a tool call is incomplete, e.g., due to streaming ending due to max_tokens."""


class ToolsExecutor:
    _tools_agent_name: str
    _toolset: Toolset

    @inject
    def __init__(
        self,
        tools_agent_name: str,
        toolset: Toolset,
        workflow_id: str,
        workflow_type: CategoryEnum,
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
    ) -> None:
        self._tools_agent_name = tools_agent_name
        self._toolset = toolset
        self._workflow_id = workflow_id
        self._logger = structlog.stdlib.get_logger("workflow")
        self._workflow_type = workflow_type
        self._internal_event_client = internal_event_client

    async def run(self, state: DuoWorkflowStateType):
        last_message = state["conversation_history"][self._tools_agent_name][-1]
        tool_calls: list[ToolCall] = getattr(last_message, "tool_calls", [])
        state_updates = {}
        responses: list[dict[str, Any] | Command] = []
        ui_chat_logs: List[UiChatLog] = []
        plan = state.get("plan", {"steps": []})

        self._create_ai_message_ui_chat_log(last_message, ui_chat_logs)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in self._toolset:
                responses.append(
                    self._process_response(tool_call, f"Tool {tool_name} not found")
                )
                continue

            result = await self._execute_tool(
                tool_name,
                tool_call,
                plan,
                last_message.response_metadata.get("stop_reason"),
            )
            response = result.get("response")
            if response and hasattr(response, "content"):
                try:
                    result["response"].content = (
                        PromptSecurity.apply_security_to_tool_response(
                            response=result["response"].content,
                            tool_name=tool_name,
                        )
                    )
                except SecurityException as e:
                    self._logger.error(
                        f"Security validation failed for tool {tool_name}: {e}"
                    )
                    raise

            chat_logs = result.get("chat_logs", [])
            if chat_logs and isinstance(chat_logs[0], dict):
                chat_logs[0].setdefault("message_sub_type", tool_name)

            if tool_name in _COMMAND_OUTPUT_TOOLS:
                if chat_logs and "tool_info" in chat_logs[0]:
                    chat_log = chat_logs[0]
                    cleaned_response = self._clean_run_command_response(response)
                    chat_log["tool_info"]["tool_response"] = cleaned_response
                    chat_log["message_sub_type"] = "command_output"
                    ui_chat_logs.extend([chat_log])
            else:
                ui_chat_logs.extend(result.get("chat_logs", []))

            responses.append(self._process_response(tool_call, result["response"]))

            if result.get("status") == WorkflowStatusEnum.ERROR:
                state_updates["status"] = WorkflowStatusEnum.ERROR
                break

        responses.append(
            Command(
                update={
                    "ui_chat_log": ui_chat_logs,
                    **state_updates,
                }
            )
        )

        return responses

    def _process_response(self, tool_call, response) -> Command | dict[str, Any]:
        if isinstance(response, Command):
            return response

        if isinstance(response, str):
            response = ToolMessage(content=response, tool_call_id=tool_call.get("id"))

        return {"conversation_history": {self._tools_agent_name: [response]}}

    def _create_ai_message_ui_chat_log(
        self, message: BaseMessage, ui_chat_logs: List[UiChatLog]
    ):
        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls and all(
            tool_call["name"] in _HIDDEN_TOOLS for tool_call in tool_calls
        ):
            return

        ai_message_content = self._extract_ai_message_text(message)

        if ai_message_content:
            ui_chat_logs.append(
                UiChatLog(
                    message_type=MessageTypeEnum.AGENT,
                    message_sub_type=None,
                    content=ai_message_content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=None,
                    additional_context=None,
                )
            )

    def _add_tool_ui_chat_log(
        self,
        tool_info: Dict[str, Any],
        status: ToolStatus,
        ui_chat_logs: List[UiChatLog],
        error_message: Optional[str] = None,
        tool_response: Optional[Any] = None,
    ):
        chat_log = self._create_tool_ui_chat_log(
            tool_name=tool_info["name"],
            tool_args=tool_info["args"],
            status=status,
            error_message=error_message,
            tool_response=tool_response,
        )
        if chat_log:
            ui_chat_logs.append(chat_log)

    async def _execute_tool(
        self,
        tool_name: str,
        tool_call: ToolCall,
        plan: Plan,
        stop_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        tool_args = tool_call.get("args", {})
        tool = self._toolset[tool_name]
        chat_logs: List[UiChatLog] = []

        if isinstance(tool, PlannerTool):
            tool.plan = copy.deepcopy(plan)
            tool.tools_agent_name = self._tools_agent_name
            tool.tool_call_id = tool_call["id"]

        try:
            with duo_workflow_metrics.time_tool_call(
                tool_name=tool_name, flow_type=self._workflow_type.value
            ):
                if stop_reason == "max_tokens":
                    raise IncompleteToolCallDueToMaxTokens(
                        f"Max tokens reached for tool {tool_name}."
                        " Try a simpler request or using a different tool."
                    )

                tool_response = await tool.ainvoke(tool_call)

            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool_name,
            )

            self._add_tool_ui_chat_log(
                tool_info={"name": tool_name, "args": tool_args},
                status=ToolStatus.SUCCESS,
                ui_chat_logs=chat_logs,
                tool_response=tool_response,
            )

            return {
                "response": tool_response,
                "chat_logs": chat_logs,
            }

        except TypeError as error:
            tool_context = {"tool": tool, "name": tool_name, "args": tool_args}
            return self._handle_type_error(tool_context, error, chat_logs)

        except ValidationError as error:
            return self._handle_validation_error(tool_name, tool_args, error, chat_logs)

        except ToolException as error:
            return self._handle_tool_error(tool_name, tool_args, error, chat_logs)

    def _handle_type_error(
        self,
        tool_context: Dict[str, Any],
        error: TypeError,
        chat_logs: List[UiChatLog],
    ) -> Dict[str, Any]:
        # log the error itself to check if the TypeError is indeed
        # a schema error.
        self._logger.error(f"Tools executor raised TypeError {error}")

        tool = tool_context["tool"]
        tool_name = tool_context["name"]
        tool_args = tool_context["args"]

        schema = (
            f"The schema is: {tool.args_schema.model_json_schema()}"
            if tool.args_schema
            else "The tool does not accept any argument"
        )

        tool_response = (
            f"Tool {tool_name} execution failed due to wrong arguments. You must adhere to the tool args "
            f"schema! {schema}"
        )
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        self._add_tool_ui_chat_log(
            tool_info={"name": tool_name, "args": tool_args},
            status=ToolStatus.FAILURE,
            ui_chat_logs=chat_logs,
            error_message="Invalid arguments",
        )

        return {
            "response": tool_response,
            "chat_logs": chat_logs,
        }

    def _handle_validation_error(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        error: ValidationError,
        chat_logs: List[UiChatLog],
    ) -> Dict[str, Any]:
        tool_response = f"Tool {tool_name} raised validation error {error}"
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        self._add_tool_ui_chat_log(
            tool_info={"name": tool_name, "args": tool_args},
            status=ToolStatus.FAILURE,
            ui_chat_logs=chat_logs,
            error_message="Validation error",
        )

        return {
            "response": tool_response,
            "chat_logs": chat_logs,
        }

    def _handle_tool_error(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        error: ToolException,
        chat_logs: List[UiChatLog],
    ) -> Dict[str, Any]:
        error_type = type(error).__name__

        self._logger.error(f"Tools executor raised error {error_type} {error}")

        tool_response = f"Tool {tool_name} raised ToolException: {str(error)}"
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        self._add_tool_ui_chat_log(
            tool_info={"name": tool_name, "args": tool_args},
            status=ToolStatus.FAILURE,
            ui_chat_logs=chat_logs,
            error_message=f"Tool call failed: {error_type}",
        )

        return {
            "response": tool_response,
            "chat_logs": chat_logs,
        }

    def _track_internal_event(
        self,
        event_name: EventEnum,
        tool_name,
        extra=None,
    ):
        if extra is None:
            extra = {}
        additional_properties = InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
            property=tool_name,
            value=self._workflow_id,
            **extra,
        )
        self._record_metric(
            event_name=event_name,
            additional_properties=additional_properties,
        )
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._workflow_type.value,
        )

    def _create_tool_ui_chat_log(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status: ToolStatus = ToolStatus.SUCCESS,
        error_message: Optional[str] = None,
        tool_response: Optional[Any] = None,
    ) -> Optional[UiChatLog]:
        display_message = self.get_tool_display_message(
            tool_name, tool_args, tool_response
        )

        if not display_message:
            return None

        content = display_message
        if error_message:
            content = f"Failed: {display_message} - {error_message}"

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_sub_type=tool_name,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            correlation_id=None,
            tool_info=(
                (
                    ToolInfo(
                        name=tool_name,
                        args=tool_args,
                        tool_response=ToolMessage(
                            content=tool_response.content[
                                :TOOL_RESPONSE_MAX_DISPLAY_MSG
                            ],
                            name=tool_response.name,
                            tool_call_id=tool_response.tool_call_id,
                        ),
                    )
                    if tool_response is not None
                    else ToolInfo(name=tool_name, args=tool_args)
                )
                if tool_name not in _ACTION_HANDLERS
                else None
            ),
            additional_context=None,
        )

    def get_tool_display_message(
        self, tool_name: str, args: Dict[str, Any], tool_response: Any = None
    ) -> Optional[str]:
        if tool_name in _HIDDEN_TOOLS:
            return None

        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        message = f"Using {tool_name}: {args_str}"

        if tool_name in self._toolset:
            tool = self._toolset[tool_name]
            message = (
                format_tool_display_message(tool, args, tool_response or "") or message
            )

        return message

    def _extract_ai_message_text(self, last_message: BaseMessage):
        if isinstance(last_message, AIMessage):
            return StrOutputParser().invoke(last_message)

        return None

    def _record_metric(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:

        if event_name == EventEnum.WORKFLOW_TOOL_FAILURE:
            tool_name = additional_properties.property or "unknown"
            failure_reason = additional_properties.extra.get("error_type", "unknown")
            duo_workflow_metrics.count_agent_platform_tool_failure(
                flow_type=self._workflow_type.value,
                tool_name=tool_name,
                failure_reason=failure_reason,
            )

    def _clean_run_command_response(self, tool_response: Any) -> Any:
        """Extract clean output from run_command tool errors."""
        if not hasattr(tool_response, "content"):
            return tool_response

        content = tool_response.content

        if not isinstance(content, str):
            return tool_response

        # Node.js pattern: "Error running tool: Process exited with code X. Output: Y"
        nodejs_match = re.search(
            r"Error running tool: Process exited with code \d+\. Output: (.*)$",
            content,
            re.DOTALL,
        )
        if nodejs_match:
            tool_response.content = nodejs_match.group(1).strip()
            return tool_response

        # Go pattern: "Error running tool: exit status X. Result: Y"
        go_match = re.search(
            r"Error running tool: exit status \d+\. Result: (.*)$",
            content,
            re.DOTALL,
        )
        if go_match:
            tool_response.content = go_match.group(1).strip()
            return tool_response

        return tool_response
