import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from langchain.tools import BaseTool
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    Plan,
    Task,
    TaskStatus,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.internal_events import (
    DuoWorkflowInternalEvent,
    InternalEventAdditionalProperties,
)
from duo_workflow_service.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
)
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tools import PipelineException, format_tool_display_message
from duo_workflow_service.tools.planner import format_task_number

_HIDDEN_TOOLS = ["get_plan"]


def _add_new_task(args: dict, plan: Plan) -> tuple[Plan, str]:
    new_task: Task = {
        "id": f"task-{len(plan['steps'])}",
        "description": args["description"],
        "status": TaskStatus.NOT_STARTED,
    }
    plan["steps"].append(new_task)
    return plan, f"Step added: {new_task['id']}"


def _remove_task(args: dict, plan: Plan) -> tuple[Plan, str]:
    plan["steps"] = [step for step in plan["steps"] if step["id"] != args["task_id"]]
    return plan, f"Task removed: {args['task_id']}"


def _update_task_description(args: dict, plan: Plan) -> tuple[Plan, str]:
    task_id = args.get("task_id")
    new_description = args.get("new_description")

    if task_id is None:
        return plan, "No task_id provided"

    for step in plan["steps"]:
        if step["id"] == task_id:
            if new_description:
                step["description"] = new_description
            return plan, f"Task updated: {task_id}"

    return plan, f"Task not found: {task_id}"


def _set_task_status(args: dict, plan: Plan) -> tuple[Plan, str]:
    for step in plan["steps"]:
        if step["id"] == args["task_id"]:
            step["status"] = TaskStatus(args["status"])
            return (
                plan,
                f"Task status set: {args['task_id']} - {args['status']}",
            )
    return plan, f"Task not found: {args['task_id']}"


_ACTION_HANDLERS = {
    "add_new_task": _add_new_task,
    "remove_task": _remove_task,
    "update_task_description": _update_task_description,
    "set_task_status": _set_task_status,
}


class ToolsExecutor:
    _tools_agent_name: str
    _tool_lookup: Dict[str, BaseTool]

    def __init__(
        self,
        tools_agent_name: str,
        agent_tools: list[BaseTool],
        workflow_id: str,
        workflow_type: CategoryEnum,
    ) -> None:
        self._tools_agent_name = tools_agent_name
        self._tool_lookup = dict(list(map(lambda x: (x.name, x), agent_tools)))
        self._workflow_id = workflow_id
        self._logger = structlog.stdlib.get_logger("workflow")
        self._workflow_type = workflow_type

    async def run(self, state: DuoWorkflowStateType):
        last_message = state["conversation_history"][self._tools_agent_name][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        tools_responses = []
        ui_chat_logs: List[UiChatLog] = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_response = f"Tool {tool_name} not found"
            tool_args = tool_call.get("args", {})

            if tool_name in _ACTION_HANDLERS:
                new_plan, tool_response = await self._handle_plan_modification(
                    tool_name, tool_args, state["plan"]
                )
                state["plan"] = new_plan

                ui_chat_log = self._create_ui_chat_log(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    status=ToolStatus.SUCCESS,
                )
                if ui_chat_log:
                    ui_chat_logs.append(ui_chat_log)

            if tool_name in self._tool_lookup:
                tool = self._tool_lookup[tool_name]

                try:
                    with duo_workflow_metrics.time_tool_call(tool_name=tool_name):
                        tool_response = await tool.arun(tool_args)

                    self._track_internal_event(
                        event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                        tool_name=tool_name,
                    )

                    ui_chat_log = self._create_ui_chat_log(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status=ToolStatus.SUCCESS,
                    )
                    if ui_chat_log and tool_name not in _ACTION_HANDLERS:
                        ui_chat_logs.append(ui_chat_log)

                except TypeError as error:
                    # log the error itself to check if the TypeError is indeed
                    # a schema error.
                    self._logger.error(f"Tools executor raised TypeError {error}")

                    schema = (
                        f"The schema is: {tool.args_schema.model_json_schema()}"
                        if tool.args_schema
                        else "The tool does not accept any argument"
                    )

                    tool_response = f"Tool {tool_name} execution failed due to wrong arguments. You must adhere to the tool args schema! {schema}"
                    self._track_internal_event(
                        event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
                        tool_name=tool_name,
                        extra={"error": str(error)},
                    )

                    ui_chat_log = self._create_ui_chat_log(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status=ToolStatus.FAILURE,
                        error_message="Invalid arguments",
                    )
                    if ui_chat_log:
                        ui_chat_logs.append(ui_chat_log)

                except ValidationError as error:
                    tool_response = f"Tool {tool_name} raised validation error {error}"
                    self._track_internal_event(
                        event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
                        tool_name=tool_name,
                        extra={"error": str(error)},
                    )

                    ui_chat_log = self._create_ui_chat_log(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status=ToolStatus.FAILURE,
                        error_message="Validation error",
                    )
                    if ui_chat_log:
                        ui_chat_logs.append(ui_chat_log)

                except PipelineException as error:
                    tool_response = f"Pipeline exception due to {error}"
                    tools_responses.append(
                        ToolMessage(
                            content=str(tool_response), tool_call_id=tool_call.get("id")
                        )
                    )
                    self._track_internal_event(
                        event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
                        tool_name=tool_name,
                        extra={"error": str(error)},
                    )

                    ui_chat_log = self._create_ui_chat_log(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status=ToolStatus.FAILURE,
                        error_message=f"Pipeline error: {error}",
                    )
                    if ui_chat_log:
                        ui_chat_logs.append(ui_chat_log)

                    return {
                        "conversation_history": {
                            self._tools_agent_name: tools_responses
                        },
                        "status": WorkflowStatusEnum.ERROR,
                        "ui_chat_log": ui_chat_logs,
                    }

            if tool_name == "get_plan":
                tool_response = json.dumps(state["plan"]["steps"])

            tools_responses.append(
                ToolMessage(content=tool_response, tool_call_id=tool_call.get("id"))
            )

        return {
            "conversation_history": {self._tools_agent_name: tools_responses},
            "plan": state["plan"],
            "ui_chat_log": ui_chat_logs,
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
        DuoWorkflowInternalEvent.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._workflow_type.value,
        )

    async def _handle_plan_modification(
        self, tool_name: str, args: dict, plan: Plan
    ) -> tuple[Plan, str]:
        handler = _ACTION_HANDLERS.get(tool_name)
        if handler:
            new_plan, response = handler(args, deepcopy(plan))
            return new_plan, response
        return plan, "Error handling plan modification"

    def _create_ui_chat_log(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status: ToolStatus = ToolStatus.SUCCESS,
        error_message: Optional[str] = None,
    ) -> Optional[UiChatLog]:
        display_message = self.get_tool_display_message(tool_name, tool_args)

        if not display_message:
            return None

        content = display_message
        if error_message:
            content = f"Failed: {display_message} - {error_message}"

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            correlation_id=None,
            tool_info=(
                ToolInfo(name=tool_name, args=tool_args)
                if status != ToolStatus.SUCCESS or tool_name not in _ACTION_HANDLERS
                else None
            ),
        )

    def get_tool_display_message(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[str]:
        if tool_name in _HIDDEN_TOOLS:
            return None

        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        message = f"Using {tool_name}: {args_str}"

        if tool_name in _ACTION_HANDLERS:
            task_num = format_task_number(args.get("task_id", ""))
            action_messages = {
                "add_new_task": f"Add new task to the plan: {args.get('description', '')[:100]}...",
                "remove_task": f"Remove task {task_num}",
                "update_task_description": f"Update description for task {task_num}",
                "set_task_status": f"Set task {task_num} to '{args.get('status', '')}'",
            }
            message = action_messages.get(tool_name, "")

        elif tool_name in self._tool_lookup:
            tool = self._tool_lookup[tool_name]
            message = format_tool_display_message(tool, args) or message

        return message
