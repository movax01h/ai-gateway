# pylint: disable=direct-environment-variable-reference

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Type
from unittest.mock import MagicMock, call, patch

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from duo_workflow_service.agents import ToolsExecutor
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    TaskStatus,
    ToolInfo,
    WorkflowStatusEnum,
)
from duo_workflow_service.internal_events import InternalEventAdditionalProperties
from duo_workflow_service.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
)
from duo_workflow_service.tools import PipelineMergeRequestNotFoundError, Toolset
from duo_workflow_service.tools.planner import (
    AddNewTaskInput,
    CreatePlanInput,
    RemoveTaskInput,
    SetTaskStatusInput,
    UpdateTaskDescriptionInput,
)


def mock_tool(name="test_tool", side_effect=None, args_schema=None):
    mock = MagicMock(BaseTool)
    mock.name = name
    mock.args_schema = args_schema
    if side_effect:
        mock.arun.side_effect = side_effect
    else:
        mock.arun.return_value = "test_tool result"
    return mock


@dataclass
class ToolTestCase:
    tool_calls: List[Dict]
    tools: Dict[MagicMock, bool]
    tools_response: List[ToolMessage]
    ai_content: Any = field(
        default_factory=lambda: [
            {
                "type": "text",
                "text": "I'll search for issues related to this repository",
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        ToolTestCase(
            tool_calls=[
                {
                    "id": "1",
                    "name": mock_tool().name,
                    "args": {"tasks": [{"description": "step1"}]},
                }
            ],
            tools={mock_tool(): True},
            tools_response=[ToolMessage(content="test_tool result", tool_call_id="1")],
        ),
        ToolTestCase(
            tool_calls=[
                {
                    "id": "1",
                    "name": "does_not_exist",
                    "args": {"summary": "done"},
                }
            ],
            tools={mock_tool(): False},
            tools_response=[
                ToolMessage(content="Tool does_not_exist not found", tool_call_id="1")
            ],
        ),
        ToolTestCase(
            tool_calls=[
                {
                    "id": "1",
                    "name": "does_not_exist",
                    "args": {"summary": "done"},
                },
                {
                    "id": "2",
                    "name": mock_tool().name,
                    "args": {"tasks": [{"description": "step1"}]},
                },
            ],
            tools={mock_tool(): True, mock_tool(name="other_tool"): False},
            tools_response=[
                ToolMessage(content="Tool does_not_exist not found", tool_call_id="1"),
                ToolMessage(content="test_tool result", tool_call_id="2"),
            ],
        ),
    ],
)
@patch("duo_workflow_service.agents.tools_executor.datetime")
@patch("duo_workflow_service.agents.tools_executor.DuoWorkflowInternalEvent")
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_run(
    mock_internal_event_tracker,
    mock_datetime,
    workflow_state,
    test_case: ToolTestCase,
):
    mock_now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.timezone = timezone

    mock_internal_event_tracker.instance = MagicMock(return_value=None)
    mock_internal_event_tracker.track_event = MagicMock(return_value=None)
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT

    # Create mock toolset
    mock_toolset = MagicMock(spec=Toolset)

    # Configure the mock toolset to behave like a dictionary
    def mock_contains(key):
        for tool in test_case.tools.keys():
            if tool.name == key:
                return True
        return False

    def mock_getitem(key):
        for tool in test_case.tools.keys():
            if tool.name == key:
                return tool
        raise KeyError(f"Tool '{key}' does not exist in executable tools")

    mock_toolset.__contains__ = MagicMock(side_effect=mock_contains)
    mock_toolset.__getitem__ = MagicMock(side_effect=mock_getitem)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=workflow_type,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(content=test_case.ai_content, tool_calls=test_case.tool_calls),
    ]

    result = await tools_executor.run(workflow_state)

    assert (
        result["conversation_history"]["planner"][-len(test_case.tools_response) :]
        == test_case.tools_response
    )

    assert "ui_chat_log" in result
    ui_chat_logs = result["ui_chat_log"]
    assert "ui_chat_log" in result
    has_non_hidden_tools = any(
        tool_call["name"] not in ["get_plan"] for tool_call in test_case.tool_calls
    )
    expected_agent_messages = 1 if has_non_hidden_tools else 0
    expected_tool_messages = sum(
        1 for tool, expect_call in test_case.tools.items() if expect_call
    )

    expected_total = expected_agent_messages + expected_tool_messages
    assert len(ui_chat_logs) == expected_total

    if expected_total > 0:
        message_index = 0

        if expected_agent_messages > 0:
            assert ui_chat_logs[message_index]["message_type"] == MessageTypeEnum.AGENT
            message_index += 1

        for i in range(message_index, len(ui_chat_logs)):
            assert ui_chat_logs[i]["message_type"] == MessageTypeEnum.TOOL

    for tool, expect_call in test_case.tools.items():
        if expect_call:
            tool.arun.assert_called_once()
            assert mock_internal_event_tracker.track_event.call_count == 1
            mock_internal_event_tracker.track_event.assert_has_calls(
                [
                    call(
                        event_name=EventEnum.WORKFLOW_TOOL_SUCCESS.value,
                        additional_properties=InternalEventAdditionalProperties(
                            label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
                            property=mock_tool().name,
                            value="123",
                        ),
                        category=workflow_type.value,
                    ),
                ]
            )
        else:
            tool.arun.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last_message, expected_message_types",
    [
        (
            lambda tool: AIMessage(
                content=[{"type": "text", "text": "I'm going to search for something"}],
                tool_calls=[
                    {
                        "id": "1",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    }
                ],
            ),
            [MessageTypeEnum.AGENT, MessageTypeEnum.TOOL],
        ),
        (
            lambda tool: AIMessage(
                content=[
                    {"type": "text", "text": "I'm just thinking without using tools"}
                ],
                tool_calls=[],
            ),
            [MessageTypeEnum.AGENT],
        ),
        (
            lambda tool: AIMessage(
                content=[{"type": "text", "text": "I'm going to use multiple tools"}],
                tool_calls=[
                    {
                        "id": "1",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    },
                    {
                        "id": "2",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step2"}]},
                    },
                ],
            ),
            [MessageTypeEnum.AGENT, MessageTypeEnum.TOOL, MessageTypeEnum.TOOL],
        ),
        (
            lambda tool: HumanMessage(content="This is a human message"),
            [],
        ),
        (
            lambda tool: AIMessage(
                content=[{"type": "text", "text": "I'm going to check the plan"}],
                tool_calls=[
                    {
                        "id": "1",
                        "name": "get_plan",
                        "args": {},
                    }
                ],
            ),
            [],
        ),
        (
            lambda tool: AIMessage(
                content=[
                    {"type": "text", "text": "I'll check the plan and add a task"}
                ],
                tool_calls=[
                    {
                        "id": "1",
                        "name": "get_plan",
                        "args": {},
                    },
                    {
                        "id": "2",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    },
                ],
            ),
            [MessageTypeEnum.AGENT, MessageTypeEnum.TOOL],
        ),
    ],
    ids=[
        "single_tool_call",
        "no_tool_call",
        "multiple_tool_call",
        "last_message_not_AIMessage",
        "all_hidden_tools",
        "mixed_hidden_visible_tools",
    ],
)
async def test_adding_ai_context_to_ui_chat_logs(
    workflow_state,
    last_message,
    expected_message_types,
):
    tool = mock_tool()

    mock_toolset = MagicMock(spec=Toolset)
    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=tool)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    workflow_state["conversation_history"]["planner"] = [last_message(tool)]
    workflow_state["plan"] = {"steps": []}

    result = await tools_executor.run(workflow_state)

    assert "ui_chat_log" in result
    assert len(result["ui_chat_log"]) == len(expected_message_types)

    for i, expected_type in enumerate(expected_message_types):
        if i < len(result["ui_chat_log"]):
            assert result["ui_chat_log"][i]["message_type"] == expected_type

    if expected_message_types and expected_message_types[0] == MessageTypeEnum.AGENT:
        message = last_message(tool)
        if isinstance(message, AIMessage) and message.content:
            if (
                isinstance(message.content, list)
                and len(message.content) > 0
                and isinstance(message.content[0], dict)
                and "text" in message.content[0]
            ):
                expected_content = message.content[0]["text"]
            else:
                expected_content = message.content
            assert result["ui_chat_log"][0]["content"] == expected_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "create_plan",
                    "args": CreatePlanInput(
                        tasks=["Task 1", "Task 2", "Task 3"]
                    ).model_dump(),
                }
            ],
            "tools_response": [ToolMessage(content="Plan created", tool_call_id="1")],
            "expected_plan": {
                "steps": [
                    {"id": "task-0", "description": "Task 1", "status": "Not Started"},
                    {"id": "task-1", "description": "Task 2", "status": "Not Started"},
                    {"id": "task-2", "description": "Task 3", "status": "Not Started"},
                ]
            },
            "expected_log_content": "Create plan with 3 tasks",
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "update_task_description",
                    "args": UpdateTaskDescriptionInput(
                        task_id="1", new_description="step1"
                    ).model_dump(),
                }
            ],
            "tools_response": [
                ToolMessage(content="Task not found: 1", tool_call_id="1")
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": "Update description for task 2",
        },
        {
            "plan": {"steps": [{"id": "1", "description": "old step1"}]},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "update_task_description",
                    "args": UpdateTaskDescriptionInput(
                        task_id="1", new_description="new step1"
                    ).model_dump(),
                }
            ],
            "tools_response": [
                ToolMessage(content="Task updated: 1", tool_call_id="1")
            ],
            "expected_plan": {"steps": [{"id": "1", "description": "new step1"}]},
            "expected_log_content": "Update description for task 2",
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "add_new_task",
                    "args": AddNewTaskInput(description="New task").model_dump(),
                }
            ],
            "tools_response": [
                ToolMessage(content="Step added: task-0", tool_call_id="1")
            ],
            "expected_plan": {
                "steps": [
                    {
                        "id": "task-0",
                        "description": "New task",
                        "status": TaskStatus.NOT_STARTED,
                    }
                ]
            },
            "expected_log_content": "Add new task to the plan: New task...",
        },
        {
            "plan": {"steps": [{"id": "1", "description": "Task to remove"}]},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "remove_task",
                    "args": RemoveTaskInput(task_id="1").model_dump(),
                }
            ],
            "tools_response": [
                ToolMessage(content="Task removed: 1", tool_call_id="1")
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": "Remove task 2",
        },
        {
            "plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task to update",
                        "status": TaskStatus.NOT_STARTED,
                    }
                ]
            },
            "tool_calls": [
                {
                    "id": "1",
                    "name": "set_task_status",
                    "args": SetTaskStatusInput(
                        task_id="1", status=TaskStatus.IN_PROGRESS
                    ).model_dump(),
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task status set: 1 - In Progress",
                    tool_call_id="1",
                )
            ],
            "expected_plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task to update",
                        "status": TaskStatus.IN_PROGRESS,
                    }
                ]
            },
            "expected_log_content": "Set task 2 to 'In Progress'",
        },
    ],
)
@patch("duo_workflow_service.agents.tools_executor.datetime")
async def test_run_with_state_manipulating_tools(
    mock_datetime,
    workflow_state,
    test_case,
):
    mock_now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.timezone = timezone

    mock_toolset = MagicMock(spec=Toolset)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=test_case["tool_calls"],
        ),
    ]
    workflow_state["plan"] = test_case["plan"]

    result = await tools_executor.run(workflow_state)

    assert (
        result["conversation_history"]["planner"][-len(test_case["tools_response"]) :]
        == test_case["tools_response"]
    )

    assert result["plan"] == test_case["expected_plan"]

    assert "ui_chat_log" in result
    ui_chat_logs = result["ui_chat_log"]
    assert len(ui_chat_logs) == 2

    agent_log = ui_chat_logs[0]
    assert agent_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert agent_log["message_type"] == MessageTypeEnum.AGENT
    assert agent_log["content"] == "test"
    assert agent_log["tool_info"] is None

    tool_log = ui_chat_logs[1]
    assert tool_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert tool_log["message_type"] == MessageTypeEnum.TOOL
    assert tool_log["content"] == test_case["expected_log_content"]
    assert tool_log["tool_info"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_call, tool_side_effect, tool_args_schema, expected_response, expected_error, expected_log_prefix,"
    "expected_tool_info",
    [
        (
            {"id": "1", "name": "test_tool", "args": {}},
            TypeError("Wrong arguments"),
            AddNewTaskInput,
            "Tool test_tool execution failed due to wrong arguments. You must adhere to the tool args schema! The "
            "schema is: {'properties': {'description': {'description': 'The description of the new task to add', "
            "'title': 'Description', 'type': 'string'}}, 'required': ['description'], 'title': 'AddNewTaskInput', "
            "'type': 'object'}",
            False,
            "Failed: Using test_tool:  - Invalid arguments",
            ToolInfo(name="test_tool", args={}),
        ),
        (
            {"id": "2", "name": "test_tool", "args": {"invalid": "data"}},
            ValidationError.from_exception_data(
                title="validation_error",
                line_errors=[
                    {
                        "ctx": {"error": "Extra inputs are not permitted"},
                        "input": "data",
                        "loc": ("invalid",),
                        "type": "extra_forbidden",
                    }
                ],
            ),
            None,
            "Tool test_tool raised validation error 1 validation error for validation_error",
            False,
            "Failed: Using test_tool: invalid=data - Validation error",
            ToolInfo(name="test_tool", args={"invalid": "data"}),
        ),
        (
            {"id": "3", "name": "test_tool", "args": {}},
            PipelineMergeRequestNotFoundError("Merge request not found"),
            None,
            "Pipeline exception",
            True,
            "Failed: Using test_tool:  - Pipeline error: Merge request not found",
            ToolInfo(name="test_tool", args={}),
        ),
    ],
)
@patch("duo_workflow_service.agents.tools_executor.datetime")
@patch("duo_workflow_service.agents.tools_executor.DuoWorkflowInternalEvent")
@patch.dict(os.environ, {"DW_INTERNAL_EVENT__ENABLED": "true"})
async def test_run_error_handling(
    mock_internal_event_tracker,
    mock_datetime,
    workflow_state,
    *,
    tool_call,
    tool_side_effect,
    tool_args_schema,
    expected_response,
    expected_error,
    expected_log_prefix,
    expected_tool_info,
):
    mock_now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.timezone = timezone

    mock_internal_event_tracker.instance = MagicMock(return_value=None)
    mock_internal_event_tracker.track_event = MagicMock(return_value=None)
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    tool = mock_tool(side_effect=tool_side_effect, args_schema=tool_args_schema)

    mock_toolset = MagicMock(spec=Toolset)

    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=tool)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=workflow_type,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(content=[{"type": "text", "text": "test"}], tool_calls=[tool_call]),
    ]

    result = await tools_executor.run(workflow_state)

    assert mock_internal_event_tracker.track_event.call_count == 1
    mock_internal_event_tracker.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_TOOL_FAILURE.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
                    property=mock_tool().name,
                    value="123",
                    error=str(tool_side_effect),
                ),
                category=workflow_type.value,
            ),
        ]
    )
    assert len(result["conversation_history"]["planner"]) == 1
    assert result["conversation_history"]["planner"][0].content.startswith(
        expected_response
    )

    if expected_error:
        assert result["status"] == WorkflowStatusEnum.ERROR

    assert "ui_chat_log" in result
    ui_chat_logs = result["ui_chat_log"]
    assert len(ui_chat_logs) == 2

    agent_log = ui_chat_logs[0]
    assert agent_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert agent_log["message_type"] == MessageTypeEnum.AGENT
    assert agent_log["content"] == "test"
    assert agent_log["tool_info"] is None

    tool_log = ui_chat_logs[1]
    assert tool_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert tool_log["message_type"] == MessageTypeEnum.TOOL
    assert tool_log["content"].startswith(expected_log_prefix)
    assert tool_log["tool_info"] == expected_tool_info
    tool.arun.assert_called_once()


@pytest.mark.parametrize(
    "tool_name, args, expected_message",
    [
        (
            "add_new_task",
            {"description": "Create a new feature"},
            "Add new task to the plan: Create a new feature...",
        ),
        ("remove_task", {"task_id": "task-1"}, "Remove task 2"),
        (
            "update_task_description",
            {"task_id": "task-2", "new_description": "Updated description"},
            "Update description for task 3",
        ),
        (
            "set_task_status",
            {"task_id": "task-3", "status": "Completed"},
            "Set task 4 to 'Completed'",
        ),
        (
            "create_plan",
            {"tasks": ["Task 1", "Task 2", "Task 3"]},
            "Create plan with 3 tasks",
        ),
        ("get_plan", {}, None),  # Hidden tool
    ],
)
def test_get_tool_display_message_action_handlers(tool_name, args, expected_message):
    tools_executor = ToolsExecutor(
        tools_agent_name="test_agent",
        toolset=MagicMock(spec=Toolset),
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    message = tools_executor.get_tool_display_message(tool_name, args)
    assert message == expected_message


class MockGetIssueInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    issue_id: int = Field(description="The internal ID of the project issue")


class MockGetIssue(BaseTool):
    name: str = "get_issue"
    description: str = "Get a single issue in a GitLab project"
    args_schema: Type[BaseModel] = MockGetIssueInput

    async def _arun(self) -> str:
        return ""

    def _run(self) -> str:
        return ""

    def format_display_message(self, args: MockGetIssueInput) -> str:
        return f"Read issue #{args.issue_id} in project {args.project_id}"


def test_get_tool_display_message_tool_lookup():
    mock_tool = MockGetIssue()

    mock_toolset = MagicMock(spec=Toolset)

    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=mock_tool)

    tools_executor = ToolsExecutor(
        tools_agent_name="test_agent",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    valid_args = {"project_id": 123, "issue_id": 456}
    message = tools_executor.get_tool_display_message("get_issue", valid_args)
    assert message == "Read issue #456 in project 123"


def test_get_tool_display_message_unknown_tool():
    mock_toolset = MagicMock(spec=Toolset)

    mock_toolset.__contains__ = MagicMock(return_value=False)

    tools_executor = ToolsExecutor(
        tools_agent_name="test_agent",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    message = tools_executor.get_tool_display_message(
        "unknown_tool", {"param": "value"}
    )
    assert message == "Using unknown_tool: param=value"
