# pylint: disable=direct-environment-variable-reference,too-many-lines
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Type, cast
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import ToolException
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from contract import contract_pb2
from duo_workflow_service.agents import ToolsExecutor
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    TaskStatus,
    ToolInfo,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools import RunCommand, Toolset
from duo_workflow_service.tools.planner import (
    AddNewTask,
    AddNewTaskInput,
    CreatePlan,
    RemoveTask,
    SetTaskStatus,
    UpdateTaskDescription,
)
from duo_workflow_service.tools.toolset import ToolType
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventLabelEnum


def mock_tool(
    name="test_tool", content="test_tool result", side_effect=None, args_schema=None
):
    mock = MagicMock(BaseTool)
    mock.name = name
    mock.args_schema = args_schema
    if side_effect:
        mock.ainvoke.side_effect = side_effect
    else:
        mock.ainvoke.return_value = ToolMessage(
            content=content, name=name, tool_call_id="fake-call-1"
        )
    return mock


@pytest.fixture(autouse=True)
def prepare_container(
    mock_duo_workflow_service_container,
):  # pylint: disable=unused-argument
    pass


@pytest.fixture(name="all_tools")
def all_tools_fixture() -> dict[str, ToolType]:
    return {
        "set_task_status": SetTaskStatus(),
        "add_new_task": AddNewTask(),
        "remove_task": RemoveTask(),
        "update_task_description": UpdateTaskDescription(),
        "create_plan": CreatePlan(),
    }


@pytest.fixture(name="toolset")
def toolset_fixture(all_tools: dict[str, ToolType]) -> Toolset:
    return Toolset(
        pre_approved=set(),
        all_tools=all_tools,
    )


@pytest.fixture(name="tools_executor")
def tools_executor_fixture(toolset: Toolset) -> ToolsExecutor:
    return ToolsExecutor(
        tools_agent_name="planner",
        toolset=toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )


@pytest.fixture(name="graph")
def graph_fixture(tools_executor: ToolsExecutor) -> Runnable:
    graph_builder = StateGraph(WorkflowState)
    graph_builder.add_node("exec", tools_executor.run)
    graph_builder.set_entry_point("exec")
    graph_builder.set_finish_point("exec")

    return graph_builder.compile()


@pytest.fixture(name="ui_chat_log")
def ui_chat_log_fixture():
    return []


@pytest.fixture(name="mock_datetime")
def mock_datetime_fixture(mock_now: datetime):
    with patch("duo_workflow_service.agents.tools_executor.datetime") as mock:
        mock.now.return_value = mock_now
        mock.timezone = timezone
        yield mock


@pytest.fixture(name="mock_action_response")
def mock_action_response_fixture():
    mock_action_response = contract_pb2.ActionResponse()
    mock_action_response.requestID = "test-request-id"
    mock_action_response.plainTextResponse.response = "/home output"
    mock_action_response.plainTextResponse.error = ""  # No error
    return mock_action_response


@pytest.fixture(name="mock_client_event")
def mock_client_event_fixture(mock_action_response):
    mock_client_event = contract_pb2.ClientEvent()
    mock_client_event.actionResponse.CopyFrom(mock_action_response)
    return mock_client_event


@dataclass
class ToolTestCase:
    tool_calls: List[Dict]
    tools: Dict[MagicMock, bool]
    tools_response: List[dict[str, Any]]
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
            tools_response=[
                {
                    "conversation_history": {
                        "planner": [
                            ToolMessage(
                                content="test_tool result",
                                name=mock_tool().name,
                                tool_call_id="fake-call-1",
                            )
                        ]
                    }
                },
            ],
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
                {
                    "conversation_history": {
                        "planner": [
                            ToolMessage(
                                content="Tool does_not_exist not found",
                                tool_call_id="1",
                            )
                        ]
                    }
                },
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
                {
                    "conversation_history": {
                        "planner": [
                            ToolMessage(
                                content="Tool does_not_exist not found",
                                tool_call_id="1",
                            )
                        ]
                    }
                },
                {
                    "conversation_history": {
                        "planner": [
                            ToolMessage(
                                content="test_tool result",
                                name=mock_tool().name,
                                tool_call_id="fake-call-1",
                            )
                        ]
                    }
                },
            ],
        ),
    ],
)
async def test_run(
    workflow_state, test_case: ToolTestCase, internal_event_client: Mock
):
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
        internal_event_client=internal_event_client,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(content=test_case.ai_content, tool_calls=test_case.tool_calls),
    ]

    result = await tools_executor.run(workflow_state)

    assert result[: len(test_case.tools_response)] == test_case.tools_response

    update = cast(Command, result[-1]).update
    assert update and "ui_chat_log" in update
    ui_chat_logs = update["ui_chat_log"]
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
            tool.ainvoke.assert_called_once()
            assert internal_event_client.track_event.call_count == 1
            internal_event_client.track_event.assert_has_calls(
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
            tool.ainvoke.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("all_tools", [{"test_tool": mock_tool()}])
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
    all_tools,
    tools_executor,
    workflow_state,
    last_message,
    expected_message_types,
):
    tool = all_tools["test_tool"]

    workflow_state["conversation_history"]["planner"] = [last_message(tool)]

    result = await tools_executor.run(workflow_state)

    update = cast(Command, result[-1]).update
    assert update and "ui_chat_log" in update
    ui_chat_log = update["ui_chat_log"]

    assert len(ui_chat_log) == len(expected_message_types)

    for i, expected_type in enumerate(expected_message_types):
        if i < len(ui_chat_log):
            assert ui_chat_log[i]["message_type"] == expected_type

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
            assert ui_chat_log[0]["content"] == expected_content


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
                    "args": {
                        "tasks": ["Task 1", "Task 2", "Task 3"],
                    },
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Plan created", name="create_plan", tool_call_id="1"
                )
            ],
            "expected_plan": {
                "steps": [
                    {"id": "task-0", "description": "Task 1", "status": "Not Started"},
                    {"id": "task-1", "description": "Task 2", "status": "Not Started"},
                    {"id": "task-2", "description": "Task 3", "status": "Not Started"},
                ]
            },
            "expected_log_content": ["Create plan with 3 tasks"],
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "create_plan",
                    "args": {
                        "tasks": ["Task 1", "Task 2", "Task 3"],
                    },
                },
                {
                    "id": "2",
                    "name": "create_plan",
                    "args": {
                        "tasks": ["Task 4", "Task 5"],
                    },
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Plan created", name="create_plan", tool_call_id="1"
                ),
                ToolMessage(
                    content="Plan created", name="create_plan", tool_call_id="2"
                ),
            ],
            "expected_plan": {
                "steps": [
                    {"id": "task-0", "description": "Task 4", "status": "Not Started"},
                    {"id": "task-1", "description": "Task 5", "status": "Not Started"},
                ]
            },
            "expected_log_content": [
                "Create plan with 3 tasks",
                "Create plan with 2 tasks",
            ],
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "update_task_description",
                    "args": {
                        "task_id": "1",
                        "new_description": "step1",
                    },
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task not found: 1",
                    name="update_task_description",
                    tool_call_id="1",
                )
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": ["Update description for task 'step1'"],
        },
        {
            "plan": {"steps": [{"id": "1", "description": "old step1"}]},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "update_task_description",
                    "args": {
                        "task_id": "1",
                        "new_description": "new step1",
                    },
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task updated: 1",
                    name="update_task_description",
                    tool_call_id="1",
                )
            ],
            "expected_plan": {"steps": [{"id": "1", "description": "new step1"}]},
            "expected_log_content": ["Update description for task 'new step1'"],
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "add_new_task",
                    "args": {"description": "New task"},
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Step added: task-0", name="add_new_task", tool_call_id="1"
                )
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
            "expected_log_content": ["Add new task to the plan: New task"],
        },
        {
            "plan": {"steps": [{"id": "1", "description": "Task to remove"}]},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task removed: 1", name="remove_task", tool_call_id="1"
                )
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": ["Remove task 'Test description 1'"],
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
                    "args": {
                        "task_id": "1",
                        "status": TaskStatus.IN_PROGRESS,
                        "description": "Test description",
                    },
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task status set: 1 - In Progress",
                    name="set_task_status",
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
            "expected_log_content": ["Set task 'Test description' to 'In Progress'"],
        },
        {
            "plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.NOT_STARTED,
                    },
                    {
                        "id": "2",
                        "description": "Task 2",
                        "status": TaskStatus.NOT_STARTED,
                    },
                ]
            },
            "tool_calls": [
                {
                    "id": "1",
                    "name": "set_task_status",
                    "args": {
                        "task_id": "1",
                        "status": TaskStatus.IN_PROGRESS,
                        "description": "Test description",
                    },
                },
                {
                    "id": "2",
                    "name": "update_task_description",
                    "args": {
                        "task_id": "2",
                        "new_description": "New description",
                    },
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Task status set: 1 - In Progress",
                    name="set_task_status",
                    tool_call_id="1",
                ),
                ToolMessage(
                    content="Task updated: 2",
                    name="update_task_description",
                    tool_call_id="2",
                ),
            ],
            "expected_plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.IN_PROGRESS,
                    },
                    {
                        "id": "2",
                        "description": "New description",
                        "status": TaskStatus.NOT_STARTED,
                    },
                ]
            },
            "expected_log_content": [
                "Set task 'Test description' to 'In Progress'",
                "Update description for task 'New description'",
            ],
        },
        {
            "plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.NOT_STARTED,
                    },
                    {
                        "id": "2",
                        "description": "Task 2",
                        "status": TaskStatus.NOT_STARTED,
                    },
                ]
            },
            "tool_calls": [
                {
                    "id": "1",
                    "name": "set_task_status",
                    "args": {
                        "task_id": "1",
                        "status": TaskStatus.IN_PROGRESS,
                        "description": "Task 1",
                    },
                },
                {
                    "id": "2",
                    "name": "remove_task",
                    "args": {
                        "task_id": "2",
                        "description": "Task 2",
                    },
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Task status set: 1 - In Progress",
                    name="set_task_status",
                    tool_call_id="1",
                ),
                ToolMessage(
                    content="Task removed: 2",
                    name="remove_task",
                    tool_call_id="2",
                ),
            ],
            "expected_plan": {
                "steps": [
                    {
                        "id": "1",
                        "description": "Task 1",
                        "status": TaskStatus.IN_PROGRESS,
                    },
                ]
            },
            "expected_log_content": [
                "Set task 'Task 1' to 'In Progress'",
                "Remove task 'Task 2'",
            ],
        },
        {
            "plan": {"steps": [{"id": "1", "description": "Task to remove"}]},
            "tool_calls": [
                {
                    "id": "1",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                },
                {
                    "id": "2",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Task removed: 1", name="remove_task", tool_call_id="1"
                ),
                ToolMessage(
                    content="Task removed: 1", name="remove_task", tool_call_id="2"
                ),
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": [
                "Remove task 'Test description 1'",
                "Remove task 'Test description 1'",
            ],
        },
    ],
)
@pytest.mark.usefixtures("mock_datetime")
async def test_state_manipulation(
    graph,
    workflow_state,
    test_case,
):
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=test_case["tool_calls"],
        ),
    ]
    workflow_state["plan"] = test_case["plan"]

    result = await graph.ainvoke(workflow_state)

    assert (
        result["conversation_history"]["planner"][-len(test_case["tools_response"]) :]
        == test_case["tools_response"]
    )

    assert result["plan"] == test_case["expected_plan"]

    assert "ui_chat_log" in result
    ui_chat_logs = result["ui_chat_log"]
    assert len(ui_chat_logs) == len(test_case["expected_log_content"]) + 1

    agent_log = ui_chat_logs[0]
    assert agent_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert agent_log["message_type"] == MessageTypeEnum.AGENT
    assert agent_log["content"] == "test"
    assert agent_log["tool_info"] is None

    for i, expected_content in enumerate(test_case["expected_log_content"]):
        tool_log = ui_chat_logs[i + 1]
        assert tool_log["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert tool_log["message_type"] == MessageTypeEnum.TOOL
        assert tool_log["content"] == expected_content
        assert tool_log["tool_info"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_call, tool_side_effect, tool_args_schema, expected_response, expected_error, expected_log_prefix, "
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
            {"id": "3", "name": "test_tool", "args": {"file_path": ".git/config"}},
            ToolException(
                "Access denied: Cannot access '.git/config' as it matches Duo Context Exclusion patterns. "
                "Path '.git/config' matches excluded pattern: '.git'."
            ),
            None,  # tool_args_schema
            "Tool test_tool raised ToolException: Access denied: Cannot access '.git/config' as it "
            "matches Duo Context Exclusion patterns. Path '.git/config' matches excluded pattern: '.git'.",
            False,  # expected_error
            "Failed: Using test_tool: file_path=.git/config - ToolException",
            ToolInfo(name="test_tool", args={"file_path": ".git/config"}),
        ),
    ],
)
@pytest.mark.usefixtures("mock_datetime")
@patch("duo_workflow_service.agents.tools_executor.duo_workflow_metrics")
async def test_run_error_handling(
    mock_duo_workflow_metrics,
    workflow_state,
    *,
    tool_call,
    tool_side_effect,
    tool_args_schema,
    internal_event_client: Mock,
    expected_response,
    expected_error,
    expected_log_prefix,
    expected_tool_info,
):
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
        internal_event_client=internal_event_client,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(content=[{"type": "text", "text": "test"}], tool_calls=[tool_call]),
    ]

    result = await tools_executor.run(workflow_state)

    assert internal_event_client.track_event.call_count == 1
    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_TOOL_FAILURE.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
                    property=mock_tool().name,
                    value="123",
                    error=str(tool_side_effect),
                    error_type=type(tool_side_effect).__name__,
                ),
                category=workflow_type.value,
            ),
        ]
    )
    assert len(result) == 2
    assert result[0]["conversation_history"]["planner"][0].content.startswith(
        expected_response
    )

    updates = cast(Command, result[1]).update

    if expected_error:
        assert updates["status"] == WorkflowStatusEnum.ERROR  # type: ignore[index]

    ui_chat_logs = updates["ui_chat_log"]  # type: ignore[index]
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

    mock_duo_workflow_metrics.count_agent_platform_tool_failure.assert_called_once_with(
        flow_type=workflow_type.value,
        tool_name=mock_tool().name,
        failure_reason=type(tool_side_effect).__name__,
    )

    tool.ainvoke.assert_called_once()


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

    def format_display_message(
        self, args: MockGetIssueInput, _tool_response: Any = None
    ) -> str:
        return f"Read issue #{args.issue_id} in project {args.project_id}"


@pytest.mark.parametrize("all_tools", [{"get_issue": MockGetIssue()}])
@pytest.mark.parametrize(
    "tool_name, args, expected_message",
    [
        (
            "get_issue",
            {"project_id": 123, "issue_id": 456},
            "Read issue #456 in project 123",
        ),
        ("get_plan", {}, None),  # Hidden tool
    ],
)
def test_get_tool_display_message_tool_lookup(
    tools_executor: ToolsExecutor, tool_name: str, args: dict, expected_message: str
):
    message = tools_executor.get_tool_display_message(tool_name, args)
    assert message == expected_message


@pytest.mark.parametrize("all_tools", [{}])
def test_get_tool_display_message_unknown_tool(tools_executor: ToolsExecutor):
    message = tools_executor.get_tool_display_message(
        "unknown_tool", {"param": "value"}
    )
    assert message == "Using unknown_tool: param=value"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "all_tools",
    [
        {
            "run_command": RunCommand(
                metadata={
                    "outbox": MagicMock(spec=asyncio.Queue),
                    "inbox": MagicMock(spec=asyncio.Queue),
                }
            )
        }
    ],
)
async def test_run_command_output(workflow_state, tools_executor, mock_client_event):
    # Configure the inbox mock to return the mock ClientEvent
    inbox_mock = tools_executor._toolset["run_command"].metadata["inbox"]
    inbox_mock.get.return_value = mock_client_event

    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "testing"}],
            tool_calls=[
                {
                    "id": "1",
                    "name": "run_command",
                    "args": {
                        "program": "echo",
                        "arguments": ["/home"],
                        "flags": ["-l"],
                    },
                }
            ],
        )
    ]

    result = await tools_executor.run(workflow_state)

    update = cast(Command, result[-1]).update
    assert update and "ui_chat_log" in update
    ui_chat_logs = update["ui_chat_log"]

    assert ui_chat_logs[-1]["tool_info"]["tool_response"]
    assert ui_chat_logs[-1]["message_sub_type"] == "command_output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "all_tools",
    [
        {
            "a": mock_tool(name="a", content="a" * 10000),
            "b": mock_tool(name="b", content="tool b"),
        }
    ],
)
@pytest.mark.usefixtures("mock_datetime")
async def test_multiple_tool_calls(workflow_state, graph):
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[
                {"id": "1", "name": "a", "args": {"a1": 1}},
                {"id": "2", "name": "b", "args": {"b1": 1}},
            ],
        )
    ]

    result = await graph.ainvoke(workflow_state)

    assert result["conversation_history"]["planner"][-2:] == [
        HumanMessage(content="a" * 10000),
        HumanMessage(content="tool b"),
    ]

    assert result["ui_chat_log"][-2:] == [
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": "a",
            "content": "Using a: a1=1",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "status": ToolStatus.SUCCESS,
            "tool_info": {
                "name": "a",
                "args": {"a1": 1},
                "tool_response": ToolMessage(
                    content="a" * 4096, name="a", tool_call_id="fake-call-1"
                ),
            },
            "correlation_id": None,
            "additional_context": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": "b",
            "content": "Using b: b1=1",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "status": ToolStatus.SUCCESS,
            "tool_info": {
                "name": "b",
                "args": {"b1": 1},
                "tool_response": ToolMessage(
                    content="tool b", name="b", tool_call_id="fake-call-1"
                ),
            },
            "correlation_id": None,
            "additional_context": None,
        },
    ]


@pytest.mark.asyncio
async def test_run_with_missing_plan_key(tools_executor):
    state_without_plan = {
        "conversation_history": {
            "planner": [
                AIMessage(
                    content=[{"type": "text", "text": "test"}],
                    tool_calls=[
                        {"id": "1", "name": "test_tool", "args": {"param": "value"}}
                    ],
                )
            ]
        },
        "status": WorkflowStatusEnum.EXECUTION,
    }

    result = await tools_executor.run(state_without_plan)

    assert result is not None
    assert isinstance(result, list)
