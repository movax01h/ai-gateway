# pylint: disable=too-many-lines
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Type, cast
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import ToolException
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from ai_gateway.instrumentators.model_requests import client_capabilities
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
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.security.prompt_scanner import DetectionType, ScanResult
from duo_workflow_service.security.scanner_factory import PromptInjectionDetectedError
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
from lib.events import GLReportingEventContext
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


@pytest.fixture(name="flow_type")
def flow_type_fixture() -> GLReportingEventContext:
    return GLReportingEventContext.from_workflow_definition("software_development")


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
def tools_executor_fixture(
    toolset: Toolset, flow_type: GLReportingEventContext
) -> ToolsExecutor:
    return ToolsExecutor(
        tools_agent_name="planner",
        toolset=toolset,
        workflow_id="123",
        workflow_type=flow_type,
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
                    "id": "tool-call-1",
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
                    "id": "tool-call-2",
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
                                tool_call_id="tool-call-2",
                            )
                        ]
                    }
                },
            ],
        ),
        ToolTestCase(
            tool_calls=[
                {
                    "id": "tool-call-3",
                    "name": "does_not_exist",
                    "args": {"summary": "done"},
                },
                {
                    "id": "tool-call-4",
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
                                tool_call_id="tool-call-3",
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
    workflow_state,
    test_case: ToolTestCase,
    internal_event_client: Mock,
    flow_type: GLReportingEventContext,
):
    # Create mock toolset
    mock_toolset = MagicMock(spec=Toolset)

    client_capabilities.set({"feature_a", "shell_command"})

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
        workflow_type=flow_type,
        internal_event_client=internal_event_client,
    )
    ai_message = AIMessage(
        content=test_case.ai_content, tool_calls=test_case.tool_calls, id="ai-msg-123"
    )
    workflow_state["conversation_history"]["planner"] = [ai_message]

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
            assert ui_chat_logs[message_index]["message_id"] == "ai-msg-123"
            message_index += 1

        # Verify tool call ids are reflected in ui_chat_log
        # Only tools that were successfully called will have ui_chat_log entries
        successful_tool_calls = [
            tc
            for tc in test_case.tool_calls
            if any(
                tool.name == tc["name"] and expect_call
                for tool, expect_call in test_case.tools.items()
            )
        ]

        for i in range(message_index, len(ui_chat_logs)):
            assert ui_chat_logs[i]["message_type"] == MessageTypeEnum.TOOL
            # Map ui_chat_log index to successful tool call index
            tool_log_index = i - message_index
            if tool_log_index < len(successful_tool_calls):
                expected_tool_call_id = successful_tool_calls[tool_log_index]["id"]
                assert ui_chat_logs[i]["message_id"] == expected_tool_call_id

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
                            client_capabilities=ANY,
                        ),
                        category=flow_type.value,
                    ),
                ]
            )
            assert set(
                internal_event_client.track_event.call_args.kwargs[
                    "additional_properties"
                ].extra["client_capabilities"]
            ) == {
                "feature_a",
                "shell_command",
            }
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
                        "id": "tool-call-ai-1",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    }
                ],
                id="ai-msg-1",
            ),
            [MessageTypeEnum.AGENT, MessageTypeEnum.TOOL],
        ),
        (
            lambda tool: AIMessage(
                content=[
                    {"type": "text", "text": "I'm just thinking without using tools"}
                ],
                tool_calls=[],
                id="ai-msg-2",
            ),
            [MessageTypeEnum.AGENT],
        ),
        (
            lambda tool: AIMessage(
                content=[{"type": "text", "text": "I'm going to use multiple tools"}],
                tool_calls=[
                    {
                        "id": "tool-call-ai-3a",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    },
                    {
                        "id": "tool-call-ai-3b",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step2"}]},
                    },
                ],
                id="ai-msg-3",
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
                        "id": "tool-call-ai-5",
                        "name": "get_plan",
                        "args": {},
                    }
                ],
                id="ai-msg-5",
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
                        "id": "tool-call-ai-6a",
                        "name": "get_plan",
                        "args": {},
                    },
                    {
                        "id": "tool-call-ai-6b",
                        "name": tool.name,
                        "args": {"tasks": [{"description": "step1"}]},
                    },
                ],
                id="ai-msg-6",
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
            # Verify AI message id is reflected in ui_chat_log
            assert ui_chat_log[0]["message_id"] == message.id

    # Verify tool call ids are reflected in ui_chat_log
    message = last_message(tool)
    if isinstance(message, AIMessage) and message.tool_calls:
        tool_log_start_index = (
            1
            if expected_message_types
            and expected_message_types[0] == MessageTypeEnum.AGENT
            else 0
        )
        visible_tool_calls = [
            tc for tc in message.tool_calls if tc["name"] not in ["get_plan"]
        ]
        for i, tool_call in enumerate(visible_tool_calls):
            log_index = tool_log_start_index + i
            if log_index < len(ui_chat_log):
                assert ui_chat_log[log_index]["message_id"] == tool_call["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "create-plan-1",
                    "name": "create_plan",
                    "args": {
                        "tasks": ["Task 1", "Task 2", "Task 3"],
                    },
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Plan created",
                    name="create_plan",
                    tool_call_id="create-plan-1",
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
                    "id": "create-plan-2a",
                    "name": "create_plan",
                    "args": {
                        "tasks": ["Task 1", "Task 2", "Task 3"],
                    },
                },
                {
                    "id": "create-plan-2b",
                    "name": "create_plan",
                    "args": {
                        "tasks": ["Task 4", "Task 5"],
                    },
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Plan created",
                    name="create_plan",
                    tool_call_id="create-plan-2a",
                ),
                ToolMessage(
                    content="Plan created",
                    name="create_plan",
                    tool_call_id="create-plan-2b",
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
                    "id": "update-task-3",
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
                    tool_call_id="update-task-3",
                )
            ],
            "expected_plan": {"steps": []},
            "expected_log_content": ["Update description for task 'step1'"],
        },
        {
            "plan": {"steps": [{"id": "1", "description": "old step1"}]},
            "tool_calls": [
                {
                    "id": "update-task-4",
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
                    tool_call_id="update-task-4",
                )
            ],
            "expected_plan": {"steps": [{"id": "1", "description": "new step1"}]},
            "expected_log_content": ["Update description for task 'new step1'"],
        },
        {
            "plan": {"steps": []},
            "tool_calls": [
                {
                    "id": "add-task-5",
                    "name": "add_new_task",
                    "args": {"description": "New task"},
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Step added: task-0",
                    name="add_new_task",
                    tool_call_id="add-task-5",
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
                    "id": "remove-task-6",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                }
            ],
            "tools_response": [
                ToolMessage(
                    content="Task removed: 1",
                    name="remove_task",
                    tool_call_id="remove-task-6",
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
                    "id": "set-status-7",
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
                    tool_call_id="set-status-7",
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
                    "id": "set-status-8a",
                    "name": "set_task_status",
                    "args": {
                        "task_id": "1",
                        "status": TaskStatus.IN_PROGRESS,
                        "description": "Test description",
                    },
                },
                {
                    "id": "update-task-8b",
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
                    tool_call_id="set-status-8a",
                ),
                ToolMessage(
                    content="Task updated: 2",
                    name="update_task_description",
                    tool_call_id="update-task-8b",
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
                    "id": "set-status-9a",
                    "name": "set_task_status",
                    "args": {
                        "task_id": "1",
                        "status": TaskStatus.IN_PROGRESS,
                        "description": "Task 1",
                    },
                },
                {
                    "id": "remove-task-9b",
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
                    tool_call_id="set-status-9a",
                ),
                ToolMessage(
                    content="Task removed: 2",
                    name="remove_task",
                    tool_call_id="remove-task-9b",
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
                    "id": "remove-task-10a",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                },
                {
                    "id": "remove-task-10b",
                    "name": "remove_task",
                    "args": {"task_id": "1", "description": "Test description 1"},
                },
            ],
            "tools_response": [
                ToolMessage(
                    content="Task removed: 1",
                    name="remove_task",
                    tool_call_id="remove-task-10a",
                ),
                ToolMessage(
                    content="Task removed: 1",
                    name="remove_task",
                    tool_call_id="remove-task-10b",
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
            id="ai-msg-state-test",
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
    (
        "tool_call",
        "tool_side_effect",
        "tool_args_schema",
        "expected_response",
        "expected_error",
        "expected_log_prefix",
        "expected_tool_info",
        "expected_extra_log",
    ),
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
            {"context": "Tools executor raised TypeError"},
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
            {"tool_call_fields": ["invalid"]},
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
            "Failed: Using test_tool: file_path=.git/config - Tool call failed: ToolException",
            ToolInfo(name="test_tool", args={"file_path": ".git/config"}),
            {"context": "Tools executor raised error"},
        ),
    ],
)
@pytest.mark.usefixtures("mock_datetime")
@patch("duo_workflow_service.agents.tools_executor.log_exception")
@patch("duo_workflow_service.agents.tools_executor.duo_workflow_metrics")
async def test_run_error_handling(
    mock_duo_workflow_metrics,
    mock_log_exception,
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
    expected_extra_log,
    flow_type: GLReportingEventContext,
):
    client_capabilities.set({"feature_a", "shell_command"})
    tool = mock_tool(side_effect=tool_side_effect, args_schema=tool_args_schema)

    mock_toolset = MagicMock(spec=Toolset)

    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=tool)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=flow_type,
        internal_event_client=internal_event_client,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[tool_call],
            response_metadata={"stop_reason": "tool_call"},
            id="ai-msg-error-test",
        ),
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
                    client_capabilities=ANY,
                ),
                category=flow_type.value,
            ),
        ]
    )

    assert set(
        internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ].extra["client_capabilities"]
    ) == {
        "feature_a",
        "shell_command",
    }

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
        flow_type=flow_type.value,
        tool_name=mock_tool().name,
        failure_reason=type(tool_side_effect).__name__,
    )

    mock_log_exception.assert_called_once_with(
        tool_side_effect,
        extra=expected_extra_log,
    )

    tool.ainvoke.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_datetime")
@patch("duo_workflow_service.agents.tools_executor.duo_workflow_metrics")
async def test_run_error_max_tokens(
    mock_duo_workflow_metrics,
    workflow_state,
    internal_event_client: Mock,
    flow_type: GLReportingEventContext,
):
    mock_toolset = MagicMock(spec=Toolset)
    client_capabilities.set({"feature_a", "shell_command"})

    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=None)

    tool_call = {
        "id": "read-file-max-tokens",
        "name": "read_file",
        "args": {"file_path": "some-long-file-name-that-is-truncated-"},
    }

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=flow_type,
        internal_event_client=internal_event_client,
    )
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[tool_call],
            response_metadata={"finish_reason": "length"},
            id="ai-msg-max-tokens",
        ),
    ]

    result = await tools_executor.run(workflow_state)

    assert internal_event_client.track_event.call_count == 1
    internal_event_client.track_event.assert_has_calls(
        [
            call(
                event_name=EventEnum.WORKFLOW_TOOL_FAILURE.value,
                additional_properties=InternalEventAdditionalProperties(
                    label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
                    property="read_file",
                    value="123",
                    error="Max tokens reached for tool read_file. Try a simpler request or using a different tool.",
                    error_type="IncompleteToolCallDueToMaxTokens",
                    client_capabilities=ANY,
                ),
                category=flow_type.value,
            ),
        ]
    )
    assert set(
        internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ].extra["client_capabilities"]
    ) == {
        "feature_a",
        "shell_command",
    }
    assert len(result) == 2
    assert result[0]["conversation_history"]["planner"][0].content.startswith(
        "Tool read_file raised ToolException: Max tokens reached for tool read_file. "
        "Try a simpler request or using a different tool."
    )

    ui_chat_logs = result[1].update["ui_chat_log"]
    assert len(ui_chat_logs) == 2

    tool_log = ui_chat_logs[1]
    assert tool_log["timestamp"] == "2025-01-01T12:00:00+00:00"
    assert tool_log["message_type"] == MessageTypeEnum.TOOL
    assert tool_log["content"].startswith(
        "Failed: Using read_file: file_path=some-long-file-name-that-is-truncated-"
        " - Tool call failed: IncompleteToolCallDueToMaxTokens"
    )
    assert tool_log["tool_info"] == ToolInfo(
        args={"file_path": "some-long-file-name-that-is-truncated-"}, name="read_file"
    )

    mock_duo_workflow_metrics.count_agent_platform_tool_failure.assert_called_once_with(
        flow_type=flow_type.value,
        tool_name="read_file",
        failure_reason="IncompleteToolCallDueToMaxTokens",
    )


class MockGetIssueInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    issue_id: int = Field(description="The internal ID of the project issue")


class MockGetIssue(BaseTool):
    name: str = "get_issue"
    description: str = "Get a single issue in a GitLab project"
    args_schema: Type[BaseModel] = MockGetIssueInput

    async def _execute(self) -> str:
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
                    "outbox": MagicMock(spec=Outbox),
                }
            )
        }
    ],
)
async def test_run_command_output(workflow_state, tools_executor, mock_client_event):
    # Configure the inbox mock to return the mock ClientEvent
    outbox_mock = tools_executor._toolset["run_command"].metadata["outbox"]

    outbox_mock.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_client_event
    )

    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "testing"}],
            tool_calls=[
                {
                    "id": "run-command-test",
                    "name": "run_command",
                    "args": {
                        "program": "echo",
                        "arguments": ["/home"],
                        "flags": ["-l"],
                    },
                }
            ],
            id="ai-msg-run-command",
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
                {"id": "tool-a-call", "name": "a", "args": {"a1": 1}},
                {"id": "tool-b-call", "name": "b", "args": {"b1": 1}},
            ],
            id="ai-msg-multiple-tools",
        )
    ]

    result = await graph.ainvoke(workflow_state)

    assert result["conversation_history"]["planner"][-2:] == [
        HumanMessage(content="a" * 10000),
        HumanMessage(content="tool b"),
    ]

    # Verify we have agent message + 2 tool messages
    assert len(result["ui_chat_log"]) == 3

    # Verify agent message
    assert result["ui_chat_log"][0] == {
        "message_type": MessageTypeEnum.AGENT,
        "message_sub_type": None,
        "content": "test",
        "timestamp": "2025-01-01T12:00:00+00:00",
        "status": ToolStatus.SUCCESS,
        "tool_info": None,
        "correlation_id": None,
        "additional_context": None,
        "message_id": "ai-msg-multiple-tools",
    }

    # Verify tool messages
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
            "message_id": "tool-a-call",
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
            "message_id": "tool-b-call",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("all_tools", [{"test_tool": mock_tool()}])
@pytest.mark.usefixtures("mock_datetime")
async def test_skip_agent_msg_prevents_duplicate_messages(
    all_tools, workflow_state, toolset, flow_type: GLReportingEventContext
):
    """Test that skip_agent_msg=True prevents agent messages from being added to ui_chat_log."""
    tool = all_tools["test_tool"]

    # Create ToolsExecutor with skip_agent_msg=True
    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=toolset,
        workflow_id="123",
        workflow_type=flow_type,
        skip_agent_msg=True,
    )

    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "I'll use a tool"}],
            tool_calls=[
                {
                    "id": "skip-agent-msg-tool-call",
                    "name": tool.name,
                    "args": {"tasks": [{"description": "step1"}]},
                }
            ],
            id="ai-msg-skip-test",
        )
    ]

    result = await tools_executor.run(workflow_state)

    update = cast(Command, result[-1]).update
    assert update and "ui_chat_log" in update
    ui_chat_log = update["ui_chat_log"]

    # Verify no agent message was added, only tool message
    assert len(ui_chat_log) == 1
    assert ui_chat_log[0]["message_type"] == MessageTypeEnum.TOOL
    assert (
        ui_chat_log[0]["content"] == "Using test_tool: tasks=[{'description': 'step1'}]"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("all_tools", [{"test_tool": mock_tool()}])
@pytest.mark.usefixtures("mock_datetime")
async def test_skip_agent_msg_false_adds_agent_message(
    all_tools, workflow_state, toolset, flow_type: GLReportingEventContext
):
    """Test that skip_agent_msg=False (default) adds agent messages to ui_chat_log."""
    tool = all_tools["test_tool"]

    # Create ToolsExecutor with skip_agent_msg=False (default)
    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=toolset,
        workflow_id="123",
        workflow_type=flow_type,
        skip_agent_msg=False,
    )

    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "I'll use a tool"}],
            tool_calls=[
                {
                    "id": "no-skip-agent-msg-tool-call",
                    "name": tool.name,
                    "args": {"tasks": [{"description": "step1"}]},
                }
            ],
            id="ai-msg-no-skip-test",
        )
    ]

    result = await tools_executor.run(workflow_state)

    update = cast(Command, result[-1]).update
    assert update and "ui_chat_log" in update
    ui_chat_log = update["ui_chat_log"]

    # Verify both agent message and tool message were added
    assert len(ui_chat_log) == 2
    assert ui_chat_log[0]["message_type"] == MessageTypeEnum.AGENT
    assert ui_chat_log[0]["content"] == "I'll use a tool"
    assert ui_chat_log[0]["message_id"] == "ai-msg-no-skip-test"
    assert ui_chat_log[1]["message_type"] == MessageTypeEnum.TOOL
    assert (
        ui_chat_log[1]["content"] == "Using test_tool: tasks=[{'description': 'step1'}]"
    )
    assert ui_chat_log[1]["message_id"] == "no-skip-agent-msg-tool-call"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_datetime")
@patch("duo_workflow_service.agents.tools_executor.apply_security_scanning")
@patch("duo_workflow_service.agents.tools_executor.log_exception")
async def test_security_exception_creates_failure_ui_chat_log(
    mock_log_exception,
    mock_apply_security,
    workflow_state,
):
    """SecurityException creates UiChatLog with ToolStatus.FAILURE."""
    scan_result = ScanResult(
        detected=True,
        blocked=True,
        detection_type=DetectionType.PROMPT_INJECTION,
        details="Malicious content detected",
    )
    security_error = PromptInjectionDetectedError(scan_result, "test_tool")
    mock_apply_security.side_effect = security_error

    tool = mock_tool(name="test_tool", content="tool response")
    mock_toolset = MagicMock(spec=Toolset)
    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=tool)
    mock_toolset.get = MagicMock(return_value=tool)

    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[{"id": "sec-test-1", "name": "test_tool", "args": {}}],
            id="ai-msg-security-test",
        )
    ]

    result = await tools_executor.run(workflow_state)

    # Verify security exception was logged
    mock_log_exception.assert_called_once()

    # Verify response content contains error message
    assert (
        "Security scan detected potentially malicious content"
        in result[0]["conversation_history"]["planner"][0].content
    )

    # Verify UI chat log has failure status
    update = cast(Command, result[-1]).update
    ui_chat_logs = update["ui_chat_log"]

    # Find the tool log (skip agent message)
    tool_log = next(
        log for log in ui_chat_logs if log["message_type"] == MessageTypeEnum.TOOL
    )

    assert tool_log["status"] == ToolStatus.FAILURE
    assert "Security error" in tool_log["content"]
    assert tool_log["tool_info"]["name"] == "test_tool"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_datetime")
async def test_invalid_tool_call_handling(
    workflow_state,
    internal_event_client: Mock,
    flow_type: GLReportingEventContext,
):
    """Test that invalid tool calls are handled gracefully with error messages and UI logs."""
    # Create invalid tool calls
    invalid_tool_calls = [
        InvalidToolCall(
            type="invalid_tool_call",
            id="invalid-call-1",
            error="JSON parsing error: unexpected token",
            name="invalid_tool",
            args="{}",
        ),
    ]

    # Setup workflow state with invalid tool calls
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[],  # No valid tool calls
            invalid_tool_calls=invalid_tool_calls,
            id="ai-msg-invalid-tool-test",
        ),
    ]

    # Create ToolsExecutor
    mock_toolset = MagicMock(spec=Toolset)
    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=flow_type,
        internal_event_client=internal_event_client,
    )

    # Execute
    result = await tools_executor.run(workflow_state)

    # Verify conversation history contains error response
    assert len(result) == 2
    error_response = result[0]["conversation_history"]["planner"][0]
    assert isinstance(error_response, ToolMessage)
    assert error_response.tool_call_id == "invalid-call-1"
    assert "Invalid or unparsable tool call received." in error_response.content

    # Verify UI chat log
    update = cast(Command, result[-1]).update
    assert update is not None
    ui_chat_logs = update["ui_chat_log"]

    # Should have agent message + invalid tool call error message
    assert len(ui_chat_logs) == 2

    # Verify agent message
    assert ui_chat_logs[0]["message_type"] == MessageTypeEnum.AGENT
    assert ui_chat_logs[0]["content"] == "test"
    assert ui_chat_logs[0]["message_id"] == "ai-msg-invalid-tool-test"
    assert ui_chat_logs[0]["status"] == ToolStatus.SUCCESS
    assert ui_chat_logs[0]["timestamp"] == "2025-01-01T12:00:00+00:00"

    # Verify invalid tool call error message
    assert ui_chat_logs[1]["message_type"] == MessageTypeEnum.TOOL
    assert ui_chat_logs[1]["status"] == ToolStatus.FAILURE
    assert "Tool call error" in ui_chat_logs[1]["content"]
    assert ui_chat_logs[1]["message_id"] == "invalid-call-1"
    assert ui_chat_logs[1]["tool_info"] is None
    assert ui_chat_logs[1]["timestamp"] == "2025-01-01T12:00:00+00:00"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_datetime")
async def test_invalid_tool_call_with_valid_tool_calls(
    workflow_state,
    internal_event_client: Mock,
    flow_type: GLReportingEventContext,
):
    """Test that invalid tool calls are handled alongside valid tool calls."""
    # Create mock tool
    tool = mock_tool(name="test_tool", content="test_tool result")

    # Create invalid tool calls
    invalid_tool_calls = [
        InvalidToolCall(
            type="invalid_tool_call",
            id="invalid-call-1",
            error="Malformed JSON",
            name="invalid_tool",
            args="{}",
        ),
    ]

    # Setup workflow state with both valid and invalid tool calls
    workflow_state["conversation_history"]["planner"] = [
        AIMessage(
            content=[{"type": "text", "text": "test"}],
            tool_calls=[
                {
                    "id": "valid-call-1",
                    "name": "test_tool",
                    "args": {"param": "value"},
                }
            ],
            invalid_tool_calls=invalid_tool_calls,
            id="ai-msg-mixed-tools-test",
        ),
    ]

    # Create mock toolset
    mock_toolset = MagicMock(spec=Toolset)
    mock_toolset.__contains__ = MagicMock(return_value=True)
    mock_toolset.__getitem__ = MagicMock(return_value=tool)

    # Create ToolsExecutor
    tools_executor = ToolsExecutor(
        tools_agent_name="planner",
        toolset=mock_toolset,
        workflow_id="123",
        workflow_type=flow_type,
        internal_event_client=internal_event_client,
    )

    # Execute
    result = await tools_executor.run(workflow_state)

    # Verify conversation history contains both responses
    assert len(result) == 3  # valid tool response + invalid tool response + Command

    # Verify valid tool response
    valid_response = result[0]["conversation_history"]["planner"][0]
    assert isinstance(valid_response, ToolMessage)
    # The mock tool returns a hardcoded tool_call_id, so we just verify it's a ToolMessage with content
    assert valid_response.content == "test_tool result"

    # Verify invalid tool response
    invalid_response = result[1]["conversation_history"]["planner"][0]
    assert isinstance(invalid_response, ToolMessage)
    assert invalid_response.tool_call_id == "invalid-call-1"
    assert "Invalid or unparsable tool call received." in invalid_response.content

    # Verify UI chat logs
    update = cast(Command, result[-1]).update
    assert update is not None
    ui_chat_logs = update["ui_chat_log"]

    # Should have agent message + valid tool message + invalid tool message
    assert len(ui_chat_logs) == 3

    # Verify agent message
    assert ui_chat_logs[0]["message_type"] == MessageTypeEnum.AGENT

    # Verify valid tool message
    assert ui_chat_logs[1]["message_type"] == MessageTypeEnum.TOOL
    assert ui_chat_logs[1]["status"] == ToolStatus.SUCCESS
    assert ui_chat_logs[1]["message_id"] == "valid-call-1"

    # Verify invalid tool message
    assert ui_chat_logs[2]["message_type"] == MessageTypeEnum.TOOL
    assert ui_chat_logs[2]["status"] == ToolStatus.FAILURE
    assert ui_chat_logs[2]["message_id"] == "invalid-call-1"
    assert ui_chat_logs[2]["message_id"] == "invalid-call-1"
