"""Shared fixtures for OneOff component tests."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from duo_workflow_service.agent_platform.experimental.components.one_off.ui_log import (
    UILogEventsOneOff,
    UILogWriterOneOffTools,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events import InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="flow_id")
def flow_id_fixture():
    """Fixture for flow ID."""
    return "test_flow_id"


@pytest.fixture(name="component_name")
def component_name_fixture():
    """Fixture for component name."""
    return "test_component"


@pytest.fixture(name="flow_type")
def flow_type_fixture():
    """Fixture for flow type."""
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture(name="inputs")
def inputs_fixture():
    """Fixture for input IOKeys."""
    return [
        IOKey(target="context", subkeys=["user_input"]),
        IOKey(target="context", subkeys=["task_description"]),
    ]


@pytest.fixture(name="prompt_variables")
def prompt_variables_fixture():
    """Fixture for prompt variables."""
    return {
        "user_input": "Please help me with this task",
        "task_description": "Complete the workflow",
    }


@pytest.fixture(name="base_flow_state")
def base_flow_state_fixture(prompt_variables):
    """Fixture for base flow state."""
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {
            **prompt_variables,
        },
    }


@pytest.fixture(name="mock_internal_event_client")
def mock_internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    """Fixture for mock tool."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.arun = AsyncMock(return_value="Tool execution result")
    return mock_tool


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    """Fixture for mock toolset."""
    mock_toolset = Mock(spec=Toolset)
    mock_toolset.__contains__ = Mock(return_value=True)
    mock_toolset.__getitem__ = Mock(return_value=mock_tool)
    mock_toolset.bindable = [mock_tool]
    return mock_toolset


@pytest.fixture(name="mock_router")
def mock_router_fixture():
    """Fixture for mock router."""
    mock_router = Mock()
    mock_router.route.return_value = "END"
    return mock_router


@pytest.fixture(name="mock_state_graph")
def mock_state_graph_fixture():
    """Fixture for mock StateGraph."""
    mock_graph = Mock()
    mock_graph.add_node = Mock()
    mock_graph.add_edge = Mock()
    mock_graph.add_conditional_edges = Mock()
    return mock_graph


@pytest.fixture(name="ui_history_one_off")
def ui_history_one_off_fixture():
    """Fixture for UIHistory with OneOff-specific writer."""
    ui_history = Mock(spec=UIHistory)
    ui_history.log = Mock(spec=UILogWriterOneOffTools)
    ui_history.log.success = Mock()
    ui_history.log.error = Mock()
    ui_history.log.warning = Mock()
    ui_history.pop_state_updates = Mock(return_value={FlowStateKeys.UI_CHAT_LOG: []})
    return ui_history


@pytest.fixture(name="ui_log_events_one_off")
def ui_log_events_one_off_fixture():
    """Fixture for OneOff UI log events list."""
    return [
        UILogEventsOneOff.ON_TOOL_CALL_INPUT,
        UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
        UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
        UILogEventsOneOff.ON_AGENT_FINAL_ANSWER,
    ]


@pytest.fixture(name="mock_correction_context")
def mock_correction_context_fixture():
    """Fixture for error correction context."""
    return {
        "execution_status": "error",
        "correction_attempts": 1,
        "last_error": "Tool execution failed",
    }


@pytest.fixture(name="flow_state_with_correction_context")
def flow_state_with_correction_context_fixture(
    base_flow_state, component_name, mock_correction_context
):
    """Fixture for flow state with error correction context."""
    state = base_flow_state.copy()
    state["context"] = {
        **state.get("context", {}),
        component_name: mock_correction_context,
    }
    return state


@pytest.fixture(name="mock_tool_calls_key")
def mock_tool_calls_key_fixture(component_name):
    """Fixture for tool calls IOKey."""
    return IOKey(target="context", subkeys=[component_name, "tool_calls"])


@pytest.fixture(name="mock_tool_responses_key")
def mock_tool_responses_key_fixture(component_name):
    """Fixture for tool responses IOKey."""
    return IOKey(target="context", subkeys=[component_name, "tool_responses"])


@pytest.fixture(name="mock_tool_call")
def mock_tool_call_fixture():
    """Fixture for mock tool call."""
    return {
        "name": "test_tool",
        "args": {"param": "value"},
        "id": "test_tool_call_id",
    }


@pytest.fixture(name="mock_ai_message_with_tool_calls")
def mock_ai_message_with_tool_calls_fixture(mock_tool_call):
    """Fixture for mock AI message with tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [mock_tool_call]
    return mock_message


@pytest.fixture(name="mock_ai_message_no_tool_calls")
def mock_ai_message_no_tool_calls_fixture():
    """Fixture for mock AI message without tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = []
    return mock_message
