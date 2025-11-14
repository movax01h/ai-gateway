"""Shared fixtures for agent component tests."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
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


@pytest.fixture(name="tool_call_id")
def tool_call_id_fixture():
    """Fixture for tool call ID."""
    return "test_tool_call_id"


@pytest.fixture(name="final_response_content")
def final_response_content_fixture():
    """Fixture for final response content."""
    return "Task completed successfully!"


@pytest.fixture(name="inputs")
def inputs_fixture():
    """Fixture for input IOKeys."""
    return [
        IOKey(target="context", subkeys=["user_input"]),
        IOKey(target="context", subkeys=["task_description"]),
    ]


@pytest.fixture(name="simple_output")
def simple_output_fixture():
    """Fixture for simple output IOKey."""
    return IOKey(target="context", subkeys=["result"])


@pytest.fixture(name="nested_output")
def nested_output_fixture():
    """Fixture for nested output IOKey."""
    return IOKey(target="context", subkeys=["workflow", "final", "response"])


@pytest.fixture(name="prompt_variables")
def prompt_variables_fixture():
    """Fixture for prompt variables."""
    return {
        "user_input": "Please help me with this task",
        "task_description": "Complete the workflow",
    }


@pytest.fixture(name="ui_history")
def ui_history_fixture():
    """Fixture for mock UIHistory."""
    mock_history = Mock(spec=UIHistory)
    mock_log = Mock()
    mock_history.log = mock_log
    mock_history.pop_state_updates.return_value = {FlowStateKeys.UI_CHAT_LOG: []}
    return mock_history


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


@pytest.fixture(name="flow_state_with_history")
def flow_state_with_history_fixture(base_flow_state, component_name):
    """Fixture for flow state with conversation history."""
    state = base_flow_state.copy()
    mock_message = Mock(spec=AIMessage)
    mock_message.content = "Previous response"
    mock_message.additional_kwargs = {}
    state["conversation_history"] = {component_name: [mock_message]}
    return state


@pytest.fixture(name="flow_state_empty_history")
def flow_state_empty_history_fixture(base_flow_state, component_name):
    """Fixture for flow state with empty conversation history."""
    state = base_flow_state.copy()
    state["conversation_history"] = {component_name: []}
    return state


@pytest.fixture(name="flow_state_no_history")
def flow_state_no_history_fixture(base_flow_state):
    """Fixture for flow state with no conversation history."""
    state = base_flow_state.copy()
    state["conversation_history"] = {}
    return state


@pytest.fixture(name="mock_ai_message")
def mock_ai_message_fixture():
    """Fixture for mock AI message."""
    mock_message = Mock(spec=AIMessage)
    mock_message.content = "Test response from agent"
    mock_message.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    mock_message.response_metadata = {
        "finish_reason": "stop"
    }  # OpenAI format used by LiteLLM
    mock_message.tool_calls = []
    mock_message.additional_kwargs = {}
    return mock_message


@pytest.fixture(name="mock_ai_message_no_metadata")
def mock_ai_message_no_metadata_fixture():
    """Fixture for mock AI message without metadata."""
    mock_message = Mock(spec=AIMessage)
    mock_message.content = "Test response without metadata"
    mock_message.usage_metadata = None
    mock_message.response_metadata = None
    mock_message.tool_calls = []
    mock_message.additional_kwargs = {}
    return mock_message


@pytest.fixture(name="mock_ai_message_empty_tools")
def mock_ai_message_empty_tools_fixture():
    """Fixture for mock AI message with empty tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = []
    return mock_message


@pytest.fixture(name="mock_final_tool_call")
def mock_final_tool_call_fixture(tool_call_id, final_response_content):
    """Fixture for mock final response tool call."""
    return {
        "id": tool_call_id,
        "name": AgentFinalOutput.tool_title,
        "args": {"final_response": final_response_content},
    }


@pytest.fixture(name="mock_other_tool_call")
def mock_other_tool_call_fixture():
    """Fixture for mock other tool call."""
    return {
        "id": "other_tool_id",
        "name": "other_tool",
        "args": {"param": "value"},
    }


@pytest.fixture(name="mock_tool_call")
def mock_tool_call_fixture():
    """Fixture for mock tool call."""
    return {
        "name": "test_tool",
        "args": {"param": "value"},
        "id": "test_tool_call_id",
    }


@pytest.fixture(name="mock_multiple_tool_calls")
def mock_multiple_tool_calls_fixture():
    """Fixture for multiple mock tool calls."""
    return [
        {
            "name": "tool_1",
            "args": {"param1": "value1"},
            "id": "tool_call_id_1",
        },
        {
            "name": "tool_2",
            "args": {"param2": "value2"},
            "id": "tool_call_id_2",
        },
    ]


@pytest.fixture(name="mock_ai_message_with_final_tool")
def mock_ai_message_with_final_tool_fixture(mock_final_tool_call):
    """Fixture for mock AI message with final response tool call."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [mock_final_tool_call]
    return mock_message


@pytest.fixture(name="mock_ai_message_with_multiple_tools")
def mock_ai_message_with_multiple_tools_fixture(
    mock_other_tool_call, mock_final_tool_call
):
    """Fixture for mock AI message with multiple tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [mock_other_tool_call, mock_final_tool_call]
    return mock_message


@pytest.fixture(name="mock_ai_message_without_final_tool")
def mock_ai_message_without_final_tool_fixture(mock_other_tool_call):
    """Fixture for mock AI message without final response tool call."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [mock_other_tool_call]
    return mock_message


@pytest.fixture(name="mock_ai_message_with_tool_calls")
def mock_ai_message_with_tool_calls_fixture(mock_tool_call):
    """Fixture for mock AI message with tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [mock_tool_call]
    return mock_message


@pytest.fixture(name="mock_ai_message_with_multiple_tool_calls")
def mock_ai_message_with_multiple_tool_calls_fixture(mock_multiple_tool_calls):
    """Fixture for mock AI message with multiple tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = mock_multiple_tool_calls
    return mock_message


@pytest.fixture(name="mock_ai_message_no_tool_calls")
def mock_ai_message_no_tool_calls_fixture():
    """Fixture for mock AI message without tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = []
    return mock_message


@pytest.fixture(name="flow_state_with_message")
def flow_state_with_message_fixture(
    base_flow_state, component_name, mock_ai_message_with_final_tool
):
    """Fixture for flow state with AI message."""
    state = base_flow_state.copy()
    state["conversation_history"] = {component_name: [mock_ai_message_with_final_tool]}
    return state


@pytest.fixture(name="flow_state_with_tool_calls")
def flow_state_with_tool_calls_fixture(
    base_flow_state, component_name, mock_ai_message_with_tool_calls
):
    """Fixture for flow state with tool calls."""
    state = base_flow_state.copy()
    state["conversation_history"] = {component_name: [mock_ai_message_with_tool_calls]}
    return state


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


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    """Fixture for mock prompt registry."""
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_prompt = Mock()
    mock_prompt.model = Mock()
    mock_prompt.model.model_name = "claude-3-sonnet"
    mock_registry.get.return_value = mock_prompt
    return mock_registry


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
