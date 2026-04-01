"""Shared fixtures for supervisor component tests."""

from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    DelegateTask,
    ManagedAgentConfig,
    build_delegate_task_model,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.experimental.state.base import RuntimeIOKey
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient

# --- Basic fixtures ---


@pytest.fixture(name="flow_id")
def flow_id_fixture():
    """Fixture for flow ID."""
    return "test_flow_id"


@pytest.fixture(name="flow_type")
def flow_type_fixture() -> GLReportingEventContext:
    """Fixture for flow type."""
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="supervisor_name")
def supervisor_name_fixture():
    """Fixture for supervisor component name."""
    return "supervisor"


@pytest.fixture(name="developer_name")
def developer_name_fixture():
    """Fixture for developer subagent name."""
    return "developer"


@pytest.fixture(name="tester_name")
def tester_name_fixture():
    """Fixture for tester subagent name."""
    return "tester"


@pytest.fixture(name="developer_description")
def developer_description_fixture():
    """Fixture for developer subagent description."""
    return "Implements code changes and features."


@pytest.fixture(name="tester_description")
def tester_description_fixture():
    """Fixture for tester subagent description."""
    return "Writes and runs tests."


@pytest.fixture(name="managed_agent_names")
def managed_agent_names_fixture(developer_name, tester_name):
    """Fixture for managed agent names list (plain strings, for tests that need just names)."""
    return [developer_name, tester_name]


@pytest.fixture(name="managed_agents_config")
def managed_agents_config_fixture(
    developer_name, developer_description, tester_name, tester_description
):
    """Fixture for managed agent config list (name + description) passed to build_delegate_task_model."""
    return [
        ManagedAgentConfig(name=developer_name, description=developer_description),
        ManagedAgentConfig(name=tester_name, description=tester_description),
    ]


@pytest.fixture(name="max_delegations")
def max_delegations_fixture():
    """Fixture for max delegations limit."""
    return 5


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


@pytest.fixture(name="ui_history")
def ui_history_fixture():
    """Fixture for mock UIHistory."""
    mock_history = Mock(spec=UIHistory)
    mock_log = Mock()
    mock_history.log = mock_log
    mock_history.pop_state_updates.return_value = {FlowStateKeys.UI_CHAT_LOG: []}
    return mock_history


# --- DelegateTask fixtures ---


@pytest.fixture(name="delegate_task_cls")
def delegate_task_cls_fixture(managed_agents_config):
    """Fixture for dynamically built DelegateTask model."""
    return build_delegate_task_model(managed_agents_config)


@pytest.fixture(name="delegate_tool_call_id")
def delegate_tool_call_id_fixture():
    """Fixture for delegate_task tool call ID."""
    return "delegate_call_123"


@pytest.fixture(name="delegate_tool_call")
def delegate_tool_call_fixture(delegate_tool_call_id, developer_name):
    """Fixture for a delegate_task tool call dict."""
    return {
        "id": delegate_tool_call_id,
        "name": DelegateTask.tool_title,
        "args": {
            "subagent_type": developer_name,
            "subsession_id": None,
            "prompt": "Implement the feature",
        },
    }


@pytest.fixture(name="delegate_tool_call_resume")
def delegate_tool_call_resume_fixture(delegate_tool_call_id, developer_name):
    """Fixture for a delegate_task tool call that resumes a session."""
    return {
        "id": delegate_tool_call_id,
        "name": DelegateTask.tool_title,
        "args": {
            "subagent_type": developer_name,
            "subsession_id": 1,
            "prompt": "Fix the bug in the implementation",
        },
    }


@pytest.fixture(name="final_response_tool_call")
def final_response_tool_call_fixture():
    """Fixture for a final_response_tool call dict."""
    return {
        "id": "final_call_456",
        "name": AgentFinalOutput.tool_title,
        "args": {"final_response": "All tasks completed."},
    }


@pytest.fixture(name="regular_tool_call")
def regular_tool_call_fixture():
    """Fixture for a regular tool call dict."""
    return {
        "id": "tool_call_789",
        "name": "read_file",
        "args": {"file_path": "README.md"},
    }


# --- AIMessage fixtures ---


@pytest.fixture(name="ai_message_with_delegate")
def ai_message_with_delegate_fixture(delegate_tool_call):
    """Fixture for AIMessage containing a delegate_task tool call."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = [delegate_tool_call]
    return msg


@pytest.fixture(name="ai_message_with_delegate_resume")
def ai_message_with_delegate_resume_fixture(delegate_tool_call_resume):
    """Fixture for AIMessage containing a delegate_task resume tool call."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = [delegate_tool_call_resume]
    return msg


@pytest.fixture(name="ai_message_with_final_response")
def ai_message_with_final_response_fixture(final_response_tool_call):
    """Fixture for AIMessage containing a final_response_tool call."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = [final_response_tool_call]
    return msg


@pytest.fixture(name="ai_message_with_regular_tool")
def ai_message_with_regular_tool_fixture(regular_tool_call):
    """Fixture for AIMessage containing a regular tool call."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = [regular_tool_call]
    return msg


@pytest.fixture(name="ai_message_no_tool_calls")
def ai_message_no_tool_calls_fixture():
    """Fixture for AIMessage with no tool calls."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = []
    return msg


# --- State fixtures ---


@pytest.fixture(name="base_flow_state")
def base_flow_state_fixture():
    """Fixture for base flow state."""
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {},
    }


@pytest.fixture(name="supervisor_flow_state")
def supervisor_flow_state_fixture(supervisor_name, base_flow_state):
    """Fixture for flow state with supervisor context initialized."""
    state = {**base_flow_state}
    state["context"] = {
        supervisor_name: {
            "max_subsession_id": 0,
            "active_subsession": None,
            "active_subagent_type": None,
            "delegation_count": 0,
        }
    }
    return state


@pytest.fixture(name="supervisor_state_with_active_subsession")
def supervisor_state_with_active_subsession_fixture(
    supervisor_name, developer_name, base_flow_state, delegate_tool_call
):
    """Fixture for flow state with an active subagent subsession."""
    state = {**base_flow_state}
    subsession_key = f"{supervisor_name}__{developer_name}__1"
    state["context"] = {
        supervisor_name: {
            "max_subsession_id": 1,
            "active_subsession": 1,
            "active_subagent_type": developer_name,
            "delegation_count": 1,
        }
    }
    ai_msg = Mock(spec=AIMessage)
    ai_msg.tool_calls = [delegate_tool_call]
    state["conversation_history"] = {
        supervisor_name: [ai_msg],
        subsession_key: [HumanMessage(content="Implement the feature")],
    }
    return state


@pytest.fixture(name="supervisor_state_with_completed_subsession")
def supervisor_state_with_completed_subsession_fixture(
    supervisor_name, developer_name, base_flow_state, delegate_tool_call
):
    """Fixture for flow state with a completed subagent subsession (final_answer written)."""
    state = {**base_flow_state}
    subsession_key = f"{supervisor_name}__{developer_name}__1"
    state["context"] = {
        supervisor_name: {
            "max_subsession_id": 1,
            "active_subsession": 1,
            "active_subagent_type": developer_name,
            "delegation_count": 1,
            developer_name: {
                "1": {"final_answer": "Implementation complete. Created feature X."}
            },
        }
    }
    ai_msg = Mock(spec=AIMessage)
    ai_msg.tool_calls = [delegate_tool_call]
    state["conversation_history"] = {
        supervisor_name: [ai_msg],
        subsession_key: [
            HumanMessage(content="Implement the feature"),
        ],
    }
    return state


# --- IOKey fixtures (optional=True so they don't KeyError on missing context) ---


@pytest.fixture(name="delegation_count_key")
def delegation_count_key_fixture(supervisor_name):
    """Fixture for delegation_count IOKey."""
    return IOKey(
        target="context", subkeys=[supervisor_name, "delegation_count"], optional=True
    )


@pytest.fixture(name="active_subsession_key")
def active_subsession_key_fixture(supervisor_name):
    """Fixture for active_subsession IOKey."""
    return IOKey(
        target="context", subkeys=[supervisor_name, "active_subsession"], optional=True
    )


@pytest.fixture(name="active_subagent_type_key")
def active_subagent_type_key_fixture(supervisor_name):
    """Fixture for active_subagent_type IOKey."""
    return IOKey(
        target="context",
        subkeys=[supervisor_name, "active_subagent_type"],
        optional=True,
    )


@pytest.fixture(name="max_subsession_id_key")
def max_subsession_id_key_fixture(supervisor_name):
    """Fixture for max_subsession_id IOKey."""
    return IOKey(
        target="context",
        subkeys=[supervisor_name, "max_subsession_id"],
        optional=True,
    )


@pytest.fixture(name="supervisor_history_key")
def supervisor_history_key_fixture(supervisor_name):
    """Fixture for supervisor conversation-history IOKey."""
    return IOKey(
        target="conversation_history",
        subkeys=[supervisor_name],
        optional=True,
    )


@pytest.fixture(name="supervisor_history_key_factory")
def supervisor_history_key_factory_fixture(supervisor_history_key):
    """Fixture for supervisor history key factory (state -> IOKey)."""
    return lambda _state: supervisor_history_key


@pytest.fixture(name="supervisor_history_runtime_key")
def supervisor_history_runtime_key_fixture(supervisor_history_key):
    """Fixture for supervisor history RuntimeIOKey."""
    return RuntimeIOKey(
        alias="conversation_history", factory=lambda _state: supervisor_history_key
    )


@pytest.fixture(name="subsession_history_key_factory")
def subsession_history_key_factory_fixture(supervisor_name):
    """Fixture for subsession history key factory (subagent_type, subsession_id) -> IOKey."""

    def factory(subagent_type: str, subsession_id: int) -> IOKey:
        key = f"{supervisor_name}__{subagent_type}__{subsession_id}"
        return IOKey(target="conversation_history", subkeys=[key], optional=True)

    return factory


# --- Mock graph/router fixtures ---


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


@pytest.fixture(name="mock_schema_registry")
def mock_schema_registry_fixture():
    """Fixture for mock response schema registry."""
    mock_registry = Mock(spec=BaseResponseSchemaRegistry)

    class CustomResponseTool(BaseModel):
        """Custom response schema for testing."""

        model_config = ConfigDict(frozen=True)

        tool_title: ClassVar[str] = "custom_response_tool"

        summary: str = Field(description="Summary of the result")
        score: int = Field(description="Score from 0 to 10")

        @classmethod
        def from_ai_message(cls, msg):
            """Build from AI message tool call."""
            return cls(**msg.tool_calls[0]["args"])

    mock_registry.get.return_value = CustomResponseTool
    return mock_registry
