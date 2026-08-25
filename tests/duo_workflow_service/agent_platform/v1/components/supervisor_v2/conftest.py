"""Shared fixtures for v2 supervisor component tests."""

from typing import ClassVar
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.nodes.agent_node import (
    AgentFinalOutput,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.component import (
    SupervisorAgentComponentV2,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    DelegateTask,
    SubagentDescriptor,
    build_delegate_task_model,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DelegationStatus,
    SubsessionRun,
)
from duo_workflow_service.agent_platform.v1.state import FlowState, FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.v1.state.base import RuntimeIOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.tools.toolset import Toolset
from duo_workflow_service.tracking.subagent_delegation import (
    SubagentDelegationTracker,
)
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


@pytest.fixture(name="subagent_names")
def subagent_names_fixture(developer_name, tester_name):
    """Fixture for subagents list of dicts, matching the YAML config format."""
    return [{"name": developer_name}, {"name": tester_name}]


@pytest.fixture(name="subagent_descriptors")
def subagent_descriptors_fixture(
    developer_name, developer_description, tester_name, tester_description
):
    """Fixture for subagent descriptor list (name + description)."""
    return [
        SubagentDescriptor(name=developer_name, description=developer_description),
        SubagentDescriptor(name=tester_name, description=tester_description),
    ]


@pytest.fixture(name="max_delegations")
def max_delegations_fixture():
    """Fixture for max delegations limit."""
    return 5


@pytest.fixture(name="mock_internal_event_client")
def mock_internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    """Fixture for mock prompt registry."""
    mock_registry = Mock(spec=LocalPromptRegistry)
    mock_prompt = Mock()
    mock_prompt.model = Mock()
    mock_prompt.model.model_name = "claude-3-sonnet"
    mock_registry.get_on_behalf.return_value = mock_prompt
    mock_registry.get_required_variables.return_value = set()
    return mock_registry


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
def delegate_task_cls_fixture(subagent_descriptors):
    """Fixture for dynamically built DelegateTask model."""
    return build_delegate_task_model(subagent_descriptors)


@pytest.fixture(name="delegate_tool_call_id")
def delegate_tool_call_id_fixture():
    """Fixture for delegate_task tool call ID."""
    return "delegate_call_123"


@pytest.fixture(name="delegate_tool_call")
def delegate_tool_call_fixture(delegate_tool_call_id, developer_name):
    """Fixture for a delegate_task tool call dict (new subsession)."""
    return {
        "id": delegate_tool_call_id,
        "name": DelegateTask.tool_title,
        "args": {
            "subagent_name": developer_name,
            "description": "Implement the feature",
            "prompt": "Implement the feature",
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
    """Fixture for AIMessage containing a single delegate_task tool call."""
    msg = Mock(spec=AIMessage)
    msg.tool_calls = [delegate_tool_call]
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
def base_flow_state_fixture() -> FlowState:
    """Fixture for base flow state."""
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {},
        "agent_context_limits": {},
    }


@pytest.fixture(name="supervisor_flow_state")
def supervisor_flow_state_fixture(supervisor_name, base_flow_state):
    """Fixture for flow state with supervisor context initialized."""
    state = {**base_flow_state}
    state["context"] = {
        supervisor_name: {
            "delegation_count": 0,
            "max_subsession_id": 0,
        }
    }
    return state


# --- IOKey fixtures (optional=True so they don't KeyError on missing context) ---


@pytest.fixture(name="delegation_count_key")
def delegation_count_key_fixture(supervisor_name):
    """Fixture for delegation_count IOKey."""
    return IOKey(
        target="context", subkeys=[supervisor_name, "delegation_count"], optional=True
    )


@pytest.fixture(name="max_subsession_id_key")
def max_subsession_id_key_fixture(supervisor_name):
    """Fixture for max_subsession_id IOKey."""
    return IOKey(
        target="context",
        subkeys=[supervisor_name, "max_subsession_id"],
        optional=True,
    )


@pytest.fixture(name="subsession_run_key_factory")
def subsession_run_key_factory_fixture(supervisor_name):
    """Fixture for the subsession run key factory: call_id -> IOKey.

    Mirrors ``SupervisorAgentComponentV2._subsession_run_key_factory``, the
    single place this naming convention lives in production.
    """

    def factory(call_id: str) -> IOKey:
        return IOKey(
            target="context",
            subkeys=[supervisor_name, "subsession_runs", call_id],
            optional=True,
        )

    return factory


@pytest.fixture(name="make_run_record")
def make_run_record_fixture():
    """Fixture returning a factory for ``SubsessionRun`` records.

    Defaults to a completed run with an answer; pass ``status``/``error``/
    ``final_answer`` to build the other outcomes ``SubagentDispatchNode``
    records.
    """

    def factory(
        subsession_id: int = 1,
        status: DelegationStatus = DelegationStatus.COMPLETED,
        error=None,
        final_answer="Task completed successfully",
    ) -> SubsessionRun:
        return SubsessionRun(
            subsession_id=subsession_id,
            status=status,
            error=error,
            final_answer=final_answer,
        )

    return factory


@pytest.fixture(name="supervisor_history_key")
def supervisor_history_key_fixture(supervisor_name):
    """Fixture for supervisor conversation-history IOKey."""
    return IOKey(
        target="conversation_history",
        subkeys=[supervisor_name],
        optional=True,
    )


@pytest.fixture(name="supervisor_history_runtime_key")
def supervisor_history_runtime_key_fixture(supervisor_history_key):
    """Fixture for supervisor history RuntimeIOKey."""
    return RuntimeIOKey(
        alias="conversation_history", factory=lambda _state: supervisor_history_key
    )


class MockSubagentComponent:
    """Minimal stub satisfying SupervisorAgentComponentV2's subagent type check.

    ``compile_as_subagent`` is a no-op Mock returning a sentinel, suitable for
    tests that only need to verify it was called. Use
    ``RoutingMockSubagentComponent`` when the test needs to actually execute
    through the dispatched subagent node inside a compiled graph.
    """

    def __init__(self, name: str, description: str = "A test subagent."):
        self.name = name
        self.description = description
        self.compile_as_subagent = Mock(return_value=Mock(name=f"{name}_compiled"))


class RoutingMockSubagentComponent:
    """Subagent stub compiling to a real, minimal single-node graph.

    Unlike ``MockSubagentComponent``, ``compile_as_subagent`` returns an
    actually-compiled ``StateGraph`` so tests can exercise
    ``SubagentDispatchNode.run`` (via ``ainvoke``) end-to-end without needing
    a fully wired ``AgentComponent``. The single node echoes back a
    component-scoped ``final_answer`` and conversation history, mirroring the
    contract ``AgentComponent.compile_as_subagent`` documents.
    """

    def __init__(self, name: str, description: str = "A test subagent.", answer=None):
        self.name = name
        self.description = description
        self._answer = answer or f"{name} finished the task."

    def compile_as_subagent(self):
        def _run(state):
            return {
                "conversation_history": {self.name: [AIMessage(content=self._answer)]},
                "context": {self.name: {"final_answer": self._answer}},
            }

        graph = StateGraph(FlowState)
        graph.add_node(self.name, _run)
        graph.set_entry_point(self.name)
        graph.set_finish_point(self.name)
        return graph.compile()


@pytest.fixture(name="mock_sub_agents")
def mock_sub_agents_fixture(
    developer_name, developer_description, tester_name, tester_description
):
    """Create mock subagent components."""
    return {
        developer_name: MockSubagentComponent(
            name=developer_name, description=developer_description
        ),
        tester_name: MockSubagentComponent(
            name=tester_name, description=tester_description
        ),
    }


# Sentinel used as a default for optional factory parameters -- distinguishes
# "caller did not pass this argument" from "caller explicitly passed None"
# (e.g. max_delegations=None means "no limit").
_UNSET = object()


@pytest.fixture(name="make_supervisor")
def make_supervisor_fixture(
    supervisor_name,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
    subagent_names,
    max_delegations,
    mock_sub_agents,
    mock_schema_registry,
):
    """Fixture that returns a factory for creating a SupervisorAgentComponentV2."""
    default_max_delegations = max_delegations
    default_subagent_names = subagent_names

    def factory(
        subagent_components=None,
        subagents=_UNSET,
        max_delegations=_UNSET,
        response_schema_id=None,
        response_schema_version=None,
        **kwargs,
    ):
        return SupervisorAgentComponentV2(
            name=supervisor_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            subagents=default_subagent_names if subagents is _UNSET else subagents,
            max_delegations=default_max_delegations
            if max_delegations is _UNSET
            else max_delegations,
            subagent_components=mock_sub_agents
            if subagent_components is None
            else subagent_components,
            schema_registry=mock_schema_registry,
            response_schema_id=response_schema_id,
            response_schema_version=response_schema_version,
            **kwargs,
        )

    return factory


_AGENT_COMPONENT_MODULE = (
    "duo_workflow_service.agent_platform.v1.components.agent.component"
)

_SUPERVISOR_MODULE = (
    "duo_workflow_service.agent_platform.v1.components.supervisor_v2.component"
)


def _compile(supervisor, mock_router):
    """Attach supervisor to a real StateGraph, set entry point, and compile it."""
    graph = StateGraph(FlowState)
    supervisor.attach(graph, mock_router)
    graph.set_entry_point(supervisor.__entry_hook__())
    return graph.compile()


# --- Node class mock fixtures ---


@pytest.fixture(name="mock_agent_node_cls")
def mock_agent_node_cls_fixture(supervisor_name):
    """Fixture for mocked AgentNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.AgentNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#agent"
        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.ToolNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tools"
        yield mock_cls


@pytest.fixture(name="mock_final_response_node_cls")
def mock_final_response_node_cls_fixture(supervisor_name):
    """Fixture for mocked FinalResponseNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.FinalResponseNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#final_response"
        yield mock_cls


@pytest.fixture(name="mock_delegation_prepare_node_cls")
def mock_delegation_prepare_node_cls_fixture(supervisor_name):
    """Fixture for mocked DelegationPrepareNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.DelegationPrepareNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#delegation_prepare"
        yield mock_cls


@pytest.fixture(name="mock_delegation_collect_node_cls")
def mock_delegation_collect_node_cls_fixture(supervisor_name):
    """Fixture for mocked DelegationCollectNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.DelegationCollectNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#delegation_collect"
        yield mock_cls


@pytest.fixture(name="mock_subagent_dispatch_node_cls")
def mock_subagent_dispatch_node_cls_fixture():
    """Fixture for mocked SubagentDispatchNode class in the v2 supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.SubagentDispatchNode") as mock_cls:
        yield mock_cls


@pytest.fixture(name="all_node_mocks")
def all_node_mocks_fixture(
    mock_agent_node_cls,
    mock_tool_node_cls,
    mock_final_response_node_cls,
    mock_delegation_prepare_node_cls,
    mock_delegation_collect_node_cls,
):
    """Activate all supervisor node mocks together (subagent dispatch nodes excluded)."""
    return {
        "agent": mock_agent_node_cls.return_value,
        "tools": mock_tool_node_cls.return_value,
        "final_response": mock_final_response_node_cls.return_value,
        "delegation_prepare": mock_delegation_prepare_node_cls.return_value,
        "delegation_collect": mock_delegation_collect_node_cls.return_value,
    }


# --- Mock graph/router fixtures ---


@pytest.fixture(name="mock_router")
def mock_router_fixture():
    """Fixture for mock router."""
    mock_router = Mock()
    mock_router.route.return_value = END
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


@pytest.fixture(name="delegation_tracker")
def delegation_tracker_fixture(
    flow_id, flow_type, mock_internal_event_client, supervisor_name
):
    """Real tracker over the mock client, so tests assert the payload actually emitted."""
    return SubagentDelegationTracker(
        flow_id=flow_id,
        flow_type=flow_type,
        internal_event_client=mock_internal_event_client,
        supervisor_name=supervisor_name,
        parallel=True,
    )
