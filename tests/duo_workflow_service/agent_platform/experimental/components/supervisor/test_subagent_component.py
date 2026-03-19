"""Test suite for SubagentComponent."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    RoutingError,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.subagent_component import (
    SubagentComponent,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowStateKeys,
    IOKey,
)
from duo_workflow_service.agent_platform.experimental.state.base import FlowState

_SUBAGENT_MODULE = (
    "duo_workflow_service.agent_platform.experimental"
    ".components.supervisor.subagent_component"
)


# --- Shared helpers ---


def _make_subagent(
    developer_name,
    developer_description,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
    **extra_kwargs,
):
    """Construct an unbound SubagentComponent with common params."""
    return SubagentComponent(
        name=developer_name,
        description=developer_description,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=[],
        prompt_id="test_prompt",
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        **extra_kwargs,
    )


def _bind(subagent, supervisor_name, developer_name):
    """Bind a subagent to a mock supervisor using session-scoped key factories."""
    session_key = f"{supervisor_name}__{developer_name}__1"
    subagent.bind_to_supervisor(
        conversation_history_key_factory=lambda _state: IOKey(
            target="conversation_history", subkeys=[session_key], optional=True
        ),
        output_key_factory=lambda _state: IOKey(
            target="context",
            subkeys=[supervisor_name, developer_name, "1", "final_answer"],
            optional=True,
        ),
    )
    return subagent


# --- Node class mock fixtures ---


@pytest.fixture(name="mock_agent_node_cls")
def mock_agent_node_cls_fixture(developer_name):
    """Fixture for mocked AgentNode class in subagent_component module."""
    with patch(f"{_SUBAGENT_MODULE}.AgentNode") as mock_cls:
        mock_cls.return_value.name = f"{developer_name}#agent"
        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(developer_name):
    """Fixture for mocked ToolNode class in subagent_component module."""
    with patch(f"{_SUBAGENT_MODULE}.ToolNode") as mock_cls:
        mock_cls.return_value.name = f"{developer_name}#tools"
        yield mock_cls


@pytest.fixture(name="mock_final_response_node_cls")
def mock_final_response_node_cls_fixture(developer_name):
    """Fixture for mocked FinalResponseNode class in subagent_component module."""
    with patch(f"{_SUBAGENT_MODULE}.FinalResponseNode") as mock_cls:
        mock_cls.return_value.name = f"{developer_name}#final_response"
        yield mock_cls


@pytest.fixture(name="all_node_mocks")
def all_node_mocks_fixture(
    mock_agent_node_cls,
    mock_tool_node_cls,
    mock_final_response_node_cls,
):
    """Activate all subagent node mocks and return instances keyed by role."""
    return {
        "agent": mock_agent_node_cls.return_value,
        "tools": mock_tool_node_cls.return_value,
        "final_response": mock_final_response_node_cls.return_value,
    }


# --- Shared subagent fixtures ---


@pytest.fixture(name="unbound_subagent")
def unbound_subagent_fixture(
    developer_name,
    developer_description,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    """Unbound SubagentComponent (bind_to_supervisor not yet called)."""
    return _make_subagent(
        developer_name,
        developer_description,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
    )


@pytest.fixture(name="subagent")
def subagent_fixture(unbound_subagent, supervisor_name, developer_name):
    """Bound SubagentComponent ready for attach."""
    return _bind(unbound_subagent, supervisor_name, developer_name)


def _final_response_state(base_flow_state, session_key):
    """Return state with a text-only AIMessage (implicit final answer)."""
    return {
        **base_flow_state,
        FlowStateKeys.CONVERSATION_HISTORY: {
            session_key: [AIMessage(content="Done", tool_calls=[])]
        },
    }


class TestSubagentComponentBindToSupervisor:
    """Tests for SubagentComponent.bind_to_supervisor."""

    def test_bind_to_supervisor_sets_factories(self, unbound_subagent):
        """bind_to_supervisor stores the injected key factories."""
        history_factory = Mock()
        output_factory = Mock()

        unbound_subagent.bind_to_supervisor(
            conversation_history_key_factory=history_factory,
            output_key_factory=output_factory,
        )

        assert unbound_subagent._conversation_history_key_factory is history_factory
        assert unbound_subagent._output_key_factory is output_factory

    def test_attach_raises_if_not_bound(self, unbound_subagent, mock_router):
        """Attach raises RuntimeError if bind_to_supervisor was not called first."""
        graph = StateGraph(FlowState)
        with pytest.raises(RuntimeError, match="must be bound to a supervisor"):
            unbound_subagent.attach(graph, mock_router)


class TestSubagentExecutionFlow:
    """Tests for SubagentComponent execution via a real compiled graph."""

    def _compile(self, subagent, mock_router):
        """Attach subagent to a real StateGraph, set entry point, compile."""
        graph = StateGraph(FlowState)
        subagent.attach(graph, mock_router)
        graph.set_entry_point(subagent.__entry_hook__())
        return graph.compile()

    def test_agent_routes_directly_to_final_response(
        self,
        subagent,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        developer_name,
    ):
        """When the agent emits a text-only response (no tool calls), execution exits via the router."""
        session_key = f"{supervisor_name}__{developer_name}__1"
        nodes = all_node_mocks
        nodes["agent"].run.return_value = _final_response_state(
            base_flow_state, session_key
        )
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        self._compile(subagent, mock_router).invoke(base_flow_state)

        nodes["agent"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_agent_routes_to_tools_then_final_response(
        self,
        subagent,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        developer_name,
    ):
        """Agent → tools → agent → final_response → exit."""
        session_key = f"{supervisor_name}__{developer_name}__1"
        nodes = all_node_mocks
        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    session_key: [
                        AIMessage(
                            content="",
                            tool_calls=[{"id": "c1", "name": "read_file", "args": {}}],
                        )
                    ]
                },
            },
            _final_response_state(base_flow_state, session_key),
        ]
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        self._compile(subagent, mock_router).invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["tools"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        mock_router.route.assert_called_once()

    def test_routing_errors_propagate(
        self,
        subagent,
        all_node_mocks,
        mock_router,
        base_flow_state,
    ):
        """RoutingError from _agent_node_router propagates out of graph execution."""
        nodes = all_node_mocks
        nodes["agent"].run.return_value = {**base_flow_state}

        with pytest.raises(RoutingError, match="Conversation history not found"):
            self._compile(subagent, mock_router).invoke(base_flow_state)
