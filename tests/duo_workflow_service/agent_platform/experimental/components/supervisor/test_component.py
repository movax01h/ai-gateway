"""Test suite for SupervisorAgentComponent."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    RoutingError,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys
from duo_workflow_service.agent_platform.experimental.state.base import FlowState


class MockSubagentComponent:
    """Minimal stub satisfying the supervisor's subagent type check.

    The supervisor's validate_config checks for the ``_is_subagent_component``
    marker attribute (set as a ClassVar on the real SubagentComponent) to
    identify managed subagents without relying on class-name string comparison.

    Both ``attach`` and ``bind_to_supervisor`` are no-op Mocks, suitable for
    tests that only need to verify those calls were made.  Use
    ``RoutingMockSubagentComponent`` when the test needs to actually execute
    through the subagent node inside a compiled graph.
    """

    _is_subagent_component = True

    def __init__(self, name: str, description: str = "A test subagent."):
        self.name = name
        self.description = description
        self.attach = Mock()
        self.bind_to_supervisor = Mock()


class RoutingMockSubagentComponent:
    """Subagent stub that wires a real graph node and routes via the injected router.

    Unlike ``MockSubagentComponent``, this stub's ``attach`` adds an actual node
    to the graph.  The node's ``run`` immediately delegates to the router passed
    at attach-time, which is ``_SubagentRouter`` → ``supervisor#subagent_return``.
    This allows tests to execute the full delegation loop without needing a real
    ``SubagentComponent`` with all its dependencies.
    """

    _is_subagent_component = True

    def __init__(self, name: str, description: str = "A test subagent."):
        self.name = name
        self.description = description
        self._router = None

    def bind_to_supervisor(self, **_kwargs):
        """No-op: the routing stub does not need key factories."""

    def attach(self, graph: StateGraph, router) -> None:
        """Add a single pass-through node whose run routes straight to the router."""
        self._router = router

        def _run(state):
            return state

        graph.add_node(f"{self.name}#agent", _run)
        graph.add_conditional_edges(f"{self.name}#agent", router.route)

    def __entry_hook__(self) -> str:
        """Return the entry node name expected by _delegation_router."""
        return f"{self.name}#agent"


@pytest.fixture(name="mock_sub_agents")
def mock_sub_agents_fixture(developer_name, tester_name):
    """Create mock subagent components."""
    return {
        developer_name: MockSubagentComponent(name=developer_name),
        tester_name: MockSubagentComponent(name=tester_name),
    }


def _make_supervisor(
    supervisor_name,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
    managed_agent_names,
    max_delegations,
    subagent_components,
):
    """Helper to construct a SupervisorAgentComponent with common params."""
    return SupervisorAgentComponent(
        name=supervisor_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=[],
        prompt_id="supervisor_prompt",
        toolset=mock_toolset,
        prompt_registry=mock_prompt_registry,
        internal_event_client=mock_internal_event_client,
        managed_agents=managed_agent_names,
        max_delegations=max_delegations,
        subagent_components=subagent_components,
    )


# --- Node class mock fixtures ---

_SUPERVISOR_MODULE = (
    "duo_workflow_service.agent_platform.experimental.components.supervisor.component"
)


@pytest.fixture(name="mock_agent_node_cls")
def mock_agent_node_cls_fixture(supervisor_name):
    """Fixture for mocked AgentNode class in supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.AgentNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#agent"
        yield mock_cls


@pytest.fixture(name="mock_tool_node_cls")
def mock_tool_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolNode class in supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.ToolNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tools"
        yield mock_cls


@pytest.fixture(name="mock_final_response_node_cls")
def mock_final_response_node_cls_fixture(supervisor_name):
    """Fixture for mocked FinalResponseNode class in supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.FinalResponseNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#final_response"
        yield mock_cls


@pytest.fixture(name="mock_delegation_node_cls")
def mock_delegation_node_cls_fixture(supervisor_name):
    """Fixture for mocked DelegationNode class in supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.DelegationNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#delegation"
        yield mock_cls


@pytest.fixture(name="mock_subagent_return_node_cls")
def mock_subagent_return_node_cls_fixture(supervisor_name):
    """Fixture for mocked SubagentReturnNode class in supervisor component module."""
    with patch(f"{_SUPERVISOR_MODULE}.SubagentReturnNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#subagent_return"
        yield mock_cls


@pytest.fixture(name="all_node_mocks")
def all_node_mocks_fixture(
    mock_agent_node_cls,
    mock_tool_node_cls,
    mock_final_response_node_cls,
    mock_delegation_node_cls,
    mock_subagent_return_node_cls,
):
    """Activate all supervisor node mocks together."""
    return {
        "agent": mock_agent_node_cls.return_value,
        "tools": mock_tool_node_cls.return_value,
        "final_response": mock_final_response_node_cls.return_value,
        "delegation": mock_delegation_node_cls.return_value,
        "subagent_return": mock_subagent_return_node_cls.return_value,
    }


class TestSupervisorAgentComponentInit:
    """Tests for SupervisorAgentComponent initialization."""

    @pytest.mark.parametrize(
        ("managed_agents", "max_delegations", "subagent_components_key", "match"),
        [
            ([], 5, "empty", "at least one managed agent"),
            (None, 0, None, "max_delegations must be at least 1"),
            (None, -1, None, "max_delegations must be at least 1"),
            (None, 5, "empty", "at least one subagent component"),
            (None, 5, "missing_tester", "Managed agent 'tester' not found"),
            (None, 5, "wrong_type", "must be of type SubagentComponent"),
        ],
        ids=[
            "empty_managed_agents",
            "zero_max_delegations",
            "negative_max_delegations",
            "empty_subagent_components",
            "missing_subagent",
            "wrong_type_subagent",
        ],
    )
    def test_invalid_params_raise(
        self,
        supervisor_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        managed_agent_names,
        max_delegations,
        mock_sub_agents,
        developer_name,
        tester_name,
        managed_agents,
        subagent_components_key,
        match,
    ):
        """Test that invalid construction parameters raise ValueError."""
        subagent_components_by_key = {
            None: mock_sub_agents,
            "empty": {},
            "missing_tester": {
                developer_name: MockSubagentComponent(name=developer_name)
            },
            "wrong_type": {
                developer_name: Mock(),  # Mock auto-creates attributes; check uses `is True` so MagicMock != True
                tester_name: MockSubagentComponent(name=tester_name),
            },
        }
        with pytest.raises(ValueError, match=match):
            _make_supervisor(
                supervisor_name,
                flow_id,
                flow_type,
                user,
                mock_toolset,
                mock_prompt_registry,
                mock_internal_event_client,
                managed_agents if managed_agents is not None else managed_agent_names,
                max_delegations,
                subagent_components_by_key[subagent_components_key],
            )


class TestSupervisorExecutionFlow:
    """Tests for SupervisorAgentComponent execution via a real compiled graph."""

    @pytest.fixture(name="supervisor")
    def supervisor_fixture(
        self,
        supervisor_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        managed_agent_names,
        max_delegations,
        mock_sub_agents,
    ):
        """Supervisor instance used across execution-flow tests."""
        return _make_supervisor(
            supervisor_name,
            flow_id,
            flow_type,
            user,
            mock_toolset,
            mock_prompt_registry,
            mock_internal_event_client,
            managed_agent_names,
            max_delegations,
            mock_sub_agents,
        )

    def _compile(self, supervisor, mock_router):
        """Attach supervisor to a real StateGraph, set entry point, compile."""
        graph = StateGraph(FlowState)
        supervisor.attach(graph, mock_router)
        graph.set_entry_point(supervisor.__entry_hook__())
        return graph.compile()

    def test_agent_routes_directly_to_final_response(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
    ):
        """When the agent emits a text-only response (no tool calls), execution exits via the router."""
        nodes = all_node_mocks

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        nodes["agent"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        nodes["delegation"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_agent_routes_to_tools_then_final_response(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
    ):
        """Agent → tools → agent → final_response → exit."""
        nodes = all_node_mocks

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[regular_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["tools"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["delegation"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_agent_routes_to_delegation_then_back_to_agent_then_final_response(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        delegate_tool_call,
    ):
        """Agent → delegation → (no active subagent) → agent → final_response → exit.

        The delegation node returns state with active_subagent_type=None, so _delegation_router falls back to the
        supervisor agent node rather than routing into a subagent subgraph (which would require a real attached
        subagent).
        """
        nodes = all_node_mocks

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[delegate_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        # Delegation node returns no active subagent → _delegation_router falls back to supervisor#agent
        nodes["delegation"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_type": None,
                    "active_subsession": None,
                }
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["delegation"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        nodes["subagent_return"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_subagents_are_bound_and_attached(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        """Attach() calls bind_to_supervisor then attach on every subagent component."""
        nodes = all_node_mocks
        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        self._compile(supervisor, mock_router)

        mock_sub_agents[developer_name].bind_to_supervisor.assert_called_once()
        mock_sub_agents[tester_name].bind_to_supervisor.assert_called_once()
        mock_sub_agents[developer_name].attach.assert_called_once()
        mock_sub_agents[tester_name].attach.assert_called_once()

    def test_subagent_router_routes_back_to_subagent_return(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
    ):
        """The router passed to each subagent always routes back to #subagent_return."""
        nodes = all_node_mocks
        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        self._compile(supervisor, mock_router)

        sub_router = mock_sub_agents[developer_name].attach.call_args[0][1]
        assert sub_router.route(base_flow_state) == f"{supervisor_name}#subagent_return"

    @pytest.fixture(name="routing_supervisor")
    def routing_supervisor_fixture(
        self,
        supervisor_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        managed_agent_names,
        max_delegations,
        developer_name,
        tester_name,
    ):
        """Supervisor wired with RoutingMockSubagentComponent for the developer subagent."""
        routing_sub_agents = {
            developer_name: RoutingMockSubagentComponent(name=developer_name),
            tester_name: MockSubagentComponent(name=tester_name),
        }
        return _make_supervisor(
            supervisor_name,
            flow_id,
            flow_type,
            user,
            mock_toolset,
            mock_prompt_registry,
            mock_internal_event_client,
            managed_agent_names,
            max_delegations,
            routing_sub_agents,
        )

    def test_full_subagent_delegation_loop(
        self,
        routing_supervisor,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call,
    ):
        """Full loop: agent → delegation → subagent node → subagent_return → agent → final_response.

        Uses RoutingMockSubagentComponent for the developer subagent so the
        delegation path through the real graph can be exercised end-to-end
        without needing a fully wired SubagentComponent.
        """
        nodes = all_node_mocks

        nodes["agent"].run.side_effect = [
            # First call: delegate to developer subagent
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[delegate_tool_call])
                    ]
                },
            },
            # Second call: after subagent_return, emit text-only final response
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        # Delegation node routes to developer#agent via _delegation_router
        nodes["delegation"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_type": developer_name,
                    "active_subsession": 1,
                }
            },
        }
        # Subagent return clears active subagent → _delegation_router falls back to supervisor#agent
        nodes["subagent_return"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_type": None,
                    "active_subsession": None,
                }
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = self._compile(routing_supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["delegation"].run.assert_called_once()
        nodes["subagent_return"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_routing_errors_propagate(
        self,
        supervisor,
        all_node_mocks,
        mock_router,
        base_flow_state,
    ):
        """RoutingError from _agent_node_router propagates out of graph execution."""
        nodes = all_node_mocks
        # Agent returns state with no conversation history → router raises RoutingError
        nodes["agent"].run.return_value = {**base_flow_state}

        compiled = self._compile(supervisor, mock_router)

        with pytest.raises(RoutingError, match="Conversation history not found"):
            compiled.invoke(base_flow_state)
