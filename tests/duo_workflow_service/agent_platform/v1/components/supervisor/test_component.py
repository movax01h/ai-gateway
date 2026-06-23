"""Test suite for v1 SupervisorAgentComponent."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import RuntimeIOKey

from .conftest import MockSubagentComponent, RoutingMockSubagentComponent, _compile


class TestSupervisorAgentComponentInit:
    """Tests for SupervisorAgentComponent initialization."""

    @pytest.mark.parametrize(
        (
            "subagents_override",
            "max_delegations_override",
            "subagent_components_key",
            "match",
        ),
        [
            ([], 5, "empty", "at least one managed agent"),
            (None, 0, None, "max_delegations must be at least 1"),
            (None, -1, None, "max_delegations must be at least 1"),
            (None, 5, "empty", "not found in subagent_components"),
            (None, 5, "missing_tester", "Managed agent 'tester' not found"),
            (None, 5, "wrong_type", "does not have a bind_to_supervisor method"),
            (
                [{"name": "developer"}, "tester"],
                5,
                None,
                "must be a dict with a 'name' key",
            ),
            (
                [{"role": "developer"}],
                5,
                None,
                "must be a dict with a 'name' key",
            ),
            (
                [{"name": "developer"}, {"name": "developer"}],
                5,
                None,
                "Duplicate subagent name 'developer'",
            ),
        ],
        ids=[
            "empty_subagents",
            "zero_max_delegations",
            "negative_max_delegations",
            "empty_subagent_components",
            "missing_subagent",
            "wrong_type_subagent",
            "entry_is_plain_string",
            "entry_dict_missing_name_key",
            "duplicate_subagent_name",
        ],
    )
    def test_invalid_params_raise(
        self,
        make_supervisor,
        mock_sub_agents,
        developer_name,
        tester_name,
        subagent_names,
        subagents_override,
        max_delegations_override,
        subagent_components_key,
        match,
    ):
        """Test that invalid construction parameters raise ValueError."""

        # Create a non-bindable component (no bind_to_supervisor method)
        class NonBindableComponent:
            def __init__(self, name: str):
                self.name = name

        subagent_components_by_key = {
            None: mock_sub_agents,
            "empty": {},
            "missing_tester": {
                developer_name: MockSubagentComponent(name=developer_name)
            },
            "wrong_type": {
                developer_name: NonBindableComponent(name=developer_name),
                tester_name: MockSubagentComponent(name=tester_name),
            },
        }
        with pytest.raises(ValueError, match=match):
            make_supervisor(
                subagents=subagents_override
                if subagents_override is not None
                else subagent_names,
                max_delegations=max_delegations_override,
                subagent_components=subagent_components_by_key[subagent_components_key],
            )

    def test_none_subagents_raises(self, make_supervisor):
        """Test that passing subagents=None raises ValueError."""
        with pytest.raises(ValueError, match="at least one managed agent"):
            make_supervisor(subagents=None)

    def test_none_max_delegations_is_valid(self, make_supervisor):
        """Test that omitting max_delegations (None) is valid and imposes no delegation limit."""
        supervisor = make_supervisor(max_delegations=None)
        assert supervisor.max_delegations is None


class TestSupervisorExecutionFlow:
    """Tests for SupervisorAgentComponent execution via a real compiled graph."""

    @pytest.mark.parametrize(
        ("response_schema_id", "response_schema_version", "final_tool_calls"),
        [
            (None, None, []),
            (
                "general/structured_response",
                "1.0.0",
                [
                    {
                        "id": "schema_call_789",
                        "name": "custom_response_tool",
                        "args": {"summary": "All done", "score": 10},
                    }
                ],
            ),
        ],
    )
    def test_agent_routes_directly_to_final_response(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        mock_toolset,
        make_supervisor,
        response_schema_id,
        response_schema_version,
        final_tool_calls,
    ):
        """When the agent emits a text-only response (no tool calls), execution exits via the router."""
        nodes = all_node_mocks

        # Fix mock_toolset to not report collision with schema tool
        mock_toolset.__contains__.side_effect = lambda name: (
            name != "custom_response_tool"
        )

        # Create supervisor with schema params
        supervisor = make_supervisor(
            response_schema_id=response_schema_id,
            response_schema_version=response_schema_version,
        )

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [
                    AIMessage(content="All done.", tool_calls=final_tool_calls)
                ]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        nodes["agent"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        nodes["delegation"].run.assert_not_called()
        mock_router.route.assert_called_once()

    @pytest.mark.parametrize(
        ("response_schema_id", "response_schema_version", "final_tool_calls"),
        [
            (None, None, []),
            (
                "general/structured_response",
                "1.0.0",
                [
                    {
                        "id": "schema_call_789",
                        "name": "custom_response_tool",
                        "args": {"summary": "All done", "score": 10},
                    }
                ],
            ),
        ],
    )
    def test_agent_routes_to_tools_then_final_response(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        mock_toolset,
        make_supervisor,
        response_schema_id,
        response_schema_version,
        final_tool_calls,
    ):
        """Agent → tools → agent → final_response → exit."""
        nodes = all_node_mocks

        # Fix mock_toolset to not report collision with schema tool
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )

        # Create supervisor with schema params
        supervisor = make_supervisor(
            response_schema_id=response_schema_id,
            response_schema_version=response_schema_version,
        )

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
                    supervisor_name: [
                        AIMessage(content="All done.", tool_calls=final_tool_calls)
                    ]
                },
            },
        ]
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["tools"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["delegation"].run.assert_not_called()
        mock_router.route.assert_called_once()

    @pytest.mark.parametrize(
        ("response_schema_id", "response_schema_version", "final_tool_calls"),
        [
            (None, None, []),
            (
                "general/structured_response",
                "1.0.0",
                [
                    {
                        "id": "schema_call_789",
                        "name": "custom_response_tool",
                        "args": {"summary": "All done", "score": 10},
                    }
                ],
            ),
        ],
    )
    def test_agent_routes_to_delegation_then_back_to_agent_then_final_response(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        delegate_tool_call,
        mock_toolset,
        make_supervisor,
        response_schema_id,
        response_schema_version,
        final_tool_calls,
    ):
        """Agent → delegation → (no active subagent) → agent → final_response → exit.

        The delegation node returns state with active_subagent_name=None, so _delegation_router falls back to the
        supervisor agent node rather than routing into a subagent subgraph.
        """
        nodes = all_node_mocks

        # Fix mock_toolset to not report collision with schema tool
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )

        # Create supervisor with schema params
        supervisor = make_supervisor(
            response_schema_id=response_schema_id,
            response_schema_version=response_schema_version,
        )

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
                    supervisor_name: [
                        AIMessage(content="All done.", tool_calls=final_tool_calls)
                    ]
                },
            },
        ]
        # Delegation node returns no active subagent → _delegation_router falls back to supervisor#agent
        nodes["delegation"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_name": None,
                    "active_subsession": None,
                }
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["delegation"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        nodes["subagent_return"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_subagents_are_bound_and_attached(
        self,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
        make_supervisor,
    ):
        """Attach() calls bind_to_supervisor then attach on every subagent component."""
        nodes = all_node_mocks

        supervisor = make_supervisor()

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        _compile(supervisor, mock_router)

        mock_sub_agents[developer_name].bind_to_supervisor.assert_called_once()
        mock_sub_agents[tester_name].bind_to_supervisor.assert_called_once()
        mock_sub_agents[developer_name].attach.assert_called_once()
        mock_sub_agents[tester_name].attach.assert_called_once()

    def test_bind_to_supervisor_passes_subsession_scoped_tool_approval_decision_key(
        self,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
        make_supervisor,
    ):
        """Attach() passes a subsession-scoped tool_approval_decision_key to each subagent.

        The key must resolve to a path that includes the supervisor name, subagent name, and subsession ID — not just
        the component name — to prevent race conditions when the same subagent runs in multiple subsessions.
        """
        nodes = all_node_mocks

        supervisor = make_supervisor()

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        _compile(supervisor, mock_router)

        # Verify that bind_to_supervisor was called with a tool_approval_decision_key
        # for the developer subagent
        dev_call_kwargs = mock_sub_agents[developer_name].bind_to_supervisor.call_args[
            1
        ]
        assert "tool_approval_decision_key" in dev_call_kwargs

        # The key should be a RuntimeIOKey that resolves to a subsession-scoped path
        tool_approval_key = dev_call_kwargs["tool_approval_decision_key"]
        assert isinstance(tool_approval_key, RuntimeIOKey)

        # Simulate a state with an active subsession to verify the resolved key path
        state_with_subsession = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subsession": 1,
                }
            },
        }
        resolved_key = tool_approval_key.to_iokey(state_with_subsession)
        assert resolved_key.target == "context"
        # Key should be namespaced with supervisor, subagent, and subsession ID as separate subkeys
        assert resolved_key.subkeys[0] == supervisor_name
        assert resolved_key.subkeys[1] == developer_name
        assert resolved_key.subkeys[2] == "1"
        assert resolved_key.subkeys[3] == "tool_approval_decision"

    def test_subagent_router_routes_back_to_subagent_return(
        self,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
        make_supervisor,
    ):
        """The router passed to each subagent always routes back to #subagent_return."""
        nodes = all_node_mocks

        supervisor = make_supervisor()

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        _compile(supervisor, mock_router)

        sub_router = mock_sub_agents[developer_name].attach.call_args[0][1]
        assert sub_router.route(base_flow_state) == f"{supervisor_name}#subagent_return"

    @pytest.mark.parametrize(
        ("response_schema_id", "response_schema_version", "final_tool_calls"),
        [
            (None, None, []),
            (
                "general/structured_response",
                "1.0.0",
                [
                    {
                        "id": "schema_call_789",
                        "name": "custom_response_tool",
                        "args": {"summary": "All done", "score": 10},
                    }
                ],
            ),
        ],
    )
    def test_full_subagent_delegation_loop(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call,
        mock_toolset,
        tester_name,
        make_supervisor,
        response_schema_id,
        response_schema_version,
        final_tool_calls,
    ):
        """Full loop: agent → delegation → subagent node → subagent_return → agent → final_response.

        Uses RoutingMockSubagentComponent for the developer subagent so the
        delegation path through the real graph can be exercised end-to-end
        without needing a fully wired SubagentComponent.
        """
        nodes = all_node_mocks

        # Fix mock_toolset to not report collision with schema tool
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )

        # Create routing supervisor with schema params
        routing_sub_agents = {
            developer_name: RoutingMockSubagentComponent(name=developer_name),
            tester_name: MockSubagentComponent(name=tester_name),
        }
        routing_supervisor = make_supervisor(
            subagent_components=routing_sub_agents,
            response_schema_id=response_schema_id,
            response_schema_version=response_schema_version,
        )

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
            # Second call: after subagent_return, emit final response
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="All done.", tool_calls=final_tool_calls)
                    ]
                },
            },
        ]
        # Delegation node routes to developer#agent via _delegation_router
        nodes["delegation"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_name": developer_name,
                    "active_subsession": 1,
                }
            },
        }
        # Subagent return clears active subagent → _delegation_router falls back to supervisor#agent
        nodes["subagent_return"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "active_subagent_name": None,
                    "active_subsession": None,
                }
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(routing_supervisor, mock_router)
        compiled.invoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["delegation"].run.assert_called_once()
        nodes["subagent_return"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_routing_errors_propagate(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        make_supervisor,
    ):
        """RoutingError from _agent_node_router propagates out of graph execution."""
        nodes = all_node_mocks

        supervisor = make_supervisor()

        # Agent returns state with no conversation history → router raises RoutingError
        nodes["agent"].run.return_value = {**base_flow_state}

        compiled = _compile(supervisor, mock_router)

        with pytest.raises(RoutingError, match="Conversation history not found"):
            compiled.invoke(base_flow_state)
