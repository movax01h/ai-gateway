# `compile_as_subagent` is a method of `AgentComponent` (agent/component.py), not its own
# module, so the file-naming-for-tests 1-source-to-1-test heuristic doesn't apply here.
# pylint: disable=file-naming-for-tests
"""Test suite for AgentComponent.compile_as_subagent (the SupervisorAgentComponentV2 dispatch path)."""

from unittest.mock import Mock

import pytest
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    SUBSESSION_ID_CONTEXT_KEY,
    AgentComponent,
    _TerminalRouter,
)
from duo_workflow_service.agent_platform.v1.state.base import RuntimeIOKey


@pytest.fixture(name="make_agent_component")
def make_agent_component_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    mock_toolset,
    mock_prompt_registry,
    mock_internal_event_client,
):
    """Factory building a minimal AgentComponent, overridable per test."""

    def _make(**overrides):
        kwargs = {
            "name": component_name,
            "flow_id": flow_id,
            "flow_type": flow_type,
            "user": user,
            "prompt_id": "test_prompt_id",
            "toolset": mock_toolset,
            "prompt_registry": mock_prompt_registry,
            "internal_event_client": mock_internal_event_client,
        }
        kwargs.update(overrides)
        return AgentComponent(**kwargs)

    return _make


@pytest.fixture(name="subagent_component")
def subagent_component_fixture(make_agent_component):
    """Fixture for an AgentComponent with a description, ready to compile_as_subagent."""
    return make_agent_component(description="A test subagent.")


class TestCompileAsSubagentValidation:
    """Tests for compile_as_subagent's validation."""

    def test_requires_description(self, make_agent_component):
        component = make_agent_component()
        with pytest.raises(ValueError, match="must have a description"):
            component.compile_as_subagent()

    def test_succeeds_with_description(self, subagent_component):
        # Should not raise.
        subagent_component.compile_as_subagent()


class TestCompileAsSubagentGraphStructure:
    """Tests for the graph compile_as_subagent produces."""

    def test_returns_compiled_state_graph(self, subagent_component):
        compiled = subagent_component.compile_as_subagent()
        assert isinstance(compiled, CompiledStateGraph)

    def test_entry_point_is_the_component_entry_hook(self, subagent_component):
        compiled = subagent_component.compile_as_subagent()
        assert subagent_component.__entry_hook__() in compiled.nodes

    def test_graph_has_no_checkpointer(self, subagent_component):
        """No checkpointer of its own -- relies on config forwarding for nested-Pregel inheritance."""
        compiled = subagent_component.compile_as_subagent()
        assert compiled.checkpointer is None

    def test_final_response_node_is_present(self, subagent_component):
        """The standard 3-node ReAct loop (agent/tools/final_response) is still fully wired."""
        compiled = subagent_component.compile_as_subagent()
        assert f"{subagent_component.name}#final_response" in compiled.nodes
        assert f"{subagent_component.name}#tools" in compiled.nodes

    def test_terminal_router_route_returns_end(self):
        """The router compile_as_subagent attaches with always routes to END."""
        router = _TerminalRouter()
        assert router.route({}) == END

    def test_terminal_router_attach_is_a_no_op(self):
        router = _TerminalRouter()
        # Should not raise, and should not require a real StateGraph.
        router.attach(graph=None)

    def test_does_not_mutate_bind_to_supervisor_state(self, subagent_component):
        """compile_as_subagent is a distinct mechanism from bind_to_supervisor -- it must not set that flag."""
        subagent_component.compile_as_subagent()
        assert not subagent_component._is_bound_to_supervisor


class TestCompileAsSubagentSessionIdKey:
    """Tests for the per-invocation subsession ID plumbing."""

    def test_session_id_key_reads_from_subsession_id_context_key(
        self, subagent_component
    ):
        subagent_component.compile_as_subagent()

        resolved = subagent_component._session_id_key.to_iokey({})
        assert resolved.target == "context"
        assert resolved.subkeys == [SUBSESSION_ID_CONTEXT_KEY]

    def test_session_id_key_is_optional(self, subagent_component):
        """Must not KeyError when a caller invokes without setting the subsession ID."""
        subagent_component.compile_as_subagent()

        resolved = subagent_component._session_id_key.to_iokey({})
        state = {
            "status": None,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
            "agent_context_limits": {},
        }
        assert resolved.value_from_state(state) is None

    def test_session_id_key_resolves_value_set_by_caller(self, subagent_component):
        subagent_component.compile_as_subagent()

        resolved = subagent_component._session_id_key.to_iokey({})
        state = {
            "status": None,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {SUBSESSION_ID_CONTEXT_KEY: 7},
            "agent_context_limits": {},
        }
        assert resolved.value_from_state(state) == 7


class TestCompileAsSubagentIndependenceFromBindToSupervisor:
    """compile_as_subagent and bind_to_supervisor are independent, purely-additive mechanisms."""

    def test_bind_to_supervisor_still_works_after_class_is_extended(
        self, make_agent_component
    ):
        """Adding compile_as_subagent must not have broken the existing bind_to_supervisor path."""
        component = make_agent_component(description="Test agent for supervisor")

        component.bind_to_supervisor(
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=Mock()
            ),
            output_key=RuntimeIOKey(alias="final_answer", factory=Mock()),
            goal_key=RuntimeIOKey(alias="goal", factory=Mock()),
            tool_approval_decision_key=RuntimeIOKey(
                alias="tool_approval_decision", factory=Mock()
            ),
            cycle_count_key=RuntimeIOKey(alias="cycle_count", factory=Mock()),
        )

        assert component._is_bound_to_supervisor
