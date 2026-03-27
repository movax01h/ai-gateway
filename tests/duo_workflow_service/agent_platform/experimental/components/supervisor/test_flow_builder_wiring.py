"""Test suite for Flow builder supervisor wiring integration."""

# pylint: disable=file-naming-for-tests

from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.flows.base import Flow
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
    FlowConfigMetadata,
)


class TestFlowBuilderSupervisorWiring:
    """Tests for Flow builder deferred construction and validation of supervisors."""

    def _make_flow(self, config: FlowConfig) -> Flow:
        """Create a Flow instance with the given config (bypassing __init__)."""
        flow = object.__new__(Flow)
        flow._config = config
        return flow

    def test_resolve_subagents_validates_existence(self):
        """Test that missing managed agent raises ValueError during resolution."""
        config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "prompt_id": "supervisor_prompt",
                    "toolset": ["get_issue"],
                    "managed_agents": ["developer"],
                    "max_delegations": 10,
                },
            ],
            routers=[{"from": "supervisor", "to": "end"}],
            flow=FlowConfigMetadata(entry_point="supervisor"),
        )

        flow = self._make_flow(config)
        components: dict[str, BaseComponent] = {"end": Mock(spec=BaseComponent)}

        with pytest.raises(
            ValueError,
            match="references managed agent 'developer' which is not defined",
        ):
            flow._resolve_subagents(config.components[0], components)

    def test_resolve_subagents_pops_from_components(self):
        """Test that resolved subagents are removed from the components dict."""
        config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "developer",
                    "type": "AgentComponent",
                    "description": "Developer agent",
                    "prompt_id": "dev_prompt",
                    "toolset": ["read_file"],
                },
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "prompt_id": "supervisor_prompt",
                    "toolset": ["get_issue"],
                    "managed_agents": ["developer"],
                    "max_delegations": 10,
                },
            ],
            routers=[{"from": "supervisor", "to": "end"}],
            flow=FlowConfigMetadata(entry_point="supervisor"),
        )

        flow = self._make_flow(config)
        developer_mock = Mock(spec=BaseComponent)
        components: dict[str, BaseComponent] = {
            "developer": developer_mock,
            "end": Mock(spec=BaseComponent),
        }

        subagents = flow._resolve_subagents(config.components[1], components)

        assert "developer" in subagents
        assert subagents["developer"] is developer_mock
        assert "developer" not in components

    def test_resolve_subagents_prevents_sharing_across_supervisors(self):
        """Test that a subagent claimed by one supervisor can't be claimed by another."""
        config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "developer",
                    "type": "AgentComponent",
                    "description": "Developer agent",
                    "prompt_id": "dev_prompt",
                    "toolset": ["read_file"],
                },
                {
                    "name": "supervisor_a",
                    "type": "SupervisorAgentComponent",
                    "prompt_id": "prompt_a",
                    "toolset": ["get_issue"],
                    "managed_agents": ["developer"],
                    "max_delegations": 10,
                },
                {
                    "name": "supervisor_b",
                    "type": "SupervisorAgentComponent",
                    "prompt_id": "prompt_b",
                    "toolset": ["get_issue"],
                    "managed_agents": ["developer"],
                    "max_delegations": 10,
                },
            ],
            routers=[
                {"from": "supervisor_a", "to": "end"},
                {"from": "supervisor_b", "to": "end"},
            ],
            flow=FlowConfigMetadata(entry_point="supervisor_a"),
        )

        flow = self._make_flow(config)
        components: dict[str, BaseComponent] = {
            "developer": Mock(spec=BaseComponent),
            "end": Mock(spec=BaseComponent),
        }

        # First supervisor claims developer
        flow._resolve_subagents(config.components[1], components)

        # Second supervisor can't — developer already popped
        with pytest.raises(
            ValueError,
            match="references managed agent 'developer' which is not defined",
        ):
            flow._resolve_subagents(config.components[2], components)

    def test_has_unresolved_dependencies_true_for_supervisor(self):
        """Test that supervisor with unbuilt subagents is detected as deferred."""
        config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "developer",
                    "type": "AgentComponent",
                    "description": "Developer agent",
                    "prompt_id": "dev_prompt",
                    "toolset": ["read_file"],
                },
                {
                    "name": "supervisor",
                    "type": "SupervisorAgentComponent",
                    "prompt_id": "supervisor_prompt",
                    "toolset": ["get_issue"],
                    "managed_agents": ["developer"],
                    "max_delegations": 10,
                },
            ],
            routers=[{"from": "supervisor", "to": "end"}],
            flow=FlowConfigMetadata(entry_point="supervisor"),
        )

        flow = self._make_flow(config)

        # Before developer is built
        components: dict[str, BaseComponent] = {}
        assert flow._has_unresolved_dependencies(config.components[1], components)

        # After developer is built
        components["developer"] = Mock(spec=BaseComponent)
        assert not flow._has_unresolved_dependencies(config.components[1], components)

    def test_has_unresolved_dependencies_false_for_regular_component(self):
        """Test that regular components are never deferred."""
        config = FlowConfig(
            version="experimental",
            environment="remote",
            components=[
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "prompt_id": "prompt",
                    "toolset": ["read_file"],
                },
            ],
            routers=[{"from": "agent", "to": "end"}],
            flow=FlowConfigMetadata(entry_point="agent"),
        )

        flow = self._make_flow(config)

        assert not flow._has_unresolved_dependencies(config.components[0], {})
