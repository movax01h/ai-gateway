"""Test suite for AgentComponent factory."""

from unittest.mock import Mock

import pytest

from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.components.factory import (
    agent_component_factory,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.component import (
    SupervisorAgentComponent,
)

# The @inject decorator wraps the class into a function when the class has
# Provide[...] fields; __wrapped__ gives the original class for isinstance() checks.
_AgentComponentClass = AgentComponent.__wrapped__  # type: ignore[attr-defined]
_SupervisorAgentComponentClass = SupervisorAgentComponent.__wrapped__  # type: ignore[attr-defined]


@pytest.fixture(name="mock_schema_registry")
def mock_schema_registry_fixture():
    """Fixture for mock schema registry."""
    return Mock(spec=BaseResponseSchemaRegistry)


class TestAgentComponentFactoryRegistry:
    """Test suite verifying factory registration in the ComponentRegistry."""

    def test_factory_registered_as_agent_component(self):
        """The factory is registered under 'AgentComponent' in the ComponentRegistry."""
        registry = ComponentRegistry.instance()
        # pylint: disable-next=unsupported-membership-test
        assert "AgentComponent" in registry

    def test_supervisor_component_not_registered_under_own_name(self):
        """SupervisorAgentComponent is not registered directly; use AgentComponent with managed_agents."""
        registry = ComponentRegistry.instance()
        # pylint: disable-next=unsupported-membership-test
        assert "SupervisorAgentComponent" not in registry

    def test_flow_injects_built_components_into_factory(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Flow passes _built_components to the registered AgentComponent factory.

        The factory reads _built_components to resolve subagent references but does not mutate the dict — removal of
        consumed subagents is handled by Flow._instantiate_component after the component is created.
        """
        registry = ComponentRegistry.instance()
        # pylint: disable-next=unsubscriptable-object
        registered_factory = registry["AgentComponent"]

        developer_mock = Mock(spec=BaseComponent)
        developer_mock.description = "Developer agent"
        developer_mock.bind_to_supervisor = Mock()

        built_components: dict[str, BaseComponent] = {"developer": developer_mock}

        result = registered_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            managed_agents=["developer"],
            max_delegations=5,
            _built_components=built_components,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(result, _SupervisorAgentComponentClass)
        # The factory must NOT pop from the shared dict — Flow owns that cleanup.
        assert "developer" in built_components
        # The created component must have the resolved subagent injected.
        assert "developer" in result.subagent_components


class TestAgentComponentFactoryDispatch:
    """Test suite verifying factory dispatch logic."""

    def test_factory_creates_agent_component_without_managed_agents(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory returns AgentComponent when managed_agents is absent."""
        component = agent_component_factory(
            name="my_agent",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="test_prompt",
            toolset=mock_toolset,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(component, _AgentComponentClass)
        assert not isinstance(component, _SupervisorAgentComponentClass)

    def test_factory_creates_agent_component_with_empty_managed_agents(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory returns AgentComponent when managed_agents is an empty list (falsy)."""
        component = agent_component_factory(
            name="my_agent",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="test_prompt",
            toolset=mock_toolset,
            managed_agents=[],
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(component, _AgentComponentClass)
        assert not isinstance(component, _SupervisorAgentComponentClass)
