"""Test suite for the v1 AgentComponent factory function."""

from unittest.mock import Mock

import pytest

from ai_gateway.response_schemas import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.base import BaseComponent
from duo_workflow_service.agent_platform.v1.components.factory import (
    agent_component_factory,
)
from duo_workflow_service.agent_platform.v1.components.registry import (
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.component import (
    SupervisorAgentComponentV2,
)
from lib.feature_flags import current_feature_flag_context
from lib.feature_flags.context import FeatureFlag

# The @inject decorator wraps the class into a function when the class has
# Provide[...] fields; __wrapped__ gives the original class for isinstance() checks.
_AgentComponentClass = AgentComponent.__wrapped__  # type: ignore[attr-defined]
_SupervisorAgentComponentClass = SupervisorAgentComponent.__wrapped__  # type: ignore[attr-defined]
_SupervisorAgentComponentV2Class = SupervisorAgentComponentV2.__wrapped__  # type: ignore[attr-defined]


@pytest.fixture(name="mock_schema_registry")
def mock_schema_registry_fixture():
    """Fixture for mock schema registry."""
    return Mock(spec=BaseResponseSchemaRegistry)


class TestAgentComponentFactoryRegistry:
    """Test suite verifying factory registration in the v1 ComponentRegistry."""

    def test_factory_registered_as_agent_component(self):
        """The factory is registered under 'AgentComponent' in the v1 ComponentRegistry."""
        registry = ComponentRegistry.instance()
        # pylint: disable-next=unsupported-membership-test
        assert "AgentComponent" in registry

    def test_supervisor_component_not_registered_under_own_name(self):
        """SupervisorAgentComponent is not registered directly; use AgentComponent with subagents."""
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
        consumed subagents is handled by FlowGraphBuilder._instantiate_component after the component is created.
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
            subagents=[{"name": "developer"}],
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

    def test_factory_creates_agent_component_without_subagents(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory returns AgentComponent when subagents is absent."""
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

    def test_factory_creates_agent_component_with_empty_subagents(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory returns AgentComponent when subagents is an empty list (falsy)."""
        component = agent_component_factory(
            name="my_agent",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="test_prompt",
            toolset=mock_toolset,
            subagents=[],
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(component, _AgentComponentClass)
        assert not isinstance(component, _SupervisorAgentComponentClass)

    def test_factory_creates_supervisor_component_with_subagents(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory returns SupervisorAgentComponent when subagents is non-empty.

        The factory passes the shared ``_built_components`` dict as
        ``subagent_components`` so that
        :meth:`SupervisorAgentComponent.validate_and_consume_subagents`
        can select and validate the named subagents.  The dict is passed
        read-only — removing consumed subagents is the caller's responsibility.
        """
        developer_mock = Mock(spec=BaseComponent)
        developer_mock.description = "Developer agent"
        developer_mock.bind_to_supervisor = Mock()

        built_components: dict[str, BaseComponent] = {"developer": developer_mock}

        result = agent_component_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            subagents=[{"name": "developer"}],
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

    def test_factory_does_not_mutate_built_components(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """Factory does not remove consumed subagents from _built_components.

        Removal of consumed subagents is the responsibility of the flow builder
        (``FlowGraphBuilder._instantiate_component``), not the factory.
        """
        developer_mock = Mock(spec=BaseComponent)
        developer_mock.description = "Developer agent"
        developer_mock.bind_to_supervisor = Mock()

        built_components: dict[str, BaseComponent] = {"developer": developer_mock}
        original_keys = set(built_components.keys())

        agent_component_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            subagents=[{"name": "developer"}],
            max_delegations=5,
            _built_components=built_components,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert set(built_components.keys()) == original_keys


@pytest.fixture(name="parallel_subagents_flag")
def parallel_subagents_flag_fixture():
    """Enable ``dap_parallel_subagents`` for one test, then restore the previous context."""
    token = current_feature_flag_context.set({FeatureFlag.DAP_PARALLEL_SUBAGENTS.value})
    yield
    current_feature_flag_context.reset(token)


class TestAgentComponentFactoryV2Dispatch:
    """Test suite verifying factory dispatch to SupervisorAgentComponentV2.

    The ``dap_parallel_subagents`` feature flag is the only thing selecting
    parallel over sequential delegation: any component declaring ``subagents``
    switches once the flag is on, and the flag doubles as the kill switch, so a
    rollback never needs a new flow config revision.
    """

    def _make_developer_mock(self):
        developer_mock = Mock(spec=BaseComponent)
        developer_mock.description = "Developer agent"
        developer_mock.compile_as_subagent = Mock()
        return developer_mock

    def test_dispatches_to_v2_when_the_feature_flag_is_enabled(
        self,
        parallel_subagents_flag,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        built_components: dict[str, BaseComponent] = {
            "developer": self._make_developer_mock()
        }

        result = agent_component_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            subagents=[{"name": "developer"}],
            max_delegations=5,
            _built_components=built_components,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(result, _SupervisorAgentComponentV2Class)

    def test_dispatches_to_v1_when_the_feature_flag_is_disabled(
        self,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        """The feature flag is the kill switch: without it, subagents are delegated to sequentially.

        No ``parallel_subagents_flag`` fixture here, so the flag context is empty. The mock exposes
        ``bind_to_supervisor`` rather than ``compile_as_subagent``, since falling back really does construct the
        sequential component, which validates its managed agents for exactly that method.
        """
        developer_mock = Mock(spec=BaseComponent)
        developer_mock.description = "Developer agent"
        developer_mock.bind_to_supervisor = Mock()
        built_components: dict[str, BaseComponent] = {"developer": developer_mock}

        result = agent_component_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            subagents=[{"name": "developer"}],
            max_delegations=5,
            _built_components=built_components,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert isinstance(result, _SupervisorAgentComponentClass)
        assert not isinstance(result, _SupervisorAgentComponentV2Class)

    def test_v2_dispatch_injects_built_components(
        self,
        parallel_subagents_flag,
        flow_id,
        flow_type,
        mock_toolset,
        mock_prompt_registry,
        mock_internal_event_client,
        mock_schema_registry,
        user,
    ):
        built_components: dict[str, BaseComponent] = {
            "developer": self._make_developer_mock()
        }

        result = agent_component_factory(
            name="supervisor",
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            prompt_id="supervisor_prompt",
            toolset=mock_toolset,
            subagents=[{"name": "developer"}],
            max_delegations=5,
            _built_components=built_components,
            prompt_registry=mock_prompt_registry,
            internal_event_client=mock_internal_event_client,
            schema_registry=mock_schema_registry,
        )

        assert "developer" in result.subagent_components
