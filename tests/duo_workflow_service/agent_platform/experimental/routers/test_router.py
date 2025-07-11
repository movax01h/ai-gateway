from unittest.mock import MagicMock, Mock, patch

import pytest
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.routers.router import (
    BaseRouter,
    Router,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey


class TestRouter:
    """Test cases for the Router class."""

    def create_mock_component(self, name: str = "test_component"):
        """Helper method to create a mock BaseComponent."""
        mock_component = MagicMock(spec=BaseComponent)
        mock_component.__entry_hook__ = MagicMock(return_value=f"{name}_entry_hook")
        mock_component.attach = Mock()
        mock_component._allowed_input_targets = ("context",)
        mock_component._allowed_output_targets = ("context",)
        mock_component.inputs = IOKey.parse_keys(["context:key"])
        mock_component.output = IOKey.parse_key("context:key")

        return mock_component

    def test_router_initialization_with_single_component(self):
        """Test Router initialization with a single to_component."""
        from_component = self.create_mock_component("from")
        to_component = self.create_mock_component("to")

        router = Router(from_component=from_component, to_component=to_component)

        assert router.from_component == from_component
        assert router.to_component == to_component
        assert router.input is None

    def test_router_initialization_with_dict_components(self):
        """Test Router initialization with dict of to_components."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        to_component_2 = self.create_mock_component("to2")
        to_components = {"route1": to_component_1, "route2": to_component_2}

        router = Router(
            from_component=from_component,
            to_component=to_components,
            input="context:key",
        )

        assert router.from_component == from_component
        assert router.to_component == to_components
        assert router.input.target == "context"
        assert router.input.subkeys == ["key"]

    def test_router_validation_input_none_with_dict_components_raises_error(self):
        """Test that Router raises error when input is None but to_component is dict."""
        from_component = self.create_mock_component("from")
        to_components = {"route1": self.create_mock_component("to1")}

        with pytest.raises(ValidationError) as exc_info:
            Router(from_component=from_component, to_component=to_components)

        error_message = str(exc_info.value)
        assert (
            "If input is None, then to_component must be a BaseComponent"
            in error_message
        )

    def test_router_validation_input_not_none_with_single_component_raises_error(self):
        """Test that Router raises error when input is not None but to_component is single component."""
        from_component = self.create_mock_component("from")
        to_component = self.create_mock_component("to")

        with pytest.raises(ValidationError) as exc_info:
            Router(
                from_component=from_component,
                to_component=to_component,
                input="context:key",
            )

        error_message = str(exc_info.value)
        assert "If input is not None, then to_component must be a dict" in error_message

    def test_router_validation_disallowed_input_target_raises_error(self):
        """Test that Router raises error for disallowed input targets."""
        from_component = self.create_mock_component("from")
        to_components = {"route1": self.create_mock_component("to1")}

        with pytest.raises(ValidationError) as exc_info:
            Router(
                from_component=from_component,
                to_component=to_components,
                input="conversation_history:key",
            )

        error_message = str(exc_info.value)
        assert "doesn't support the input target" in error_message
        assert "conversation_history" in error_message

    def test_router_attach_calls_from_component_attach(self):
        """Test that Router.attach calls from_component.attach with correct parameters."""
        from_component = self.create_mock_component("from")
        to_component = self.create_mock_component("to")
        graph = Mock(spec=StateGraph)

        router = Router(from_component=from_component, to_component=to_component)

        router.attach(graph)

        from_component.attach.assert_called_once_with(graph, router)

    def test_router_route_with_no_input_returns_single_component_entry_hook(self):
        """Test Router.route returns single component entry hook when input is None."""
        from_component = self.create_mock_component("from")
        to_component = self.create_mock_component("to")
        state = Mock(spec=FlowState)

        router = Router(from_component=from_component, to_component=to_component)

        result = router.route(state)

        assert result == "to_entry_hook"
        to_component.__entry_hook__.assert_called_once()

    @patch(
        "duo_workflow_service.agent_platform.experimental.routers.base.IOKey.parse_key",
        return_value=Mock(spec=IOKey, target="context", subkeys=["key"]),
    )
    def test_router_route_with_input_returns_matching_component_entry_hook(
        self, mock_iokey
    ):
        """Test Router.route returns matching component entry hook when input matches."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        to_component_2 = self.create_mock_component("to2")
        state = {}
        mock_iokey.return_value.value_from_state.return_value = "route1"

        router = Router(
            input="context:key",
            from_component=from_component,
            to_component={"route1": to_component_1, "route2": to_component_2},
        )

        result = router.route(state)

        assert result == "to1_entry_hook"
        to_component_1.__entry_hook__.assert_called_once()
        mock_iokey.return_value.value_from_state.assert_called_once_with(state)

    @patch(
        "duo_workflow_service.agent_platform.experimental.routers.base.IOKey.parse_key",
        return_value=Mock(spec=IOKey, target="context", subkeys=["key"]),
    )
    def test_router_route_with_input_returns_default_route_when_no_match(
        self, mock_iokey
    ):
        """Test Router.route returns default route when no matching route found."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        default_component = self.create_mock_component("default")
        mock_iokey.return_value.value_from_state.return_value = "non_existing_route"
        state = {}

        router = Router(
            input="context:key",
            from_component=from_component,
            to_component={
                "route1": to_component_1,
                BaseRouter.DEFAULT_ROUTE: default_component,
            },
        )

        result = router.route(state)

        assert result == "default_entry_hook"
        default_component.__entry_hook__.assert_called_once()
        mock_iokey.return_value.value_from_state.assert_called_once_with(state)

    @patch(
        "duo_workflow_service.agent_platform.experimental.routers.base.IOKey.parse_key",
        return_value=Mock(spec=IOKey, target="context", subkeys=["key"]),
    )
    def test_router_route_with_input_raises_keyerror_when_no_match_and_no_default(
        self, mock_iokey
    ):
        """Test Router.route raises KeyError when no matching route and no default route."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        mock_iokey.return_value.value_from_state.return_value = "non_existing_route"
        state = {}

        router = Router(
            input="context:key",
            from_component=from_component,
            to_component={"route1": to_component_1},
        )

        with pytest.raises(KeyError) as exc_info:
            router.route(state)

        error_message = str(exc_info.value)
        assert "Route key" in error_message
        assert "not found in conditions" in error_message
        mock_iokey.return_value.value_from_state.assert_called_once_with(state)

    @patch(
        "duo_workflow_service.agent_platform.experimental.routers.base.IOKey.parse_key",
        return_value=Mock(spec=IOKey, target="context", subkeys=["key"]),
    )
    def test_router_route_with_empty_variables_returns_default_route(self, mock_iokey):
        """Test Router.route returns default route when value_from_state returns empty dict."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        default_component = self.create_mock_component("default")
        state = {}
        mock_iokey.return_value.value_from_state.return_value = None

        router = Router(
            input="context:key",
            from_component=from_component,
            to_component={
                "route1": to_component_1,
                BaseRouter.DEFAULT_ROUTE: default_component,
            },
        )

        result = router.route(state)

        assert result == "default_entry_hook"
        default_component.__entry_hook__.assert_called_once()
        mock_iokey.return_value.value_from_state.assert_called_once_with(state)

    @patch(
        "duo_workflow_service.agent_platform.experimental.routers.base.IOKey.parse_key",
        return_value=Mock(spec=IOKey, target="context", subkeys=["key"]),
    )
    def test_router_route_with_integer_route_key(self, mock_iokey):
        """Test Router.route works with integer route keys."""
        from_component = self.create_mock_component("from")
        to_component_1 = self.create_mock_component("to1")
        to_component_2 = self.create_mock_component("to2")
        state = {}

        mock_iokey.return_value.value_from_state.return_value = 1

        router = Router(
            input="context:key",
            from_component=from_component,
            to_component={"1": to_component_1, "2": to_component_2},
        )

        result = router.route(state)

        assert result == "to1_entry_hook"
        to_component_1.__entry_hook__.assert_called_once()
        mock_iokey.return_value.value_from_state.assert_called_once_with(state)

    def test_router_allowed_input_targets(self):
        """Test that Router has correct allowed input targets."""
        assert Router._allowed_input_targets == ("context", "status")
