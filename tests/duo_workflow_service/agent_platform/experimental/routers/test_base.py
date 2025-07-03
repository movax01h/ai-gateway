from unittest.mock import patch

import pytest
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.routers.base import BaseRouter
from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey


class ConcreteRouter(BaseRouter):
    """Concrete implementation of BaseRouter for testing purposes."""

    _allowed_input_targets = tuple(["context"])

    def attach(self, graph: StateGraph):
        """Mock implementation of abstract method."""

    def route(self, state: FlowState) -> str:
        """Mock implementation of abstract method."""
        return self.DEFAULT_ROUTE


class TestBaseRouter:

    @patch("duo_workflow_service.agent_platform.experimental.routers.base.IOKey")
    def test_input_filed_parsing(self, mock_iokey_class):
        """Test validation passes when input targets are in allowed targets."""
        # Create real IOKey instances
        input_key = IOKey(target="context")

        # Mock IOKey.parse_keys and IOKey.parse_key methods to return real instances
        mock_iokey_class.parse_key.return_value = input_key

        component = ConcreteRouter(
            input="context",
        )

        assert isinstance(component.input, IOKey)
        assert component.input.target == "context"

        # Verify IOKey methods were called
        mock_iokey_class.parse_key.assert_called_once_with("context")

    @patch("duo_workflow_service.agent_platform.experimental.routers.base.IOKey")
    def test_validate_input_filed_with_allowed_target_and_non_input_field(
        self, mock_iokey_class
    ):
        """Test validation passes when input filed is None."""

        router = ConcreteRouter()

        assert router.input is None
        mock_iokey_class.parse_key.assert_not_called()

    @patch("duo_workflow_service.agent_platform.experimental.routers.base.IOKey")
    def test_validate_input_field_with_disallowed_target_raises_error(
        self, mock_iokey_class
    ):
        """Test validation fails when input target is not in allowed targets."""
        input_key = IOKey(target="conversation_history")
        mock_iokey_class.parse_key.return_value = input_key

        with pytest.raises(ValidationError) as exc_info:
            ConcreteRouter(
                input="conversation_history",
            )

        error_message = str(exc_info.value)
        assert "ConcreteRouter" in error_message
        assert "doesn't support the input target" in error_message
        assert "conversation_history" in error_message
