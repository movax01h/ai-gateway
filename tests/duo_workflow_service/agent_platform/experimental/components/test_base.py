from unittest.mock import patch

import pytest
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import IOKey
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture
def flow_type():
    """Fixture for flow type."""
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


class ConcreteComponent(BaseComponent):
    """Concrete implementation of BaseComponent for testing purposes."""

    _allowed_input_targets = ("context", "conversation_history")
    _allowed_output_targets = ("context", "status")

    def attach(self, graph: StateGraph, router: RouterProtocol):
        """Mock implementation of abstract method."""

    def __entry_hook__(self) -> str:
        """Mock implementation of abstract method."""
        return f"{self.name}_entry"


class TestBaseComponentValidateFields:
    """Test BaseComponent field validation methods."""

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_validate_input_fields_with_allowed_targets(
        self, mock_iokey_class, flow_type
    ):
        """Test validation passes when input targets are in allowed targets."""
        # Create real IOKey instances
        input1 = IOKey(target="context")
        input2 = IOKey(target="conversation_history")
        output = IOKey(target="context")

        # Mock IOKey.parse_keys and IOKey.parse_key methods to return real instances
        mock_iokey_class.parse_keys.return_value = [input1, input2]
        mock_iokey_class.parse_key.return_value = output

        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            inputs=["context", "conversation_history"],
            output="context",
        )

        assert len(component.inputs) == 2
        assert component.inputs[0].target == "context"
        assert component.inputs[1].target == "conversation_history"
        assert component.output.target == "context"

        # Verify IOKey methods were called
        mock_iokey_class.parse_keys.assert_called_once_with(
            ["context", "conversation_history"]
        )
        mock_iokey_class.parse_key.assert_called_once_with("context")

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_validate_input_fields_with_disallowed_input_target_raises_error(
        self, mock_iokey_class, flow_type
    ):
        """Test validation fails when input target is not in allowed targets."""
        # Create real IOKey instance with disallowed target
        input_key = IOKey(
            target="status"
        )  # Using valid target but not allowed for this component

        mock_iokey_class.parse_keys.return_value = [input_key]

        with pytest.raises(ValidationError) as exc_info:
            ConcreteComponent(
                name="test_component",
                flow_id="test_workflow",
                flow_type=flow_type,
                inputs=["status"],  # This target is not in _allowed_input_targets
            )

        mock_iokey_class.parse_keys.assert_called_once()
        error_message = str(exc_info.value)
        assert "ConcreteComponent" in error_message
        assert "doesn't support the input target" in error_message
        assert "status" in error_message

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_validate_output_field_with_disallowed_output_target_raises_error(
        self, mock_iokey_class, flow_type
    ):
        """Test validation fails when output target is not in allowed targets."""
        # Create real IOKey instance with disallowed output target
        output_key = IOKey(
            target="conversation_history"
        )  # Valid target but not allowed for output

        mock_iokey_class.parse_key.return_value = output_key

        with pytest.raises(ValidationError) as exc_info:
            ConcreteComponent(
                name="test_component",
                flow_id="test_workflow",
                flow_type=flow_type,
                output="conversation_history",  # This target is not in _allowed_output_targets
            )

        error_message = str(exc_info.value)
        assert "ConcreteComponent" in error_message
        assert "doesn't support the output target" in error_message
        assert "conversation_history" in error_message

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_validate_mixed_valid_and_invalid_input_targets_raises_error(
        self, mock_iokey_class, flow_type
    ):
        """Test validation fails when one of multiple input targets is invalid."""
        # Create real IOKey instances - one valid, one invalid for this component
        input1 = IOKey(target="context")  # Valid
        input2 = IOKey(target="status")  # Valid IOKey but not allowed for input

        mock_iokey_class.parse_keys.return_value = [input1, input2]

        with pytest.raises(ValidationError) as exc_info:
            ConcreteComponent(
                name="test_component",
                flow_id="test_workflow",
                flow_type=flow_type,
                inputs=["context", "status"],
            )

        error_message = str(exc_info.value)
        assert "ConcreteComponent" in error_message
        assert "doesn't support the input target" in error_message
        assert "status" in error_message

    def test_entry_hook_returns_expected_format(self, flow_type):
        """Test that __entry_hook__ returns the expected format."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        entry_name = component.__entry_hook__()
        assert entry_name == "test_component_entry"

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_component_without_inputs_or_output_fields(
        self, mock_iokey_class, flow_type
    ):
        """Test component creation when inputs and output are not provided."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        # IOKey parsing methods should not be called when fields are not provided
        mock_iokey_class.parse_keys.assert_not_called()
        mock_iokey_class.parse_key.assert_not_called()

        assert component.inputs == []
        assert component.output is None
