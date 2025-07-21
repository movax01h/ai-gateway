from unittest.mock import patch

import pytest
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import IOKey
from duo_workflow_service.agent_platform.experimental.state.base import IOKeyTemplate
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture
def flow_type():
    """Fixture for flow type."""
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


class ConcreteComponent(BaseComponent):
    """Concrete implementation of BaseComponent for testing purposes."""

    _allowed_input_targets = ("context", "conversation_history")
    _outputs = (
        IOKeyTemplate(target="context", subkeys=["<name>", "result"]),
        IOKeyTemplate(target="status"),
    )
    supported_environments = ("platform", "local")

    def attach(self, graph: StateGraph, router: RouterProtocol):
        """Mock implementation of abstract method."""

    def __entry_hook__(self) -> str:
        """Mock implementation of abstract method."""
        return f"{self.name}_entry"


class ComponentWithoutOutputs(BaseComponent):
    """Component without any outputs for testing."""

    _allowed_input_targets = ("context",)
    _outputs = ()
    supported_environments = ("platform",)

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

        # Mock IOKey.parse_keys method to return real instances
        mock_iokey_class.parse_keys.return_value = [input1, input2]

        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            inputs=["context", "conversation_history"],
        )

        assert len(component.inputs) == 2
        assert component.inputs[0].target == "context"
        assert component.inputs[1].target == "conversation_history"

        # Verify IOKey methods were called
        mock_iokey_class.parse_keys.assert_called_once_with(
            ["context", "conversation_history"]
        )

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
    def test_component_without_inputs_fields(self, mock_iokey_class, flow_type):
        """Test component creation when inputs are not provided."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        # IOKey parsing methods should not be called when fields are not provided
        mock_iokey_class.parse_keys.assert_not_called()

        assert component.inputs == []


class TestBaseComponentOutputs:
    """Test BaseComponent outputs property and related functionality."""

    def test_outputs_property_with_template_replacement(self, flow_type):
        """Test that outputs property correctly replaces template placeholders."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        outputs = component.outputs

        assert len(outputs) == 2
        assert all(isinstance(output, IOKey) for output in outputs)

        # First output should have component name replaced in subkeys
        assert outputs[0].target == "context"
        assert outputs[0].subkeys == ["test_component", "result"]

        # Second output should be simple status
        assert outputs[1].target == "status"
        assert outputs[1].subkeys is None

    def test_outputs_property_with_different_component_name(self, flow_type):
        """Test outputs property with different component name."""
        component = ConcreteComponent(
            name="my_custom_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        outputs = component.outputs

        assert len(outputs) == 2
        # Component name should be replaced in template
        assert outputs[0].subkeys == ["my_custom_component", "result"]

    def test_outputs_property_with_component_without_outputs(self, flow_type):
        """Test outputs property when component has no outputs defined."""
        component = ComponentWithoutOutputs(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        outputs = component.outputs

        assert len(outputs) == 0

    def test_outputs_property_immutability(self, flow_type):
        """Test that outputs property returns a new tuple each time."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        outputs1 = component.outputs
        outputs2 = component.outputs

        # Should be equal but not the same object
        assert outputs1 == outputs2
        assert outputs1 is not outputs2


class TestBaseComponentSupportedEnvironments:
    """Test BaseComponent supported_environments functionality."""

    def test_supported_environments_inheritance(self, flow_type):
        """Test that supported_environments is properly inherited."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
        )

        # Should inherit from class variable
        assert component.supported_environments == ("platform", "local")
