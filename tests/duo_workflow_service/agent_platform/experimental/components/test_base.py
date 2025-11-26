from unittest.mock import Mock, patch

import pytest
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    EndComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="flow_type")
def flow_type_fixture():
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
        self, mock_iokey_class, flow_type, user
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
            user=user,
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
        self, mock_iokey_class, flow_type, user
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
                user=user,
                inputs=["status"],  # This target is not in _allowed_input_targets
            )

        mock_iokey_class.parse_keys.assert_called_once()
        error_message = str(exc_info.value)
        assert "ConcreteComponent" in error_message
        assert "doesn't support the input target" in error_message
        assert "status" in error_message

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_validate_mixed_valid_and_invalid_input_targets_raises_error(
        self, mock_iokey_class, flow_type, user
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
                user=user,
                inputs=["context", "status"],
            )

        error_message = str(exc_info.value)
        assert "ConcreteComponent" in error_message
        assert "doesn't support the input target" in error_message
        assert "status" in error_message

    def test_entry_hook_returns_expected_format(self, flow_type, user):
        """Test that __entry_hook__ returns the expected format."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        entry_name = component.__entry_hook__()
        assert entry_name == "test_component_entry"

    @patch("duo_workflow_service.agent_platform.experimental.components.base.IOKey")
    def test_component_without_inputs_fields(self, mock_iokey_class, flow_type, user):
        """Test component creation when inputs are not provided."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        # IOKey parsing methods should not be called when fields are not provided
        mock_iokey_class.parse_keys.assert_not_called()

        assert component.inputs == []


class TestBaseComponentOutputs:
    """Test BaseComponent outputs property and related functionality."""

    def test_outputs_property_with_template_replacement(self, flow_type, user):
        """Test that outputs property correctly replaces template placeholders."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
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

    def test_outputs_property_with_different_component_name(self, flow_type, user):
        """Test outputs property with different component name."""
        component = ConcreteComponent(
            name="my_custom_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        outputs = component.outputs

        assert len(outputs) == 2
        # Component name should be replaced in template
        assert outputs[0].subkeys == ["my_custom_component", "result"]

    def test_outputs_property_with_component_without_outputs(self, flow_type, user):
        """Test outputs property when component has no outputs defined."""
        component = ComponentWithoutOutputs(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        outputs = component.outputs

        assert len(outputs) == 0

    def test_outputs_property_immutability(self, flow_type, user):
        """Test that outputs property returns a new tuple each time."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        outputs1 = component.outputs
        outputs2 = component.outputs

        # Should be equal but not the same object
        assert outputs1 == outputs2
        assert outputs1 is not outputs2


class TestBaseComponentSupportedEnvironments:
    """Test BaseComponent supported_environments functionality."""

    def test_supported_environments_inheritance(self, flow_type, user):
        """Test that supported_environments is properly inherited."""
        component = ConcreteComponent(
            name="test_component",
            flow_id="test_workflow",
            flow_type=flow_type,
            user=user,
        )

        # Should inherit from class variable
        assert component.supported_environments == ("platform", "local")


class TestEndComponent:
    """Test EndComponent functionality."""

    @pytest.fixture(name="end_component")
    def end_component_fixture(self, flow_type, user):
        """Fixture providing an EndComponent instance."""
        return EndComponent(
            name="end",
            flow_id="test-workflow",
            flow_type=flow_type,
            user=user,
        )

    def test_entry_hook_returns_terminate_flow(self, end_component):
        """Test that __entry_hook__ returns 'terminate_flow'."""
        assert end_component.__entry_hook__() == "terminate_flow"

    def test_attach_adds_node_and_edge(self, end_component):
        """Test that attach method adds node and edge to graph."""
        mock_graph = Mock(spec=StateGraph)

        end_component.attach(mock_graph)

        # Verify node was added with correct name and some callable function
        mock_graph.add_node.assert_called_once()
        call_args = mock_graph.add_node.call_args
        assert call_args[0][0] == "terminate_flow"  # Node name
        assert callable(call_args[0][1])  # Function is callable

        # Verify edge to END was added
        mock_graph.add_edge.assert_called_once_with("terminate_flow", END)

    def test_attach_with_router_parameter(self, end_component):
        """Test that attach method works with optional router parameter."""
        mock_graph = Mock(spec=StateGraph)
        mock_router = Mock()

        # Should work with router parameter (even though it's not used)
        end_component.attach(mock_graph, mock_router)

        # Verify the graph methods were called
        mock_graph.add_node.assert_called_once()
        mock_graph.add_edge.assert_called_once_with("terminate_flow", END)

    @pytest.mark.asyncio
    async def test_end_component_sets_completed_status(self, end_component):
        """Test that EndComponent sets status to COMPLETED when executed in a real graph."""

        # Create a real StateGraph with FlowState
        graph = StateGraph(FlowState)

        # Attach the end component
        end_component.attach(graph)

        # Set entry point and compile
        graph.set_entry_point("terminate_flow")
        compiled_graph = graph.compile()

        # Create initial state
        initial_state = FlowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            ui_chat_log=[],
            context={"test": "data"},
        )

        # Run the graph
        result = await compiled_graph.ainvoke(initial_state)

        # Verify the status was set to COMPLETED
        assert result["status"] == WorkflowStatusEnum.COMPLETED.value
