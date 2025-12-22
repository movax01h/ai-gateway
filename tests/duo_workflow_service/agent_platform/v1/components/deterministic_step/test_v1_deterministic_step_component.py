"""Test suite for DeterministicStepComponent class."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools import BaseTool
from pydantic import BaseModel, ValidationError

from duo_workflow_service.agent_platform.v1.components.deterministic_step.component import (
    DeterministicStepComponent,
)
from duo_workflow_service.agent_platform.v1.components.deterministic_step.ui_log import (
    UILogWriterDeterministicStep,
)
from duo_workflow_service.agent_platform.v1.state import IOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    """Fixture for mock tool."""
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool._arun = AsyncMock(return_value="tool_result")
    tool.args_schema = None  # Default to no schema
    return tool


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    """Fixture for mock toolset."""
    toolset = Mock(spec=Toolset)
    toolset.__getitem__ = Mock(return_value=mock_tool)
    toolset.__contains__ = Mock(return_value=True)
    toolset.keys = Mock(
        return_value=["test_tool", "example_tool"]
    )  # Added keys() method
    return toolset


@pytest.fixture(name="component_name")
def component_name_fixture():
    """Fixture for component name."""
    return "test_component"


@pytest.fixture(name="flow_id")
def flow_id_fixture():
    """Fixture for flow ID."""
    return "test_flow_123"


@pytest.fixture(name="flow_type")
def flow_type_fixture():
    """Fixture for flow type."""
    return GLReportingEventContext.from_workflow_definition("software_development")


@pytest.fixture(name="inputs")
def inputs_fixture():
    """Fixture for component inputs."""
    return [
        IOKey(target="context", subkeys=["user_input"]),
        IOKey(target="context", subkeys=["task_description"]),
    ]


@pytest.fixture(name="deterministic_component")
def deterministic_component_fixture(
    component_name, flow_id, flow_type, user, mock_toolset
):
    """Fixture for DeterministicStepComponent instance."""
    return DeterministicStepComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=["context:user_input", "context:task_description"],
        tool_name="test_tool",
        toolset=mock_toolset,
    )


@pytest.fixture(name="mock_state_graph")
def mock_state_graph_fixture():
    """Fixture for mock StateGraph."""
    return Mock()


@pytest.fixture(name="mock_router")
def mock_router_fixture():
    """Fixture for mock router."""
    return Mock()


@pytest.fixture(name="base_flow_state")
def base_flow_state_fixture():
    """Fixture for base flow state."""
    return {
        "status": "in_progress",
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {"input_param": "test_value"},
    }


@pytest.fixture(name="tool_name")
def tool_name_fixture():
    """Fixture for tool name."""
    return "example_tool"


@pytest.fixture(name="ui_log_events")
def ui_log_events_fixture():
    """Fixture for UI log events."""
    return []


@pytest.fixture(name="ui_role_as")
def ui_role_as_fixture():
    """Fixture for UI role."""
    return "tool"


@pytest.fixture(name="mock_internal_event_client")
def mock_internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="deterministic_step_component")
def deterministic_step_component_fixture(
    component_name,
    flow_id,
    flow_type,
    user,
    tool_name,
    ui_log_events,
    ui_role_as,
    mock_toolset,
    mock_internal_event_client,
):
    """Fixture for DeterministicStepComponent instance."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = tool_name
    mock_tool.args_schema = None
    mock_toolset.__getitem__ = Mock(return_value=mock_tool)
    mock_toolset.__contains__ = Mock(return_value=True)

    return DeterministicStepComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        user=user,
        inputs=["context:user_input", "context:task_description"],
        tool_name=tool_name,
        toolset=mock_toolset,
        internal_event_client=mock_internal_event_client,
        ui_log_events=ui_log_events,
        ui_role_as=ui_role_as,
    )


@pytest.fixture(name="mock_deterministic_step_node_cls")
def mock_deterministic_step_node_cls_fixture(component_name):
    """Fixture for mocked DeterministicStepNode class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.deterministic_step.component.DeterministicStepNode"
    ) as mock_cls:
        mock_node = Mock()
        mock_node.name = f"{component_name}#deterministic_step"
        mock_cls.return_value = mock_node
        yield mock_cls


@pytest.fixture(name="toolset_with_schema_tool")
def toolset_with_schema_tool_fixture():
    """Fixture for toolset containing a tool with args_schema."""

    class MockSchema(BaseModel):
        required_param: str
        optional_param: str = "default"

    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "schema_tool"
    mock_tool.args_schema = MockSchema

    toolset = Mock(spec=Toolset)
    toolset.__getitem__ = Mock(return_value=mock_tool)
    toolset.__contains__ = Mock(return_value=True)
    toolset.keys = Mock(return_value=["schema_tool"])

    return toolset


@pytest.fixture(name="toolset_with_no_args_tool")
def toolset_with_no_args_tool_fixture():
    """Fixture for toolset containing a tool that takes no arguments."""

    class NoArgsSchema(BaseModel):
        pass  # No fields means no arguments

    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "no_args_tool"
    mock_tool.args_schema = NoArgsSchema

    toolset = Mock(spec=Toolset)
    toolset.__getitem__ = Mock(return_value=mock_tool)
    toolset.__contains__ = Mock(return_value=True)
    toolset.keys = Mock(return_value=["no_args_tool"])

    return toolset


class TestDeterministicStepComponentInitialization:
    """Test suite for DeterministicStepComponent initialization."""

    @pytest.mark.parametrize(
        "input_output",
        [
            "context:user_input",
            "conversation_history:agent_component",
        ],
    )
    def test_allowed_targets_through_validation(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        input_output,
    ):
        """Test that component validates input targets correctly."""
        # This should succeed without raising an exception
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[input_output],
            toolset=mock_toolset,
            tool_name="test_tool",
        )
        assert component.validated_tool is not None

    @pytest.mark.parametrize(
        "input_output",
        [
            "status",
            "ui_chat_log",
        ],
    )
    def test_not_allowed_targets_through_validation(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        input_output,
    ):
        """Test that component validates input targets correctly."""
        # This should succeed without raising an exception
        with pytest.raises(ValidationError, match="doesn't support the input target"):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=[input_output],
                toolset=mock_toolset,
                tool_name="test_tool",
            )


class TestDeterministicStepComponentToolValidation:
    """Test suite for DeterministicStepComponent tool validation."""

    def test_tool_not_found_in_toolset(self, component_name, flow_id, flow_type, user):
        """Test that component raises error when tool is not found in toolset."""
        mock_toolset = Mock(spec=Toolset)
        mock_toolset.__contains__ = Mock(return_value=False)
        mock_toolset.keys = Mock(return_value=["available_tool_1", "available_tool_2"])

        with pytest.raises(
            KeyError, match="Tool 'nonexistent_tool' not found in toolset"
        ):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                tool_name="nonexistent_tool",
                toolset=mock_toolset,
            )

    def test_tool_validation_with_schema_success(
        self, component_name, flow_id, flow_type, user, toolset_with_schema_tool
    ):
        """Test successful tool validation when tool has schema."""
        # Create component with matching inputs
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:required_param", "context:optional_param"],
            tool_name="schema_tool",
            toolset=toolset_with_schema_tool,
        )

        assert (
            component.validated_tool
            == toolset_with_schema_tool.__getitem__.return_value
        )

    def test_tool_validation_missing_required_params(
        self, component_name, flow_id, flow_type, toolset_with_schema_tool
    ):
        """Test tool validation fails when required parameters are missing."""
        # Create component missing the required_param
        with pytest.raises(
            ValueError, match="Missing required parameters: \\['required_param'\\]"
        ):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                inputs=["context:optional_param"],
                tool_name="schema_tool",
                toolset=toolset_with_schema_tool,
            )

    def test_tool_validation_unknown_params(
        self, component_name, flow_id, flow_type, user, toolset_with_schema_tool
    ):
        """Test tool validation fails when unknown parameters are provided."""
        # Create component with unknown parameter
        with pytest.raises(
            ValueError, match="Unknown parameters: \\['unknown_param'\\]"
        ):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:required_param", "context:unknown_param"],
                tool_name="schema_tool",
                toolset=toolset_with_schema_tool,
            )

    def test_tool_validation_no_schema(
        self, component_name, flow_id, flow_type, user, mock_toolset
    ):
        """Test tool validation passes when tool has no schema."""
        # Tool with no schema should pass validation regardless of inputs
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:any_param", "context:another_param"],
            tool_name="test_tool",
            toolset=mock_toolset,
        )

        assert component.validated_tool is not None

    def test_missing_tool_name(
        self, component_name, flow_id, flow_type, user, mock_toolset
    ):
        """Test that validation fails when tool_name is missing."""
        with pytest.raises(ValidationError, match="tool_name is required"):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                toolset=mock_toolset,
                # tool_name is missing
            )

    def test_missing_toolset(self, component_name, flow_id, flow_type, user):
        """Test that validation fails when toolset is missing."""
        with pytest.raises(ValidationError, match="toolset is required"):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                tool_name="test_tool",
                # toolset is missing
            )


class TestValidateToolArguments:
    def test_no_args_tool_with_inputs_provided(
        self, component_name, flow_id, flow_type, user, toolset_with_no_args_tool
    ):
        """Test that providing inputs to a tool that takes no arguments fails."""
        with pytest.raises(ValueError, match="Unknown parameters: \\['some_param'\\]"):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:some_param"],
                tool_name="no_args_tool",
                toolset=toolset_with_no_args_tool,
            )

    def test_no_args_tool_with_no_inputs(
        self, component_name, flow_id, flow_type, user, toolset_with_no_args_tool
    ):
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[],  # No inputs provided
            tool_name="no_args_tool",
            toolset=toolset_with_no_args_tool,
        )
        assert component.validated_tool is not None

    def test_validate_tool_arguments_with_alias(
        self, component_name, flow_id, flow_type, user
    ):
        class MockSchema(BaseModel):
            actual_param_name: str

        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "alias_tool"
        mock_tool.args_schema = MockSchema

        toolset = Mock(spec=Toolset)
        toolset.__getitem__ = Mock(return_value=mock_tool)
        toolset.__contains__ = Mock(return_value=True)
        toolset.keys = Mock(return_value=["alias_tool"])

        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=[{"from": "context:some_key", "as": "actual_param_name"}],
            tool_name="alias_tool",
            toolset=toolset,
        )
        assert component.validated_tool is not None

    def test_validate_tool_with_multiple_missing_required(
        self, component_name, flow_id, flow_type, user
    ):
        """Test error message when multiple required parameters are missing."""

        class MultiRequiredSchema(BaseModel):
            param1: str
            param2: int
            param3: bool
            optional: str = "default"

        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "multi_required_tool"
        mock_tool.args_schema = MultiRequiredSchema

        toolset = Mock(spec=Toolset)
        toolset.__getitem__ = Mock(return_value=mock_tool)
        toolset.__contains__ = Mock(return_value=True)
        toolset.keys = Mock(return_value=["multi_required_tool"])

        with pytest.raises(
            ValueError, match="Missing required parameters: \\['param2', 'param3'\\]"
        ):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:param1"],  # Only providing one of three required
                tool_name="multi_required_tool",
                toolset=toolset,
            )


class TestDeterministicStepComponentEntryHook:
    """Test suite for DeterministicStepComponent entry hook."""

    def test_entry_hook_returns_correct_node_name(
        self, deterministic_component, component_name
    ):
        """Test that __entry_hook__ returns the correct node name."""
        expected_entry_node = f"{component_name}#deterministic_step"
        assert deterministic_component.__entry_hook__() == expected_entry_node


class TestDeterministicStepComponentAttachNodes:
    """Test suite for DeterministicStepComponent attach method."""

    def test_attach_creates_node_with_correct_parameters(
        self,
        mock_deterministic_step_node_cls,
        deterministic_step_component,
        mock_state_graph,
        mock_router,
        component_name,
        flow_id,
        flow_type,
        inputs,
        tool_name,
        ui_log_events,
    ):
        """Test that node is created with correct parameters."""
        deterministic_step_component.attach(mock_state_graph, mock_router)

        # Verify DeterministicStepNode creation
        mock_deterministic_step_node_cls.assert_called_once()
        node_call_kwargs = mock_deterministic_step_node_cls.call_args[1]

        assert node_call_kwargs["name"] == f"{component_name}#deterministic_step"
        assert node_call_kwargs["tool_name"] == tool_name
        assert node_call_kwargs["inputs"] == inputs
        assert node_call_kwargs["flow_id"] == flow_id
        assert node_call_kwargs["flow_type"] == flow_type
        assert (
            node_call_kwargs["internal_event_client"]
            == deterministic_step_component.internal_event_client
        )
        assert "validated_tool" in node_call_kwargs
        assert node_call_kwargs["validated_tool"] is not None
        assert (
            node_call_kwargs["validated_tool"]
            == deterministic_step_component.validated_tool
        )

        # Verify UI logging
        assert "ui_history" in node_call_kwargs
        assert isinstance(node_call_kwargs["ui_history"], UIHistory)
        assert node_call_kwargs["ui_history"].events == ui_log_events

    def test_attach_uses_correct_ui_log_writer(
        self,
        mock_deterministic_step_node_cls,
        deterministic_step_component,
        mock_state_graph,
        mock_router,
    ):
        """Test that the correct UI log writer class is used."""
        deterministic_step_component.attach(mock_state_graph, mock_router)

        # Get the ui_history argument
        node_call_kwargs = mock_deterministic_step_node_cls.call_args[1]
        ui_history = node_call_kwargs["ui_history"]

        assert ui_history.writer_class == UILogWriterDeterministicStep


class TestDeterministicStepComponentAttachEdges:
    """Test suite for DeterministicStepComponent graph structure."""

    def test_attach_creates_graph_structure(
        self,
        deterministic_step_component,
        mock_state_graph,
        mock_router,
        component_name,
        mock_deterministic_step_node_cls,
    ):
        """Test that attach method creates proper graph structure."""
        deterministic_step_component.attach(mock_state_graph, mock_router)

        expected_node_name = f"{component_name}#deterministic_step"

        # Verify node was added
        mock_state_graph.add_node.assert_called_once_with(
            expected_node_name, mock_deterministic_step_node_cls.return_value.run
        )

        # Verify conditional edge was added
        mock_state_graph.add_conditional_edges.assert_called_once_with(
            expected_node_name, mock_router.route
        )

    def test_attach_no_internal_routing(
        self,
        deterministic_step_component,
        mock_state_graph,
        mock_router,
    ):
        """Test that component has no internal routing logic."""
        deterministic_step_component.attach(mock_state_graph, mock_router)

        # Should not have any regular edges (only conditional edges to router)
        mock_state_graph.add_edge.assert_not_called()

        # Should have exactly one conditional edge (to the router)
        assert mock_state_graph.add_conditional_edges.call_count == 1


class TestDeterministicStepComponentIntegration:
    """Test suite for DeterministicStepComponent integration aspects."""

    def test_component_requires_tool_name(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
        mock_internal_event_client,
    ):
        """Test that component requires tool_name parameter."""
        with pytest.raises(ValidationError):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                # tool_name is missing
                toolset=mock_toolset,
                internal_event_client=mock_internal_event_client,
            )

    def test_component_requires_toolset(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        tool_name,
        mock_internal_event_client,
    ):
        """Test that component requires toolset parameter."""
        with pytest.raises(ValidationError):
            DeterministicStepComponent(
                name=component_name,
                flow_id=flow_id,
                flow_type=flow_type,
                user=user,
                inputs=["context:user_input"],
                tool_name=tool_name,
                # toolset is missing
                internal_event_client=mock_internal_event_client,
            )

    def test_component_with_empty_ui_log_events(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        tool_name,
        mock_toolset,
        mock_internal_event_client,
    ):
        """Test that component can be created with empty ui_log_events."""
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:user_input"],
            tool_name=tool_name,
            toolset=mock_toolset,
            internal_event_client=mock_internal_event_client,
            # ui_log_events not provided, should use default empty list
        )
        assert component.ui_log_events == []
        assert component.validated_tool is not None

    def test_validated_tool_is_always_set(
        self,
        component_name,
        flow_id,
        flow_type,
        user,
        mock_toolset,
    ):
        """Test that validated_tool is always set after component creation."""
        component = DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            user=user,
            inputs=["context:user_input"],
            tool_name="test_tool",
            toolset=mock_toolset,
        )

        assert component.validated_tool is not None
        assert component.validated_tool == mock_toolset.__getitem__.return_value
