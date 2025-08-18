"""Test suite for DeterministicStepComponent class."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools import BaseTool
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.deterministic_step.component import (
    DeterministicStepComponent,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus
from duo_workflow_service.security.prompt_security import SecurityException
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    """Fixture for mock tool."""
    tool = Mock(spec=BaseTool)
    tool.name = "test_tool"
    tool._arun = AsyncMock(return_value="tool_result")
    return tool


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture(mock_tool):
    """Fixture for mock toolset."""
    toolset = Mock(spec=Toolset)
    toolset.__getitem__ = Mock(return_value=mock_tool)
    toolset.__contains__ = Mock(return_value=True)
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
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture(name="inputs")
def inputs_fixture():
    """Fixture for component inputs."""
    return ["context:user_input", "context:task_description"]


@pytest.fixture(name="deterministic_component")
def deterministic_component_fixture(
    component_name, flow_id, flow_type, inputs, mock_toolset
):
    """Fixture for DeterministicStepComponent instance."""
    return DeterministicStepComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        inputs=inputs,
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
        mock_toolset,
        input_output,
    ):
        """Test that component validates input targets correctly."""
        # This should succeed without raising an exception
        DeterministicStepComponent(
            name=component_name,
            flow_id=flow_id,
            flow_type=flow_type,
            inputs=[input_output],
            toolset=mock_toolset,
            tool_name="test_tool",
        )

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
                inputs=[input_output],
                toolset=mock_toolset,
                tool_name="test_tool",
            )


class TestDeterministicStepComponentEntryHook:
    """Test suite for DeterministicStepComponent entry hook."""

    def test_entry_hook_returns_correct_node_name(
        self, deterministic_component, component_name
    ):
        """Test that __entry_hook__ returns the correct node name."""
        expected_entry_node = f"{component_name}#deterministic_step"
        assert deterministic_component.__entry_hook__() == expected_entry_node


class TestDeterministicStepComponentAttach:
    """Test suite for DeterministicStepComponent attach method."""

    def test_attach_creates_node_and_edges(
        self, deterministic_component, mock_state_graph, mock_router, component_name
    ):
        """Test that attach method creates node and conditional edges."""
        deterministic_component.attach(mock_state_graph, mock_router)

        expected_node_name = f"{component_name}#deterministic_step"

        # Verify node was added
        mock_state_graph.add_node.assert_called_once_with(
            expected_node_name, deterministic_component._execute_tool
        )

        # Verify conditional edges were added
        mock_state_graph.add_conditional_edges.assert_called_once_with(
            expected_node_name, mock_router.route
        )


class TestDeterministicStepComponentOutputs:
    """Test suite for DeterministicStepComponent outputs property."""

    def test_outputs_property_returns_correct_keys(
        self, deterministic_component, component_name
    ):
        """Test that outputs property returns correctly formatted IOKeys."""
        outputs = deterministic_component.outputs

        assert len(outputs) == 2

        # Check ui_chat_log output
        ui_log_output = outputs[0]
        assert ui_log_output.target == "ui_chat_log"
        assert ui_log_output.subkeys is None

        # Check tool_result output
        tool_result_output = outputs[1]
        assert tool_result_output.target == "context"
        assert tool_result_output.subkeys == [component_name, "tool_result"]

    def test_get_output_key_method(self, deterministic_component, component_name):
        """Test that get_output_key returns the correct IOKey for tool result."""
        output_key = deterministic_component.get_output_key()

        assert output_key.target == "context"
        assert output_key.subkeys == [component_name, "tool_result"]


@pytest.mark.asyncio
class TestDeterministicStepComponentExecuteTool:
    """Test suite for DeterministicStepComponent _execute_tool method."""

    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.get_vars_from_state"
    )
    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.PromptSecurity"
    )
    async def test_execute_tool_success(
        self,
        mock_security,
        mock_get_vars,
        deterministic_component,
        base_flow_state,
        mock_tool,
    ):
        """Test successful tool execution."""
        # Setup mocks
        mock_get_vars.return_value = {"param1": "value1"}
        mock_security.apply_security_to_tool_response.return_value = "secure_result"

        result = await deterministic_component._execute_tool(base_flow_state)

        # Verify tool was called
        mock_tool._arun.assert_called_once_with(param1="value1")

        # Verify security was applied
        mock_security.apply_security_to_tool_response.assert_called_once_with(
            response="tool_result", tool_name="test_tool"
        )

        # Verify result structure
        assert "ui_chat_log" in result
        assert "context" in result
        assert result["context"]["test_component"]["tool_result"] == "secure_result"

    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.get_vars_from_state"
    )
    async def test_execute_tool_missing_tool_error(
        self, mock_get_vars, deterministic_component, base_flow_state
    ):
        """Test tool execution when tool is not found in toolset."""
        mock_get_vars.return_value = {"param1": "value1"}
        deterministic_component.toolset.__contains__ = Mock(return_value=False)

        result = await deterministic_component._execute_tool(base_flow_state)

        # Verify error handling
        assert "ui_chat_log" in result
        assert "context" in result
        assert result["context"]["test_component"]["tool_result"] is None
        assert "error" in result["context"]["test_component"]

        # Check error UI log
        ui_log = result["ui_chat_log"][0]
        assert ui_log["status"] == ToolStatus.FAILURE
        assert "not found in toolset" in ui_log["content"]

    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.get_vars_from_state"
    )
    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.PromptSecurity"
    )
    async def test_execute_tool_security_exception(
        self,
        mock_security,
        mock_get_vars,
        deterministic_component,
        base_flow_state,
    ):
        """Test tool execution with security validation failure."""
        mock_get_vars.return_value = {"param1": "value1"}
        mock_security.apply_security_to_tool_response.side_effect = SecurityException(
            "Security error"
        )

        result = await deterministic_component._execute_tool(base_flow_state)

        # Verify error handling
        assert "ui_chat_log" in result
        assert "context" in result
        assert result["context"]["test_component"]["tool_result"] is None
        assert "error" in result["context"]["test_component"]

        # Check error UI log
        ui_log = result["ui_chat_log"][0]
        assert ui_log["status"] == ToolStatus.FAILURE
        assert "Security error" in ui_log["content"]

    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.get_vars_from_state"
    )
    async def test_execute_tool_general_exception(
        self, mock_get_vars, deterministic_component, base_flow_state, mock_tool
    ):
        """Test tool execution with general exception."""
        mock_get_vars.return_value = {"param1": "value1"}
        mock_tool._arun.side_effect = Exception("Tool execution failed")

        result = await deterministic_component._execute_tool(base_flow_state)

        # Verify error handling
        assert "ui_chat_log" in result
        assert "context" in result
        assert result["context"]["test_component"]["tool_result"] is None
        assert "error" in result["context"]["test_component"]

        # Check error UI log
        ui_log = result["ui_chat_log"][0]
        assert ui_log["status"] == ToolStatus.FAILURE
        assert "Tool execution failed" in ui_log["content"]


class TestDeterministicStepComponentIntegration:
    """Integration tests for DeterministicStepComponent."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.component.PromptSecurity"
    )
    async def test_full_workflow_execution(
        self, mock_security, deterministic_component, mock_tool
    ):
        """Test complete workflow execution from state to output."""
        # Setup
        mock_security.apply_security_to_tool_response.return_value = "secure_result"

        state = {
            "status": "in_progress",
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "user_input": "test_input_value",
                "task_description": "test_description",
            },
        }

        # Execute
        result = await deterministic_component._execute_tool(state)

        # Verify complete result structure
        assert "ui_chat_log" in result
        assert "context" in result

        # Check UI log structure
        ui_log = result["ui_chat_log"][0]
        assert ui_log["message_type"] == MessageTypeEnum.TOOL
        assert ui_log["status"] == ToolStatus.SUCCESS
        assert ui_log["tool_info"]["name"] == "test_tool"

        # Check context structure
        context = result["context"]["test_component"]
        assert context["tool_result"] == "secure_result"

        # Verify tool was called with extracted variables
        mock_tool._arun.assert_called_once_with(
            user_input="test_input_value", task_description="test_description"
        )

    def test_component_chaining_scenario(self, flow_id, flow_type, mock_toolset):
        """Test scenario where component reads from another component's output."""
        # Create component that reads from another component's output
        inputs = ["context:read_config.tool_result", "context:edit_instructions"]

        component = DeterministicStepComponent(
            name="process_config",
            flow_id=flow_id,
            flow_type=flow_type,
            inputs=inputs,
            tool_name="edit_file",
            toolset=mock_toolset,
        )

        # Verify output key is correctly namespaced
        output_key = component.get_output_key()
        assert output_key.target == "context"
        assert output_key.subkeys == ["process_config", "tool_result"]

    def test_multiple_input_extraction(self, flow_id, flow_type, mock_toolset):
        """Test component with multiple input parameters."""
        inputs = ["context:search_pattern", "context:target_directory"]

        component = DeterministicStepComponent(
            name="search_content",
            flow_id=flow_id,
            flow_type=flow_type,
            inputs=inputs,
            tool_name="grep",
            toolset=mock_toolset,
        )

        # Verify component is properly configured
        assert len(component.inputs) == 2
        assert component.inputs[0].subkeys == ["search_pattern"]
        assert component.inputs[1].subkeys == ["target_directory"]
