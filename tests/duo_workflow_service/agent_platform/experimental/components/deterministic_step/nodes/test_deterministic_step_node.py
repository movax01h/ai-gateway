from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.experimental.components.deterministic_step.nodes.deterministic_step_node import (
    DeterministicStepNode,
)
from duo_workflow_service.agent_platform.experimental.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from lib.internal_events.event_enum import CategoryEnum, EventEnum


@pytest.fixture(name="mock_prompt_security")
def mock_prompt_security_fixture():
    """Fixture for mocking apply_security_scanning."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.nodes.deterministic_step_node.apply_security_scanning"
    ) as mock_security:
        mock_security.return_value = "Sanitized response"
        yield mock_security


@pytest.fixture(name="mock_logger")
def mock_logger_fixture():
    """Fixture for mocking structlog logger."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.nodes.deterministic_step_node.structlog"
    ) as mock_structlog:
        mock_logger = Mock()
        mock_structlog.stdlib.get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(name="mock_tool_monitoring")
def mock_tool_monitoring_fixture():
    """Fixture for mocking duo_workflow_metrics for tool operations."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.nodes.deterministic_step_node.duo_workflow_metrics"
    ) as mock_metrics:
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics.time_tool_call.return_value = mock_context_manager
        yield mock_metrics


@pytest.fixture(name="mock_get_vars_from_state")
def mock_get_vars_from_state_fixture():
    """Fixture for mocking get_vars_from_state."""
    with patch(
        "duo_workflow_service.agent_platform.experimental.components.deterministic_step.nodes.deterministic_step_node.get_vars_from_state"
    ) as mock_get_vars:
        mock_get_vars.return_value = {"param": "value"}
        yield mock_get_vars


@pytest.fixture(name="mock_tool")
def mock_tool_fixture():
    """Fixture for mock tool."""
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_tool.arun = AsyncMock(return_value="Tool execution result")
    mock_tool.args_schema = None
    return mock_tool


@pytest.fixture(name="tool_responses_key")
def tool_responses_key_fixture():
    """Fixture for tool responses IOKey."""
    return IOKey(target="context", subkeys=["responses"])


@pytest.fixture(name="tool_error_key")
def tool_error_key_fixture():
    """Fixture for tool error IOKey."""
    return IOKey(target="context", subkeys=["errors"])


@pytest.fixture(name="execution_result_key")
def execution_result_key_fixture():
    """Fixture for execution result IOKey."""
    return IOKey(target="context", subkeys=["status"])


@pytest.fixture(name="flow_id")
def flow_id_fixture():
    """Fixture for flow ID."""
    return "test_flow_id"


@pytest.fixture(name="flow_type")
def flow_type_fixture():
    """Fixture for flow type."""
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture(name="inputs")
def inputs_fixture():
    """Fixture for inputs."""
    return [
        IOKey(target="context", subkeys=["input1"]),
        IOKey(target="context", subkeys=["input2"]),
    ]


@pytest.fixture(name="mock_internal_event_client")
def mock_internal_event_client_fixture():
    """Fixture for mock internal event client."""
    mock_client = Mock()
    mock_client.track_event = Mock()
    return mock_client


@pytest.fixture(name="ui_history")
def ui_history_fixture():
    """Fixture for UI history."""
    mock_ui_history = Mock()
    mock_ui_history.log = Mock()
    mock_ui_history.log.success = Mock()
    mock_ui_history.log.error = Mock()
    mock_ui_history.pop_state_updates = Mock(return_value={})
    return mock_ui_history


@pytest.fixture(name="deterministic_step_node")
def deterministic_step_node_fixture(
    inputs,
    mock_tool,
    flow_id,
    flow_type,
    mock_internal_event_client,
    ui_history,
    tool_responses_key,
    tool_error_key,
    execution_result_key,
    mock_tool_monitoring,
    mock_prompt_security,
    mock_logger,
    mock_get_vars_from_state,
):
    """Fixture for DeterministicStepNode instance."""
    return DeterministicStepNode(
        name="test_node",
        tool_name="test_tool",
        inputs=inputs,
        flow_id=flow_id,
        flow_type=flow_type,
        internal_event_client=mock_internal_event_client,
        ui_history=ui_history,
        tool_responses_key=tool_responses_key,
        tool_error_key=tool_error_key,
        execution_result_key=execution_result_key,
        validated_tool=mock_tool,
    )


class TestDeterministicStepNode:
    """Test suite for DeterministicStepNode class focusing on the run method."""

    @pytest.mark.asyncio
    async def test_run_success(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_get_vars_from_state,
        mock_tool_monitoring,
        mock_prompt_security,
        ui_history,
        inputs,
        mock_internal_event_client,
        flow_type,
        flow_id,
    ):
        """Test successful run with tool execution."""
        result = await deterministic_step_node.run(workflow_state)

        # Verify get_vars_from_state was called
        mock_get_vars_from_state.assert_called_once_with(inputs, workflow_state)

        # Verify result structure contains IOKey locations
        assert FlowStateKeys.CONTEXT in result
        assert "responses" in result[FlowStateKeys.CONTEXT]
        assert "errors" in result[FlowStateKeys.CONTEXT]
        assert "status" in result[FlowStateKeys.CONTEXT]
        assert result[FlowStateKeys.CONTEXT]["responses"] == "Sanitized response"
        assert result[FlowStateKeys.CONTEXT]["errors"] is None
        assert result[FlowStateKeys.CONTEXT]["status"] == "success"

        # Verify tool execution was called
        mock_tool.arun.assert_called_once_with({"param": "value"})

        # Verify security sanitization was called
        mock_prompt_security.assert_called_once()
        call_kwargs = mock_prompt_security.call_args[1]
        assert call_kwargs["response"] == "Tool execution result"
        assert call_kwargs["tool_name"] == "test_tool"

        # Verify ui_history.log.success was called
        ui_history.log.success.assert_called_once()
        call_args = ui_history.log.success.call_args
        assert call_args[1]["tool"] == mock_tool
        assert call_args[1]["tool_call_args"] == {"param": "value"}
        assert call_args[1]["tool_response"] == "Sanitized response"
        assert (
            call_args[1]["event"]
            == UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

        # Verify monitoring was called
        mock_tool_monitoring.time_tool_call.assert_called_once_with(
            tool_name="test_tool",
            flow_type=flow_type.value,
        )

        # Verify internal event tracking for success
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_SUCCESS.value
        assert call_args[1]["category"] == flow_type.value

        # Verify additional properties
        additional_props = call_args[1]["additional_properties"]
        assert hasattr(additional_props, "property")
        assert additional_props.property == "test_tool"
        assert hasattr(additional_props, "value")
        assert additional_props.value == flow_id

    @pytest.mark.asyncio
    async def test_run_type_error_handling(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_get_vars_from_state,
    ):
        """Test run handles TypeError during tool execution."""
        # Configure tool to raise TypeError
        type_error = TypeError("Invalid argument type")
        mock_tool.arun = AsyncMock(side_effect=type_error)
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {},
        }

        result = await deterministic_step_node.run(workflow_state)

        # Verify error message in result
        assert FlowStateKeys.CONTEXT in result
        assert "errors" in result[FlowStateKeys.CONTEXT]
        assert "status" in result[FlowStateKeys.CONTEXT]
        error_msg = result[FlowStateKeys.CONTEXT]["errors"]
        assert "Tool test_tool execution failed due to wrong arguments" in error_msg
        assert "The schema is:" in error_msg
        assert result[FlowStateKeys.CONTEXT]["status"] == "failed"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify ui_history.log.error was called
        ui_history.log.error.assert_called_once()

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name="test_tool",
            failure_reason="TypeError",
        )

    @pytest.mark.asyncio
    async def test_run_validation_error_handling(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_get_vars_from_state,
    ):
        """Test run handles ValidationError during tool execution."""
        # Configure tool to raise ValidationError
        validation_error = ValidationError.from_exception_data(
            "ValidationError",
            [{"type": "missing", "loc": ["field"], "msg": "Field required"}],
        )
        mock_tool.arun = AsyncMock(side_effect=validation_error)

        result = await deterministic_step_node.run(workflow_state)

        # Verify error message in result
        assert FlowStateKeys.CONTEXT in result
        assert "errors" in result[FlowStateKeys.CONTEXT]
        assert "status" in result[FlowStateKeys.CONTEXT]
        error_msg = result[FlowStateKeys.CONTEXT]["errors"]
        assert "raised validation error" in error_msg
        assert result[FlowStateKeys.CONTEXT]["status"] == "failed"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name="test_tool",
            failure_reason="ValidationError",
        )

    @pytest.mark.asyncio
    async def test_run_generic_exception_handling(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_get_vars_from_state,
    ):
        """Test run handles generic exceptions during tool execution."""
        # Configure tool to raise generic exception
        generic_error = Exception("Generic error")
        mock_tool.arun = AsyncMock(side_effect=generic_error)

        result = await deterministic_step_node.run(workflow_state)

        # Verify error message in result
        assert FlowStateKeys.CONTEXT in result
        assert "errors" in result[FlowStateKeys.CONTEXT]
        assert "status" in result[FlowStateKeys.CONTEXT]
        error_msg = result[FlowStateKeys.CONTEXT]["errors"]
        assert "Tool runtime exception due to Generic error" in error_msg
        assert result[FlowStateKeys.CONTEXT]["status"] == "failed"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Check that additional_properties contains error information
        additional_props = call_args[1]["additional_properties"]
        assert hasattr(additional_props, "property")
        assert additional_props.property == "test_tool"
        assert hasattr(additional_props, "extra")
        assert additional_props.extra["error"] == "Generic error"
        assert additional_props.extra["error_type"] == "Exception"

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name="test_tool",
            failure_reason="Exception",
        )


class TestDeterministicStepNodeEdgeCases:
    """Test suite for edge cases in DeterministicStepNode."""

    @pytest.mark.asyncio
    async def test_run_with_empty_tool_args(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_get_vars_from_state,
    ):
        """Test run with empty tool arguments."""
        mock_get_vars_from_state.return_value = {}

        result = await deterministic_step_node.run(workflow_state)

        # Verify tool was called with empty args
        mock_tool.arun.assert_called_once_with({})
        # Verify successful execution
        assert result[FlowStateKeys.CONTEXT]["status"] == "success"

    @pytest.mark.asyncio
    async def test_run_with_tool_without_schema(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_get_vars_from_state,
    ):
        """Test TypeError formatting when tool has no args_schema."""
        # Configure tool with no schema
        mock_tool.args_schema = None
        mock_tool.arun = AsyncMock(side_effect=TypeError("No args"))

        result = await deterministic_step_node.run(workflow_state)

        # Verify error message mentions no arguments
        error_msg = result[FlowStateKeys.CONTEXT]["errors"]
        assert "The tool does not accept any argument" in error_msg

    @pytest.mark.asyncio
    async def test_ui_history_state_updates(
        self,
        deterministic_step_node,
        workflow_state,
        ui_history,
        mock_get_vars_from_state,
    ):
        """Test that UI history state updates are properly merged."""
        # Configure ui_history to return state updates
        ui_state_updates = {
            "ui_messages": ["message1", "message2"],
            "ui_state": {"key": "value"},
        }
        ui_history.pop_state_updates.return_value = ui_state_updates

        result = await deterministic_step_node.run(workflow_state)

        # Verify UI state updates are included in result
        assert "ui_messages" in result
        assert result["ui_messages"] == ["message1", "message2"]
        assert "ui_state" in result
        assert result["ui_state"] == {"key": "value"}
        assert FlowStateKeys.CONTEXT in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_response,sanitized_response",
        [
            (["item1", "item2"], ["sanitized1", "sanitized2"]),
            ({"key": "value"}, {"key": "sanitized"}),
        ],
        ids=["list_type", "dict_type"],
    )
    async def test_response_with_complex_types(
        self,
        deterministic_step_node,
        workflow_state,
        mock_tool,
        mock_get_vars_from_state,
        mock_prompt_security,
        tool_response,
        sanitized_response,
    ):
        """Test that list and dict responses are handled properly."""
        mock_tool.arun = AsyncMock(return_value=tool_response)
        mock_prompt_security.return_value = sanitized_response

        result = await deterministic_step_node.run(workflow_state)

        # Verify security was called with original response
        mock_prompt_security.assert_called_once()
        call_kwargs = mock_prompt_security.call_args[1]
        assert call_kwargs["response"] == tool_response
        assert call_kwargs["tool_name"] == "test_tool"

        # Should handle response successfully
        assert FlowStateKeys.CONTEXT in result
        assert "responses" in result[FlowStateKeys.CONTEXT]
        assert result[FlowStateKeys.CONTEXT]["responses"] == sanitized_response
