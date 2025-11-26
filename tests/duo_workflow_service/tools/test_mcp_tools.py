import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.tools import BaseTool

from contract import contract_pb2
from duo_workflow_service.tools.mcp_tools import (
    UNTRUSTED_MCP_WARNING,
    McpTool,
    convert_mcp_tools_to_langchain_tool_classes,
)


@pytest.mark.asyncio
async def test_convert_mcp_tools_to_langchain_tool_classes():
    metadata = {"outbox": AsyncMock()}
    mcp_tools = [
        contract_pb2.McpTool(
            name="tool1", description="Tool 1 description", inputSchema="{}"
        ),
        contract_pb2.McpTool(
            name="tool2",
            description="Tool 2 description",
            inputSchema='{"properties":{}}',
        ),
    ]
    with patch(
        "duo_workflow_service.tools.mcp_tools._execute_action", new_callable=AsyncMock
    ) as mock_execute_action:
        mock_execute_action.return_value = "Tool execution result"
        result = convert_mcp_tools_to_langchain_tool_classes(mcp_tools)

        assert len(result) == 2

        first_tool_cls = result[0]
        second_tool_cls = result[1]

        assert first_tool_cls.name == "tool1"
        assert second_tool_cls.name == "tool2"

        first_tool = first_tool_cls(metadata=metadata)
        second_tool = second_tool_cls(metadata=metadata)

        assert first_tool.name == "tool1"
        assert second_tool.name == "tool2"

        expected_description_1 = f"{UNTRUSTED_MCP_WARNING}\n\nTool 1 description"
        expected_description_2 = f"{UNTRUSTED_MCP_WARNING}\n\nTool 2 description"

        assert first_tool.description == expected_description_1
        assert second_tool.description == expected_description_2

        assert first_tool.metadata == metadata
        assert second_tool.metadata == metadata
        assert first_tool.args_schema == {}
        assert second_tool.args_schema == {"properties": {}}

        test_args = {"arg1": "value1"}
        execution_result = await first_tool._arun(**test_args)
        assert execution_result == "Tool execution result"

        mock_execute_action.assert_called_once_with(
            metadata,
            contract_pb2.Action(
                runMCPTool=contract_pb2.RunMCPTool(
                    name="tool1", args=json.dumps(test_args)
                )
            ),
        )


@pytest.mark.asyncio
async def test_mcp_tool_run_method():
    tool = McpTool(name="test_tool", description="Test tool", metadata={})

    with pytest.raises(
        NotImplementedError, match="This tool can only be run asynchronously"
    ):
        tool._run()


@pytest.mark.asyncio
async def test_mcp_tool_format_display_message():
    tool = McpTool(name="test_tool", description="Test tool", metadata={})
    arguments = {"key": "value"}

    message = tool.format_display_message(arguments)
    assert message == "Run MCP tool test_tool: {'key': 'value'}"


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.mcp_tools.structlog")
@patch("duo_workflow_service.tools.mcp_tools.current_event_context")
@patch("duo_workflow_service.tools.mcp_tools._execute_action")
async def test_mcp_tool_logging_with_event_context(
    mock_execute_action, mock_current_event_context, mock_structlog
):
    """Test that MCP tool logs with event context when available."""
    # Import here to avoid import-outside-toplevel warning
    from lib.internal_events.context import (  # pylint: disable=import-outside-toplevel
        EventContext,
    )

    # Setup mocks
    mock_logger = MagicMock()
    mock_structlog.stdlib.get_logger.return_value = mock_logger
    mock_execute_action.return_value = "test result"

    # Create test event context
    test_event_context = EventContext(
        instance_id="test-instance",
        host_name="gitlab.example.com",
        realm="saas",
        is_gitlab_team_member=True,
        global_user_id="user-123",
        correlation_id="corr-456",
    )
    mock_current_event_context.get.return_value = test_event_context

    # Create and run tool
    tool = McpTool(name="test_tool", description="Test tool", metadata={})
    arguments = {"arg1": "value1", "arg2": "value2"}

    result = await tool._arun(**arguments)

    # Verify logging was called with expected context
    mock_logger.info.assert_called_once_with(
        "Executing MCP tool",
        extra={
            "tool_name": "test_tool",
            "mcp_tool_args_count": 2,
            "instance_id": "test-instance",
            "host_name": "gitlab.example.com",
            "realm": "saas",
            "is_gitlab_team_member": "True",
            "global_user_id": "user-123",
            "correlation_id": "corr-456",
        },
    )

    # Verify action execution
    mock_execute_action.assert_called_once()
    assert result == "test result"


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.mcp_tools.structlog")
@patch("duo_workflow_service.tools.mcp_tools.current_event_context")
@patch("duo_workflow_service.tools.mcp_tools._execute_action")
async def test_mcp_tool_logging_without_event_context(
    mock_execute_action, mock_current_event_context, mock_structlog
):
    """Test that MCP tool logs gracefully when event context is not available."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_structlog.stdlib.get_logger.return_value = mock_logger
    mock_execute_action.return_value = "test result"
    mock_current_event_context.get.return_value = None

    # Create and run tool
    tool = McpTool(name="test_tool", description="Test tool", metadata={})
    arguments = {"arg1": "value1"}

    result = await tool._arun(**arguments)

    # Verify logging was called with basic context only
    mock_logger.info.assert_called_once_with(
        "Executing MCP tool",
        extra={
            "tool_name": "test_tool",
            "mcp_tool_args_count": 1,
        },
    )

    # Verify action execution
    mock_execute_action.assert_called_once()
    assert result == "test result"
