import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.tools import DuoBaseTool
from duo_workflow_service.tools.mcp_tools import (
    UNTRUSTED_MCP_WARNING,
    McpTool,
    convert_mcp_tools_to_configs,
    sanitize_llm_name,
    sanitize_python_identifier,
)


@pytest.mark.asyncio
async def test_convert_mcp_tools_to_configs():
    """Test that convert_mcp_tools_to_configs returns correct config dicts."""
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

    result = convert_mcp_tools_to_configs(mcp_tools)

    assert len(result) == 2

    first_config = result[0]
    second_config = result[1]

    # Check first config
    assert first_config["llm_name"] == "tool1"
    assert first_config["original_name"] == "tool1"
    assert (
        first_config["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nTool 1 description"
    )
    assert first_config["args_schema"] == {}

    # Check second config
    assert second_config["llm_name"] == "tool2"
    assert second_config["original_name"] == "tool2"
    assert (
        second_config["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nTool 2 description"
    )
    assert second_config["args_schema"] == {"properties": {}}


@pytest.mark.asyncio
async def test_mcp_tool_instance_creation_and_execution():
    """Test that McpTool instances can be created from configs and executed."""
    metadata = {"outbox": AsyncMock()}
    mcp_tools = [
        contract_pb2.McpTool(
            name="tool1", description="Tool 1 description", inputSchema="{}"
        ),
    ]

    with patch(
        "duo_workflow_service.tools.mcp_tools._execute_action", new_callable=AsyncMock
    ) as mock_execute_action:
        mock_execute_action.return_value = "Tool execution result"

        # Convert to configs
        configs = convert_mcp_tools_to_configs(mcp_tools)
        config = configs[0]

        # Create instance from config (as ToolsRegistry does)
        tool = McpTool(
            name=config["llm_name"],
            description=config["description"],
            args_schema=config["args_schema"],
            metadata=metadata,
        )
        tool._original_mcp_name = config["original_name"]

        assert tool.name == "tool1"
        assert tool.description == f"{UNTRUSTED_MCP_WARNING}\n\nTool 1 description"
        assert tool.metadata == metadata
        assert tool.args_schema == {}
        assert isinstance(tool, DuoBaseTool)

        test_args = {"arg1": "value1"}
        execution_result = await tool._arun(**test_args)
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
    assert isinstance(tool, DuoBaseTool)
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
            "tool_class": "McpTool",
            "original_mcp_name": "test_tool",  # Falls back to self.name when _original_mcp_name is None
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
    assert isinstance(tool, DuoBaseTool)
    mock_logger.info.assert_called_once_with(
        "Executing MCP tool",
        extra={
            "tool_name": "test_tool",
            "tool_class": "McpTool",
            "original_mcp_name": "test_tool",  # Falls back to self.name when _original_mcp_name is None
            "mcp_tool_args_count": 1,
        },
    )

    # Verify action execution
    mock_execute_action.assert_called_once()
    assert result == "test result"


@pytest.mark.asyncio
async def test_convert_mcp_tools_handles_duplicate_names():
    """Test that tools with duplicate names are both included in configs."""
    mcp_tools = [
        contract_pb2.McpTool(
            name="search", description="First search tool", inputSchema="{}"
        ),
        contract_pb2.McpTool(
            name="delete", description="Delete tool", inputSchema="{}"
        ),
        contract_pb2.McpTool(
            name="search", description="Second search tool", inputSchema="{}"
        ),
    ]

    result = convert_mcp_tools_to_configs(mcp_tools)

    assert len(result) == 3

    # All tools should have their configs, even duplicates
    assert result[0]["llm_name"] == "search"
    assert result[0]["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nFirst search tool"

    assert result[1]["llm_name"] == "delete"
    assert result[1]["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nDelete tool"

    assert result[2]["llm_name"] == "search"
    assert result[2]["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nSecond search tool"


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("Tool-Name_123", "Tool-Name_123"),
        ("tool name!", "tool_name"),
        ("a*b&c", "a_b_c"),
        ("__abc", "abc"),
        ("--abc", "abc"),
    ],
)
def test_sanitize_llm_name_valid(input_name, expected):
    assert sanitize_llm_name(input_name) == expected


@pytest.mark.parametrize(
    "input_name",
    [
        "",
        "!!!",
        "   ",
    ],
)
def test_sanitize_llm_name_invalid(input_name):
    with pytest.raises(ValueError):
        sanitize_llm_name(input_name)


def test_sanitize_llm_name_max_length():
    long_name = "A" * 300
    result = sanitize_llm_name(long_name)
    assert len(result) == 128


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("tool_name", "tool_name"),
        ("Tool123_Name", "Tool123_Name"),
        ("a b c", "a_b_c"),
        ("a*b&c", "a_b_c"),
        ("_abc", "_abc"),
        ("tool$name", "tool_name"),
        ("class", "tool_class"),
        ("if", "tool_if"),
        ("9_name", "tool_9_name"),
        ("value!!!", "value___"),
    ],
)
def test_sanitize_python_identifier(input_name, expected):
    assert sanitize_python_identifier(input_name) == expected


@pytest.mark.parametrize(
    "input_name",
    [
        "",
        "!!!",
        "   ",
    ],
)
def test_sanitize_python_identifier_invalid(input_name):
    with pytest.raises(ValueError):
        sanitize_python_identifier(input_name)
