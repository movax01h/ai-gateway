import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain.tools import BaseTool

from contract import contract_pb2
from duo_workflow_service.tools.mcp_tools import (
    McpTool,
    convert_mcp_tools_to_langchain_tools,
)


@pytest.mark.asyncio
async def test_convert_mcp_tools_to_langchain_tools():
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
        result = convert_mcp_tools_to_langchain_tools(metadata, mcp_tools)

        assert len(result) == 2
        assert all(isinstance(tool, McpTool) for tool in result)
        assert all(isinstance(tool, BaseTool) for tool in result)
        assert result[0].name == "tool1"
        assert result[0].description == "Tool 1 description"
        assert result[1].name == "tool2"
        assert result[1].description == "Tool 2 description"
        assert result[0].metadata == metadata
        assert result[1].metadata == metadata

        assert result[0].args_schema == {}
        assert result[1].args_schema == {"properties": {}}

        test_args = {"arg1": "value1"}
        execution_result = await result[0]._arun(**test_args)
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
