import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain.tools import BaseTool

from contract import contract_pb2
from duo_workflow_service.tools.mcp_tools import (
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
        assert first_tool.description == "Tool 1 description"
        assert second_tool.description == "Tool 2 description"
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
