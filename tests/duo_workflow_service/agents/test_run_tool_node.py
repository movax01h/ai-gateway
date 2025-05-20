"""Test module for RunToolNode class."""

# pylint: disable=file-naming-for-tests

from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.agents.run_tool_node import RunToolNode
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus


@pytest.mark.asyncio
async def test_run_tool_node_execution():
    """Test RunToolNode execution with single tool parameter set."""
    # Mock setup
    tool = AsyncMock()
    tool._arun = AsyncMock(return_value="tool_output")
    tool.name = "test_tool"

    input_parser = Mock(return_value=[{"param1": "value1"}])
    output_parser = Mock(return_value={"updated_key": "updated_value"})

    node = RunToolNode(
        tool=tool, input_parser=input_parser, output_parser=output_parser
    )

    # Execute
    state = {"initial_key": "initial_value"}
    result = await node.run(state)

    # Verify
    input_parser.assert_called_once_with(state)
    tool._arun.assert_called_once_with(param1="value1")
    output_parser.assert_called_once_with(["tool_output"], state)

    assert "ui_chat_log" in result
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.TOOL
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert "updated_key" in result
    assert result["updated_key"] == "updated_value"


@pytest.mark.asyncio
async def test_run_tool_node_multiple_params():
    """Test RunToolNode execution with multiple tool parameter sets."""
    # Mock setup
    tool = AsyncMock()
    tool._arun = AsyncMock(side_effect=["output1", "output2"])
    tool.name = "test_tool"

    input_parser = Mock(return_value=[{"param1": "value1"}, {"param1": "value2"}])
    output_parser = Mock(return_value={"updated_key": "updated_value"})

    node = RunToolNode(
        tool=tool, input_parser=input_parser, output_parser=output_parser
    )

    # Execute
    state = {"initial_key": "initial_value"}
    result = await node.run(state)

    # Verify
    input_parser.assert_called_once_with(state)
    assert tool._arun.call_count == 2
    output_parser.assert_called_once_with(["output1", "output2"], state)

    assert len(result["ui_chat_log"]) == 2
    assert all(
        log["message_type"] == MessageTypeEnum.TOOL for log in result["ui_chat_log"]
    )
    assert all(log["status"] == ToolStatus.SUCCESS for log in result["ui_chat_log"])
