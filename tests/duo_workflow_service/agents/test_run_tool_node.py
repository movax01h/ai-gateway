"""Test module for RunToolNode class."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool as langchain_tool

from duo_workflow_service.agents.run_tool_node import RunToolNode
from duo_workflow_service.audit_events.callback_handler import (
    AuditEventCallbackHandler,
)
from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.context import audit_collector_context
from duo_workflow_service.audit_events.event_types import AuditEventType
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus
from duo_workflow_service.security.prompt_security import SecurityException
from lib.internal_events.event_enum import CategoryEnum


@pytest.mark.asyncio
async def test_run_tool_node_execution():
    """Test RunToolNode execution with single tool parameter set."""
    # Mock setup
    tool = AsyncMock()
    tool.ainvoke = AsyncMock(return_value="tool_output")
    tool.name = "test_tool"

    input_parser = Mock(return_value=[{"param1": "value1"}])
    output_parser = Mock(return_value={"updated_key": "updated_value"})

    node = RunToolNode(
        tool=tool,
        input_parser=input_parser,
        output_parser=output_parser,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    # Execute
    state = {"initial_key": "initial_value"}
    result = await node.run(state)

    # Verify
    input_parser.assert_called_once_with(state)
    tool.ainvoke.assert_called_once_with({"param1": "value1"})
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
    tool.ainvoke = AsyncMock(side_effect=["output1", "output2"])
    tool.name = "test_tool"

    input_parser = Mock(return_value=[{"param1": "value1"}, {"param1": "value2"}])
    output_parser = Mock(return_value={"updated_key": "updated_value"})

    node = RunToolNode(
        tool=tool,
        input_parser=input_parser,
        output_parser=output_parser,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    # Execute
    state = {"initial_key": "initial_value"}
    result = await node.run(state)

    # Verify
    input_parser.assert_called_once_with(state)
    assert tool.ainvoke.call_count == 2
    output_parser.assert_called_once_with(["output1", "output2"], state)

    assert len(result["ui_chat_log"]) == 2
    assert all(
        log["message_type"] == MessageTypeEnum.TOOL for log in result["ui_chat_log"]
    )
    assert all(log["status"] == ToolStatus.SUCCESS for log in result["ui_chat_log"])


@pytest.mark.asyncio
async def test_run_tool_node_security_layer():
    """Test RunToolNode execution with security layer."""
    # Mock setup
    tool = AsyncMock()
    # Return outputs with dangerous tags that should be encoded
    tool.ainvoke = AsyncMock(
        side_effect=[
            "output1 with <goal>dangerous tag</goal>",
            "output2 with <system>another tag</system>",
        ]
    )
    tool.name = "test_tool"

    input_parser = Mock(return_value=[{"param1": "value1"}, {"param1": "value2"}])
    output_parser = Mock(return_value={"updated_key": "updated_value"})

    node = RunToolNode(
        tool=tool,
        input_parser=input_parser,
        output_parser=output_parser,
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    # Execute
    state = {"initial_key": "initial_value"}
    result = await node.run(state)
    assert result
    # Verify
    input_parser.assert_called_once_with(state)
    assert tool.ainvoke.call_count == 2

    # Verify that the output_parser received the secured outputs
    output_parser.assert_called_once()
    secured_outputs = output_parser.call_args[0][0]

    # Check that dangerous tags were encoded by the security layer
    assert len(secured_outputs) == 2
    assert secured_outputs[0] == "output1 with &lt;goal&gt;dangerous tag&lt;/goal&gt;"
    assert secured_outputs[1] == "output2 with &lt;system&gt;another tag&lt;/system&gt;"


@pytest.mark.asyncio
async def test_run_tool_node_emits_each_tool_audit_event_once():
    """One tool call yields exactly one invoked and one response audit event."""
    workflow_id = "42"
    collector = AuditEventCollector(
        client=Mock(),
        workflow_id=workflow_id,
        buffer_size=100,
        flush_interval_seconds=1_000,
    )
    handler = AuditEventCallbackHandler(collector=collector, workflow_id=workflow_id)

    @langchain_tool
    def read_file(file_path: str) -> str:
        """Return file contents."""
        return f"contents of {file_path}"

    node = RunToolNode(
        tool=read_file,
        input_parser=Mock(return_value=[{"file_path": "Jenkinsfile"}]),
        output_parser=Mock(return_value={}),
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    token = audit_collector_context.set(collector)
    try:
        # Callbacks reach the tool through the run config, as inside a compiled graph.
        await RunnableLambda(node.run).ainvoke({}, config={"callbacks": [handler]})
    finally:
        audit_collector_context.reset(token)

    events = [
        (event.event_type, event.workflow_id)
        for event in collector._buffer  # pylint: disable=protected-access
    ]
    assert events == [
        (AuditEventType.AI_TOOL_INVOKED, workflow_id),
        (AuditEventType.AI_TOOL_RESPONSE_RECEIVED, workflow_id),
    ]


@pytest.mark.asyncio
async def test_run_tool_node_failure_event_carries_workflow_id():
    """The security rejection RunToolNode emits itself must carry the workflow id."""
    workflow_id = "42"
    collector = AuditEventCollector(
        client=Mock(),
        workflow_id=workflow_id,
        buffer_size=100,
        flush_interval_seconds=1_000,
    )
    tool = AsyncMock()
    tool.ainvoke = AsyncMock(return_value="tool_output")
    tool.name = "test_tool"

    node = RunToolNode(
        tool=tool,
        input_parser=Mock(return_value=[{"param1": "value1"}]),
        output_parser=Mock(return_value={}),
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )

    token = audit_collector_context.set(collector)
    try:
        with patch(
            "duo_workflow_service.agents.run_tool_node.apply_security_scanning",
            side_effect=SecurityException("dangerous content"),
        ):
            result = await node.run({})
    finally:
        audit_collector_context.reset(token)

    assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
    events = [
        (event.event_type, event.workflow_id, event.error_type)
        for event in collector._buffer  # pylint: disable=protected-access
    ]
    assert events == [
        (AuditEventType.AI_TOOL_EXECUTION_FAILED, workflow_id, "SecurityException")
    ]
