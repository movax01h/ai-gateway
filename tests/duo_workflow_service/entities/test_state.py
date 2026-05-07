from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from duo_workflow_service.entities.state import (
    TOOL_RESPONSE_MAX_DISPLAY_MSG,
    MessageTypeEnum,
    ToolInfo,
    UiChatLog,
    _conversation_history_reducer,
    _ui_chat_log_reducer,
    build_tool_info,
)


class TestConversationHistoryReducer:
    def test_appends_new_messages_to_existing(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": [AIMessage(content="hi there")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 2
        assert result["agent1"][0].content == "hello"
        assert result["agent1"][1].content == "hi there"

    def test_returns_copy_when_new_is_none(self):
        current = {"agent1": [HumanMessage(content="hello")]}

        result = _conversation_history_reducer(current, None)

        assert result == current
        assert result is not current

    def test_adds_new_agent_key(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent2": [HumanMessage(content="world")]}

        result = _conversation_history_reducer(current, new)

        assert "agent1" in result
        assert "agent2" in result
        assert len(result["agent2"]) == 1

    def test_skips_empty_new_messages(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": []}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 1

    def test_handles_empty_current(self):
        current: Dict[str, List[BaseMessage]] = {}
        new = {"agent1": [HumanMessage(content="hello")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 1
        assert result["agent1"][0].content == "hello"

    def test_does_not_mutate_current(self):
        original_messages = [HumanMessage(content="hello")]
        current = {"agent1": original_messages}
        new = {"agent1": [AIMessage(content="hi")]}

        result = _conversation_history_reducer(current, new)

        # Original current dict should not be mutated
        assert len(current["agent1"]) == 1
        assert len(result["agent1"]) == 2

    def test_handles_multiple_agents(self):
        current = {
            "planner": [HumanMessage(content="plan this")],
            "executor": [HumanMessage(content="execute this")],
        }
        new = {
            "planner": [AIMessage(content="here's the plan")],
            "executor": [AIMessage(content="done executing")],
        }

        result = _conversation_history_reducer(current, new)

        assert len(result["planner"]) == 2
        assert len(result["executor"]) == 2

    def test_does_not_trim_messages(self):
        """Verify the reducer appends without trimming.

        Token-based trimming is deferred to agent run time via maybe_compact_history(), so the reducer should never
        discard messages.
        """
        # Build a large history that would have triggered the old trim logic
        large_history = [HumanMessage(content=f"message {i}" * 500) for i in range(100)]
        current = {"agent1": large_history}
        new = {"agent1": [AIMessage(content="new response")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 101

    def test_idempotency_returns_new_object_each_call(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": [AIMessage(content="hi")]}

        result1 = _conversation_history_reducer(current, new)
        result2 = _conversation_history_reducer(current, new)

        assert result1 is not result2
        assert result1 == result2


def test_ui_chat_log_reducer():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "additional_context": None,
            "message_id": None,
        },
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
    ]


def test_ui_chat_log_reducer_idempotency():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    _ui_chat_log_reducer(current, new)
    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
    ]


def test_ui_chat_log_reducer_with_none():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    new: Optional[List[UiChatLog]] = None

    result = _ui_chat_log_reducer(current, new)

    assert result == current
    assert result is not current


def test_ui_chat_log_reducer_with_empty_lists():
    current: List[UiChatLog] = []
    new: Optional[List[UiChatLog]] = []

    result = _ui_chat_log_reducer(current, new)

    assert not result
    assert result is not current


def test_build_tool_info_without_response():
    result = build_tool_info("my_tool", {"arg1": "val1"})

    assert result == ToolInfo(name="my_tool", args={"arg1": "val1"})
    assert "tool_response" not in result


def test_build_tool_info_with_short_response():
    result = build_tool_info("my_tool", {"arg1": "val1"}, tool_response="short output")

    assert result["tool_response"] == "short output"


def test_build_tool_info_truncates_long_string_response():
    long_response = "x" * (TOOL_RESPONSE_MAX_DISPLAY_MSG + 100)

    result = build_tool_info("my_tool", {}, tool_response=long_response)

    assert len(result["tool_response"]) == TOOL_RESPONSE_MAX_DISPLAY_MSG


def test_build_tool_info_does_not_truncate_non_string_response():
    non_string_response = {"key": "value", "data": [1, 2, 3]}

    result = build_tool_info("my_tool", {}, tool_response=non_string_response)

    assert result["tool_response"] == non_string_response
