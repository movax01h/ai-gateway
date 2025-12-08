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
    MessageTypeEnum,
    UiChatLog,
    _conversation_history_reducer,
    _ui_chat_log_reducer,
)


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
