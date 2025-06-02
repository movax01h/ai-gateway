from typing import Dict, List, Optional, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from duo_workflow_service.entities.state import (
    MAX_CONTEXT_TOKENS,
    MAX_SINGLE_MESSAGE_TOKENS,
    MessageTypeEnum,
    UiChatLog,
    _conversation_history_reducer,
    _pretrim_large_messages,
    _restore_message_consistency,
    _ui_chat_log_reducer,
)


def test_conversation_history_reducer():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
            AIMessage(content="second message"),
        ],
        "agent_b": [],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="third message"),
        ],
        "agent_c": [
            SystemMessage(content="system message"),
            HumanMessage(content="fourth message"),
        ],
    }

    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
            AIMessage(content="second message"),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="third message"),
        ],
        "agent_c": [
            SystemMessage(content="system message"),
            HumanMessage(content="fourth message"),
        ],
    }


def test_conversation_history_reducer_idempotency():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
            AIMessage(content="second message"),
        ],
        "agent_b": [],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="fourth message"),
        ],
    }

    _conversation_history_reducer(current, new)
    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
            AIMessage(content="second message"),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="fourth message"),
        ],
    }


def test_conversation_history_reducer_with_none():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
        ],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = None

    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
        ],
    }


@patch("duo_workflow_service.entities.state.MAX_CONTEXT_TOKENS", 22)
def test_conversation_history_reducer_exceeding_context_limit():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first a message"),
            HumanMessage(content="second a message"),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="first b message"),
        ],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [HumanMessage(content="third a message")],
        "agent_b": [HumanMessage(content="second b message")],
    }

    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="second a message"),
            HumanMessage(content="third a message"),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="first b message"),
            HumanMessage(content="second b message"),
        ],
    }


@patch("duo_workflow_service.entities.state.MAX_CONTEXT_TOKENS", 20)
def test_conversation_history_reducer_exceeding_context_limit_for_existing_message():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first a message"),
            AIMessage(content="first ai message"),
            ToolMessage(content="first tool message", tool_call_id="tool-call-1"),
            AIMessage(content="second ai message"),
        ]
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [
            ToolMessage(content="second tool message", tool_call_id="tool-call-2")
        ]
    }

    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first tool message"),
            AIMessage(content="second ai message"),
            HumanMessage(content="second tool message"),
        ]
    }


@patch("duo_workflow_service.entities.state.MAX_SINGLE_MESSAGE_TOKENS", 100)
def test_conversation_history_reducer_single_message_too_large():
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first a message"),
            HumanMessage(content="second a message"),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="first b message"),
        ],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [HumanMessage(content="This is a very large message" * 50)],
        "agent_b": [HumanMessage(content="second b message")],
    }

    result = _conversation_history_reducer(current, new)

    assert result == {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first a message"),
            HumanMessage(content="second a message"),
            HumanMessage(
                content="Previous message was too large for context window and was omitted. Please respond based on the visible context."
            ),
        ],
        "agent_b": [
            SystemMessage(content="system message"),
            HumanMessage(content="first b message"),
            HumanMessage(content="second b message"),
        ],
    }


@patch("duo_workflow_service.entities.state.MAX_SINGLE_MESSAGE_TOKENS", 100)
def test_pretrim_large_messages():
    token_counter = MagicMock()
    # Simulate token count
    token_counter.count_tokens.side_effect = lambda msgs: (
        50 if "small" in msgs[0].content else 150
    )

    messages = [
        HumanMessage(content="This is a small message"),
        HumanMessage(content="This is a large message that exceeds the limit"),
    ]

    # Cast the list to List[BaseMessage] to satisfy mypy
    result = _pretrim_large_messages(cast(List[BaseMessage], messages), token_counter)

    assert len(result) == 2
    assert result[0].content == "This is a small message"
    assert (
        result[1].content
        == "Previous message was too large for context window and was omitted. Please respond based on the visible context."
    )


@patch("langchain_core.messages.trim_messages")
def test_conversation_history_reducer_error_handling(mock_trim_messages):
    # Setup mock to raise an exception
    mock_trim_messages.side_effect = ValueError("Simulated error in trim_messages")

    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
        ],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [HumanMessage(content="new message")],
    }

    result = _conversation_history_reducer(current, new)

    # Verify the fallback mechanism worked and check that at least the system message is retained
    assert "agent_a" in result
    assert isinstance(result["agent_a"], list)
    assert len(result["agent_a"]) > 0

    system_messages = [
        msg for msg in result["agent_a"] if isinstance(msg, SystemMessage)
    ]
    assert len(system_messages) > 0


@patch("duo_workflow_service.entities.state.trim_messages")
def test_conversation_history_reducer_empty_result_handling(mock_trim_messages):
    # Setup mock to return an empty list to simulate all messages being trimmed
    mock_trim_messages.return_value = []

    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first message"),
        ],
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [HumanMessage(content="new message")],
    }

    result = _conversation_history_reducer(current, new)

    # Verify the fallback mechanism worked
    # At minimum we should have the system message and the last message
    assert "agent_a" in result
    assert isinstance(result["agent_a"], list)
    assert len(result["agent_a"]) > 0
    assert any(isinstance(msg, SystemMessage) for msg in result["agent_a"])
    assert any(msg.content == "new message" for msg in result["agent_a"])


@patch("duo_workflow_service.entities.state.MAX_CONTEXT_TOKENS", 22)
@patch("duo_workflow_service.entities.state.logger")
@patch("duo_workflow_service.entities.state.trim_messages")
def test_conversation_history_reducer_without_any_warning(
    mock_trim_messages, mock_logger
):
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
        ],
    }

    new: Dict[str, List[BaseMessage]] = {
        "agent_a": [HumanMessage(content="new message")],
    }

    # Configure the mock to return the combined messages (no trimming effect)
    mock_trim_messages.return_value = current["agent_a"] + new["agent_a"]

    _conversation_history_reducer(current, new)

    mock_logger.warning.assert_not_called()


@patch("duo_workflow_service.entities.state.MAX_CONTEXT_TOKENS", 22)
@patch("duo_workflow_service.entities.state.logger")
@patch("duo_workflow_service.entities.state.trim_messages")
def test_conversation_history_reducer_with_loop_warning(
    mock_trim_messages, mock_logger
):
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
            HumanMessage(content="first human message"),
        ],
    }

    new: Dict[str, List[BaseMessage]] = {
        "agent_a": [HumanMessage(content="new human message")],
    }

    # Configure the mock to return only existing messages, indicating possible loop
    mock_trim_messages.return_value = current["agent_a"]

    _conversation_history_reducer(current, new)

    mock_logger.warning.assert_called_with(
        "Trimming resulted in identical message state - possible conversation loop",
        agent_name="agent_a",
    )


def test_ui_chat_log_reducer():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "context_elements": None,
        },
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        }
    ]

    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "context_elements": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        },
    ]


def test_ui_chat_log_reducer_idempotency():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        }
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        }
    ]

    _ui_chat_log_reducer(current, new)
    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
        },
    ]


def test_ui_chat_log_reducer_with_none():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "context_elements": None,
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

    assert result == []
    assert result is not current


@pytest.mark.parametrize(
    "messages, expected_types, expected_contents, expected_tool_call_ids",
    [
        (  # Test case: Orphaned tool message (no parent tool call)
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(content="ai message without tool call"),
                ToolMessage(
                    content="orphaned tool response", tool_call_id="tool-call-1"
                ),
            ],
            [
                SystemMessage,
                HumanMessage,
                AIMessage,
                HumanMessage,
            ],
            [
                "system message",
                "human message",
                "ai message without tool call",
                "orphaned tool response",
            ],
            [None, None, None, None],
        ),
        (  # Test case: Multiple tool call/response pairs
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(
                    content="first ai message with tool call",
                    tool_calls=[
                        {
                            "id": "tool-call-1",
                            "name": "test_tool1",
                            "args": {"arg1": "value1"},
                        }
                    ],
                ),
                ToolMessage(content="first tool response", tool_call_id="tool-call-1"),
                HumanMessage(content="another human message"),
                AIMessage(
                    content="second ai message with tool call",
                    tool_calls=[
                        {
                            "id": "tool-call-2",
                            "name": "test_tool2",
                            "args": {"arg2": "value2"},
                        }
                    ],
                ),
                ToolMessage(content="second tool response", tool_call_id="tool-call-2"),
            ],
            [
                SystemMessage,
                HumanMessage,
                AIMessage,
                ToolMessage,
                HumanMessage,
                AIMessage,
                ToolMessage,
            ],
            [
                "system message",
                "human message",
                "first ai message with tool call",
                "first tool response",
                "another human message",
                "second ai message with tool call",
                "second tool response",
            ],
            [None, None, None, "tool-call-1", None, None, "tool-call-2"],
        ),
        (  # Test case: Mixed valid and orphaned tool messages
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(
                    content="ai message with tool call",
                    tool_calls=[
                        {
                            "id": "tool-call-1",
                            "name": "test_tool",
                            "args": {"arg1": "value1"},
                        }
                    ],
                ),
                ToolMessage(content="valid tool response", tool_call_id="tool-call-1"),
                ToolMessage(
                    content="orphaned tool response", tool_call_id="tool-call-missing"
                ),
            ],
            [SystemMessage, HumanMessage, AIMessage, ToolMessage, HumanMessage],
            [
                "system message",
                "human message",
                "ai message with tool call",
                "valid tool response",
                "orphaned tool response",
            ],
            [None, None, None, "tool-call-1", None],
        ),
        (  # Test case: Tool message with empty ID
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(content="ai message"),
                ToolMessage(
                    content="tool response with empty id", tool_call_id=""  # Empty ID
                ),
            ],
            [SystemMessage, HumanMessage, AIMessage, HumanMessage],
            [
                "system message",
                "human message",
                "ai message",
                "tool response with empty id",
            ],
            [None, None, None, None],
        ),
        # Test case: Tool message with empty content
        (
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(content="ai message"),
                ToolMessage(content="", tool_call_id="tool-call-1"),
            ],
            [SystemMessage, HumanMessage, AIMessage],
            [
                "system message",
                "human message",
                "ai message",
            ],
            [None, None, None],
        ),
        # Test case: Tool message with string "None" content
        (
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                AIMessage(content="ai message"),
                ToolMessage(content="None", tool_call_id="tool-call-1"),
            ],
            [SystemMessage, HumanMessage, AIMessage, HumanMessage],
            [
                "system message",
                "human message",
                "ai message",
                "None",
            ],
            [None, None, None, None],
        ),
    ],
)
def test_restore_message_consistency(
    messages, expected_types, expected_contents, expected_tool_call_ids
):
    result = _restore_message_consistency(messages)

    assert len(result) == len(expected_types)

    for i, (expected_type, expected_content, expected_tool_call_id) in enumerate(
        zip(expected_types, expected_contents, expected_tool_call_ids)
    ):
        assert isinstance(result[i], expected_type)
        assert result[i].content == expected_content

        if isinstance(result[i], ToolMessage) and expected_tool_call_id is not None:
            tool_message = cast(
                ToolMessage, result[i]
            )  # type casting to ToolMessage for linter
            assert tool_message.tool_call_id == expected_tool_call_id


def test_restore_message_consistency_tool_message_before_tool_call():
    ai_message = AIMessage(
        content="ai message with tool call",
        tool_calls=[
            {"id": "tool-call-1", "name": "test_tool", "args": {"arg1": "value1"}}
        ],
    )
    tool_message = ToolMessage(content="tool response", tool_call_id="tool-call-1")

    # (this is an invalid sequence that should be corrected)
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        tool_message,  # Tool message appears before its parent AI message
        ai_message,
    ]

    result = _restore_message_consistency(messages)

    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], HumanMessage)  # Converted from ToolMessage
    assert result[2].content == "tool response"
    assert isinstance(result[3], AIMessage)


@pytest.mark.parametrize(
    "trim_result, expected_result",
    [
        (  # Test case: Valid tool message preserved
            [
                SystemMessage(content="system message"),
                AIMessage(
                    content="ai message with tool call",
                    tool_calls=[
                        {
                            "id": "tool-call-1",
                            "name": "test_tool",
                            "args": {"arg1": "value1"},
                        }
                    ],
                ),
                ToolMessage(content="tool response", tool_call_id="tool-call-1"),
            ],
            [
                SystemMessage(content="system message"),
                AIMessage(
                    content="ai message with tool call",
                    tool_calls=[
                        {
                            "id": "tool-call-1",
                            "name": "test_tool",
                            "args": {"arg1": "value1"},
                        }
                    ],
                ),
                ToolMessage(content="tool response", tool_call_id="tool-call-1"),
            ],
        ),
        (  # Test case: Orphaned tool message converted
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                ToolMessage(
                    content="orphaned tool response", tool_call_id="tool-call-1"
                ),
            ],
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                HumanMessage(content="orphaned tool response"),  # Converted
            ],
        ),
        # Test case: Orphaned tool message with empty content is dropped
        (
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                ToolMessage(content="", tool_call_id="tool-call-1"),
            ],
            [
                SystemMessage(content="system message"),
                HumanMessage(content="human message"),
                # No message for empty content
            ],
        ),
    ],
)
@patch("duo_workflow_service.entities.state.trim_messages")
@patch("duo_workflow_service.entities.state.ApproximateTokenCounter")
@patch("duo_workflow_service.entities.state._restore_message_consistency")
def test_conversation_history_reducer_with_tool_messages(
    mock_restore_message_consistency,
    mock_token_counter,
    mock_trim_messages,
    trim_result,
    expected_result,
):
    mock_trim_messages.return_value = trim_result
    mock_restore_message_consistency.return_value = expected_result

    # Cast the dictionary to satisfy mypy
    current: Dict[str, List[BaseMessage]] = {
        "agent_a": [
            SystemMessage(content="system message"),
        ]
    }

    new: Optional[Dict[str, List[BaseMessage]]] = {
        "agent_a": [
            AIMessage(
                content="ai message with tool call",
                tool_calls=[
                    {
                        "id": "tool-call-1",
                        "name": "test_tool",
                        "args": {"arg1": "value1"},
                    }
                ],
            ),
            ToolMessage(content="tool response", tool_call_id="tool-call-1"),
        ]
    }

    mock_token_counter_instance = MagicMock()
    mock_token_counter.return_value = mock_token_counter_instance
    mock_token_counter_instance.count_tokens.return_value = 10  # Below threshold

    result = _conversation_history_reducer(current, new)
    assert result["agent_a"] == expected_result
