from typing import List, cast
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

from duo_workflow_service.conversation.trimmer import (
    _deduplicate_additional_context,
    _pretrim_large_messages,
    _restore_message_consistency,
    get_messages_profile,
    trim_conversation_history,
)
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext


def test_pretrim_large_messages():
    token_counter = MagicMock()
    max_single_messages_tokens = 100
    # Simulate token count
    token_counter.count_tokens.side_effect = lambda msgs: (
        50 if "small" in msgs[0].content else 150
    )

    messages = [
        HumanMessage(content="This is a small message"),
        HumanMessage(content="This is a large message that exceeds the limit"),
    ]

    # Cast the list to List[BaseMessage] to satisfy mypy
    result = _pretrim_large_messages(
        cast(List[BaseMessage], messages), token_counter, max_single_messages_tokens
    )

    assert len(result) == 2
    assert result[0].content == "This is a small message"
    assert (
        result[1].content
        == "Previous message was too large for context window and was omitted. Please respond based on the visible context."
    )


def test_deduplicate_additional_context():
    messages = [
        HumanMessage(
            content="Message 1",
            additional_kwargs={
                "additional_context": [
                    AdditionalContext(category="issue", content="Extra 1")
                ]
            },
        ),
        HumanMessage(
            content="Message 2",
            additional_kwargs={
                "additional_context": [{"content": "Extra 2"}, {"content": "Extra 1"}]
            },
        ),
        HumanMessage(
            content="Message 3",
            additional_kwargs={
                "additional_context": [{"content": "Extra 2"}, {"content": "Extra 1"}]
            },
        ),
        HumanMessage(
            content="Message 4",
            additional_kwargs={
                "additional_context": [
                    AdditionalContext(category="issue", content="Extra 1"),
                    AdditionalContext(category="issue", content="Extra 2"),
                    AdditionalContext(category="issue", content="Extra 3"),
                ]
            },
        ),
    ]

    result = _deduplicate_additional_context(cast(List[BaseMessage], messages))

    assert len(result) == 4

    assert len(result[0].additional_kwargs["additional_context"]) == 1
    assert result[0].additional_kwargs["additional_context"][0].content == "Extra 1"

    assert len(result[1].additional_kwargs["additional_context"]) == 1
    # Extra 2 was in a dict in the last item and we don't change the type
    assert (
        result[1].additional_kwargs["additional_context"][0].get("content") == "Extra 2"
    )

    # Everything in Message 3 was duplicated from above
    assert len(result[2].additional_kwargs["additional_context"]) == 0

    assert len(result[3].additional_kwargs["additional_context"]) == 1
    assert result[3].additional_kwargs["additional_context"][0].content == "Extra 3"


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
@patch("duo_workflow_service.conversation.trimmer.trim_messages")
def test_trim_conversation_history_with_tool_messages(
    mock_trim_messages,
    trim_result,
    expected_result,
):
    """Test that trim_conversation_history properly handles tool messages.

    This test verifies the integration between trim_messages and _restore_message_consistency to ensure tool messages
    are handled correctly.
    """
    mock_trim_messages.return_value = trim_result

    messages = [
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
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    assert result == expected_result


def test_trim_conversation_history_exceeding_context_limit():
    """Test that messages are trimmed when exceeding context limit."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        HumanMessage(content="second a message"),
        HumanMessage(content="third a message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=22
    )

    # Should trim to fit within context limit
    # System message should be preserved
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Should have fewer messages than original
    assert len(result) <= len(messages)


def test_trim_conversation_history_with_tool_messages_exceeding_context_limit_for_existing_message():
    """Test trimming with tool messages that exceed context limit."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        AIMessage(content="first ai message"),
        ToolMessage(content="first tool message", tool_call_id="tool-call-1"),
        AIMessage(content="second ai message"),
        ToolMessage(content="second tool message", tool_call_id="tool-call-2"),
    ]

    result = trim_conversation_history(
        messages=messages,
        component_name="agent_a",
        max_context_tokens=20,  # Very small limit
    )

    # Should trim but maintain message consistency
    assert len(result) > 0
    # System message should be preserved
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Orphaned tool messages should be converted to HumanMessage
    for msg in result:
        if isinstance(msg, ToolMessage):
            # If it's still a ToolMessage, it should have a valid parent
            assert msg.tool_call_id is not None


def test_trim_conversation_history_single_message_too_large():
    """Test that single oversized messages are replaced with placeholder."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        HumanMessage(content="second a message"),
        HumanMessage(content="This is a very large message" * 100),
    ]

    result = trim_conversation_history(
        messages=messages,
        component_name="agent_a",
        max_context_tokens=1_000,
    )

    # The very large message should be replaced with placeholder
    placeholder_found = any(
        "Previous message was too large for context window" in msg.content
        for msg in result
        if isinstance(msg, HumanMessage)
    )
    assert placeholder_found


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
def test_trim_conversation_history_error_handling(mock_trim_messages):
    """Test fallback mechanism when trim_messages raises an exception."""
    mock_trim_messages.side_effect = ValueError("Simulated error in trim_messages")

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="second message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Verify the fallback mechanism worked
    assert len(result) > 0
    # System message should be retained
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Should keep some recent messages
    assert any(isinstance(msg, HumanMessage) for msg in result)


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
def test_trim_conversation_history_empty_result_handling(mock_trim_messages):
    """Test fallback when trim_messages returns empty list."""
    mock_trim_messages.return_value = []

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="new message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Verify the fallback mechanism worked
    assert len(result) > 0
    # At minimum should have system message
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Should have at least one recent message
    assert any(msg.content == "new message" for msg in result)


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_trim_conversation_history_without_warnings(mock_logger):
    """Test that no warnings are logged when trimming succeeds normally."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="new message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Should not log any warnings for normal operation
    mock_logger.warning.assert_not_called()
    # Should return the messages
    assert len(result) == 2


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_trim_conversation_history_single_human_message_no_unnecessary_fallback(
    mock_logger,
):
    """Test that single human message doesn't trigger unnecessary fallback.

    Reproduces production bug: single human message should not trigger fallback,
    which would lead to ['human', 'human'] pattern in subsequent calls.
    """

    # First call: trim a single human message
    messages_1 = [HumanMessage(content="first message")]

    result_1 = trim_conversation_history(
        messages=messages_1, component_name="agent_a", max_context_tokens=400_000
    )

    assert len(result_1) == 1
    assert result_1[0].content == "first message"

    # Check that fallback warning was NOT triggered
    warning_calls = [
        call
        for call in mock_logger.warning.call_args_list
        if "Trim resulted in empty messages/invalid messages" in str(call)
    ]
    fallback_triggered = len(warning_calls) > 0

    assert (
        not fallback_triggered
    ), "Fallback should not trigger for valid single human message"

    # Second call: add another human message
    messages_2 = [
        HumanMessage(content="first message"),
        HumanMessage(content="second message"),
    ]

    result_2 = trim_conversation_history(
        messages=messages_2, component_name="agent_a", max_context_tokens=400_000
    )

    # Should have both messages
    assert len(result_2) == 2
    assert [msg.type for msg in result_2] == ["human", "human"]
    assert result_2[0].content == "first message"
    assert result_2[1].content == "second message"


def test_get_messages_profile():
    messages = []
    token_counter = ApproximateTokenCounter(agent_name="context_builder")
    roles, tokens = get_messages_profile(messages, token_counter=token_counter)
    assert roles == []
    assert tokens == 0

    messages = [
        HumanMessage(content="Hi"),
        AIMessage(
            content="Hello",
            tool_calls=[
                ToolCall(
                    name="foo", args={"a": 1}, id="toolu_vrtx_01A975mGkpbGsENdtz3hKqej"
                )
            ],
        ),
        ToolMessage(tool_call_id="toolu_vrtx_01A975mGkpbGsENdtz3hKqej", content="hi"),
    ]
    roles, tokens = get_messages_profile(messages, token_counter=token_counter)
    assert roles == ["human", "ai", "tool"]
    assert tokens == pytest.approx(4742, abs=50)
