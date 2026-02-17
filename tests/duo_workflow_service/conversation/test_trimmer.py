from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from duo_workflow_service.conversation.trimmer import (
    _build_tool_call_indices,
    _deduplicate_additional_context,
    _estimate_tokens_from_history,
    _pretrim_large_messages,
    _restore_message_consistency,
    trim_conversation_history,
)
from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter
from duo_workflow_service.workflows.type_definitions import AdditionalContext

# Error message for invalid tool call responses
INVALID_TOOL_ERROR_MESSAGE = (
    "While processing your request, GitLab Duo Chat encountered a problem making a call to the "
    "invalid_tool tool. Try again, or rephrase your request. If the problem "
    "persists, start a new chat, and/or select a different model for your request."
)


def test_pretrim_large_messages():
    token_counter = MagicMock()
    max_single_messages_tokens = 100
    # Simulate token count
    token_counter.count_tokens.side_effect = lambda msgs, include_tool_tokens=True: (
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
                    content="tool response with empty id",
                    tool_call_id="",  # Empty ID
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
@patch("duo_workflow_service.conversation.trimmer.TikTokenCounter")
@patch("duo_workflow_service.conversation.trimmer._estimate_tokens_from_history")
def test_trim_conversation_history_with_tool_messages(
    mock_estimate_tokens,
    mock_token_counter_cls,
    mock_trim_messages,
    trim_result,
    expected_result,
):
    """Test that trim_conversation_history properly handles tool messages.

    This test verifies the integration between trim_messages and _restore_message_consistency to ensure tool messages
    are handled correctly.
    """
    # Return a high token count so the fast paths don't skip trimming
    mock_estimate_tokens.return_value = 999_999
    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = 999_999
    mock_token_counter_cls.return_value = mock_counter

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
    large_content = "This is a very large message. " * 5000  # ~30k tokens
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        HumanMessage(content="second a message"),
        HumanMessage(content=large_content),
    ]

    # Budget = 0.7 * 50_000 = 35_000; single message limit = 0.65 * 35_000 = 22_750
    # The large message (~30k tokens) exceeds the single message limit
    result = trim_conversation_history(
        messages=messages,
        component_name="agent_a",
        max_context_tokens=50_000,
    )

    # The very large message should be replaced with placeholder
    placeholder_found = any(
        "Previous message was too large for context window" in msg.content
        for msg in result
        if isinstance(msg, HumanMessage)
    )
    assert placeholder_found


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch("duo_workflow_service.conversation.trimmer.TikTokenCounter")
@patch("duo_workflow_service.conversation.trimmer._estimate_tokens_from_history")
def test_trim_conversation_history_error_handling(
    mock_estimate_tokens, mock_token_counter_cls, mock_trim_messages
):
    """Test fallback mechanism when trim_messages raises an exception."""
    # Force the expensive path by reporting high token count
    mock_estimate_tokens.return_value = 999_999
    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = 999_999
    mock_token_counter_cls.return_value = mock_counter

    mock_trim_messages.side_effect = ValueError("Simulated error in trim_messages")

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="second message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (and raised)
    mock_trim_messages.assert_called_once()
    # Verify the fallback mechanism worked
    assert len(result) > 0
    # System message should be retained
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Should keep some recent messages (fallback uses min_recent=5)
    assert any(isinstance(msg, HumanMessage) for msg in result)


def test_build_tool_call_indices_with_valid_tool_calls():
    """Test that _build_tool_call_indices correctly tracks valid tool calls."""
    messages = [
        SystemMessage(content="system message"),
        AIMessage(
            content="ai message with tool call",
            tool_calls=[
                {"id": "tool-call-1", "name": "test_tool", "args": {"arg1": "value1"}},
                {
                    "id": "tool-call-2",
                    "name": "another_tool",
                    "args": {"arg2": "value2"},
                },
            ],
        ),
        AIMessage(
            content="another ai message",
            tool_calls=[
                {"id": "tool-call-3", "name": "third_tool", "args": {"arg3": "value3"}},
            ],
        ),
    ]

    indices = _build_tool_call_indices(messages)

    assert indices == {
        "tool-call-1": 1,
        "tool-call-2": 1,
        "tool-call-3": 2,
    }


def test_build_tool_call_indices_with_invalid_tool_calls():
    """Test that _build_tool_call_indices correctly tracks invalid tool calls."""
    messages = [
        SystemMessage(content="system message"),
        AIMessage(
            content="ai message with invalid tool calls",
            invalid_tool_calls=[
                {
                    "id": "invalid-call-1",
                },
                {
                    "id": "invalid-call-2",
                },
            ],
        ),
        AIMessage(
            content="another ai message",
            invalid_tool_calls=[
                {
                    "id": "invalid-call-3",
                },
            ],
        ),
    ]

    indices = _build_tool_call_indices(messages)

    assert indices == {
        "invalid-call-1": 1,
        "invalid-call-2": 1,
        "invalid-call-3": 2,
    }


def test_build_tool_call_indices_with_mixed_tool_calls():
    """Test that _build_tool_call_indices handles both valid and invalid tool calls."""
    messages = [
        SystemMessage(content="system message"),
        AIMessage(
            content="ai message with both valid and invalid tool calls",
            tool_calls=[
                {"id": "valid-call-1", "name": "test_tool", "args": {"arg1": "value1"}},
            ],
            invalid_tool_calls=[
                {
                    "id": "invalid-call-1",
                },
            ],
        ),
    ]

    indices = _build_tool_call_indices(messages)

    assert indices == {
        "valid-call-1": 1,
        "invalid-call-1": 1,
    }


def test_restore_message_consistency_with_invalid_tool_call_responses():
    """Test that ToolMessages from invalid tool calls are preserved when they have valid parents."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="ai message with invalid tool call",
            invalid_tool_calls=[{"id": "invalid-call-1"}],
        ),
        ToolMessage(
            content=INVALID_TOOL_ERROR_MESSAGE,
            tool_call_id="invalid-call-1",
        ),
    ]

    result = _restore_message_consistency(messages)

    # All messages should be preserved since the ToolMessage has a valid parent
    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)

    # The ToolMessage should be preserved as-is
    assert result[3].content == INVALID_TOOL_ERROR_MESSAGE
    assert result[3].tool_call_id == "invalid-call-1"


def test_restore_message_consistency_with_orphaned_invalid_tool_response():
    """Test that orphaned ToolMessages from invalid tool calls are converted to HumanMessages."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        # No AIMessage with invalid_tool_calls - this makes the ToolMessage orphaned
        ToolMessage(
            content=INVALID_TOOL_ERROR_MESSAGE,
            tool_call_id="missing-invalid-call",
        ),
    ]

    result = _restore_message_consistency(messages)

    # The orphaned ToolMessage should be converted to HumanMessage
    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], HumanMessage)  # Converted from ToolMessage

    # The converted message should have the same content
    assert result[2].content == INVALID_TOOL_ERROR_MESSAGE


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch("duo_workflow_service.conversation.trimmer.TikTokenCounter")
@patch("duo_workflow_service.conversation.trimmer._estimate_tokens_from_history")
def test_trim_conversation_history_empty_result_handling(
    mock_estimate_tokens, mock_token_counter_cls, mock_trim_messages
):
    """Test fallback when trim_messages returns empty list."""
    # Force the expensive path by reporting high token count
    mock_estimate_tokens.return_value = 999_999
    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = 999_999
    mock_token_counter_cls.return_value = mock_counter

    mock_trim_messages.return_value = []

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="new message"),
    ]

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (and returned [])
    mock_trim_messages.assert_called_once()
    # Verify the fallback mechanism worked
    assert len(result) > 0
    # At minimum should have system message
    assert any(isinstance(msg, SystemMessage) for msg in result)
    # Should have at least one recent message (fallback uses min_recent=3)
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


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_trim_conversation_history_skips_when_under_budget(mock_logger):
    """Verify that when messages are well under budget, expensive trimming is skipped but the function still returns the
    messages (with maintenance applied)."""
    messages = cast(
        List[BaseMessage],
        [
            SystemMessage(content="system message"),
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ],
    )

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # All messages should be returned
    assert len(result) == 3
    assert result[0].content == "system message"
    assert result[1].content == "hello"
    assert result[2].content == "hi there"

    # Should log the skip message with utilization info
    skip_calls = [
        call
        for call in mock_logger.info.call_args_list
        if "Skipping trimming" in str(call)
    ]
    assert len(skip_calls) == 1

    # Should NOT have logged "Starting trimming" (the expensive path)
    start_calls = [
        call
        for call in mock_logger.info.call_args_list
        if "Starting trimming" in str(call)
    ]
    assert len(start_calls) == 0


@patch("duo_workflow_service.conversation.trimmer._estimate_tokens_from_history")
@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch("duo_workflow_service.conversation.trimmer.TikTokenCounter")
def test_trim_conversation_history_trims_when_over_threshold(
    mock_token_counter_cls,
    mock_trim_messages,
    mock_estimate_tokens,
):
    """Verify full trimming kicks in when token utilization exceeds TRIM_THRESHOLD."""
    # Make fast path estimation return over budget so we proceed to slow path
    mock_estimate_tokens.return_value = 999_999

    # Make the token counter report high utilization (over 80% of budget)
    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = 999_999
    mock_token_counter_cls.return_value = mock_counter

    input_messages = cast(
        List[BaseMessage],
        [
            SystemMessage(content="system message"),
            HumanMessage(content="hello"),
            AIMessage(content="response"),
        ],
    )

    trimmed = cast(
        List[BaseMessage],
        [
            SystemMessage(content="system message"),
            HumanMessage(content="hello"),
        ],
    )
    mock_trim_messages.return_value = trimmed

    result = trim_conversation_history(
        messages=input_messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (the expensive path ran)
    mock_trim_messages.assert_called_once()
    # Verify the token budget (0.7 * max_context_tokens) was passed, not raw max_context_tokens
    assert mock_trim_messages.call_args.kwargs["max_tokens"] == int(0.7 * 400_000)
    assert len(result) == 2


def test_trim_conversation_history_under_budget_returns_messages_unchanged():
    """Verify messages are returned unchanged when under budget.

    Uses a realistic conversation: system prompt, user message, AI with tool_calls,
    tool response, and a trailing HumanMessage (as produced by ToolNodeWithErrorCorrection).
    The expensive trimming pipeline is skipped when within the token budget.
    """
    messages = cast(
        List[BaseMessage],
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Please help me with this task."),
            AIMessage(
                content="I'll use a tool to help.",
                tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
            ),
            ToolMessage(content="search result data", tool_call_id="call_1"),
            HumanMessage(
                content="Tool execution completed successfully after 0 correction attempts."
            ),
        ],
    )

    result = trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Under budget: no trimming applied, messages pass through as-is
    assert len(result) == 5
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert isinstance(result[4], HumanMessage)
    assert result[3].tool_call_id == "call_1"
    assert "completed successfully" in result[4].content


@pytest.mark.parametrize(
    "token_count, expect_trim_called",
    [
        # Just under budget (0.7 * 100_000 = 70_000) → fast path, no trimming
        (69_999, False),
        # Just over budget → expensive path, trimming runs
        (70_001, True),
    ],
)
@patch("duo_workflow_service.conversation.trimmer._estimate_tokens_from_history")
@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch("duo_workflow_service.conversation.trimmer.TikTokenCounter")
def test_trim_conversation_history_threshold_boundary(
    mock_token_counter_cls,
    mock_trim_messages,
    mock_estimate_tokens,
    token_count,
    expect_trim_called,
):
    """Verify trimming triggers exactly at the TRIM_THRESHOLD boundary."""
    # Fast path 2 uses _estimate_tokens_from_history - mock it to return the test token_count
    mock_estimate_tokens.return_value = token_count

    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = token_count
    mock_token_counter_cls.return_value = mock_counter

    mock_trim_messages.return_value = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
    ]

    messages = cast(
        List[BaseMessage],
        [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
            AIMessage(content="response"),
        ],
    )

    trim_conversation_history(
        messages=messages, component_name="agent_a", max_context_tokens=100_000
    )

    if expect_trim_called:
        mock_trim_messages.assert_called_once()
    else:
        mock_trim_messages.assert_not_called()


class TestEstimateTokensFromHistory:
    def test_empty_messages(self):
        token_counter = TikTokenCounter("")
        assert _estimate_tokens_from_history([], token_counter) == 0

    def test_uses_total_tokens_from_ai_message(self):
        """When there's an AIMessage with usage_metadata, use its total_tokens."""
        human = HumanMessage(content="Hello")
        ai = AIMessage(
            content="Hi there!",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        messages = [human, ai]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        assert result == 15

    def test_estimates_trailing_messages(self):
        """Messages after the last AIMessage should be estimated with TikTokenCounter."""
        human1 = HumanMessage(content="Hello")
        ai = AIMessage(
            content="Hi!",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        human2 = HumanMessage(content="How are you?")
        messages = [human1, ai, human2]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        expected = 15 + token_counter.count_tokens([human2], include_tool_tokens=False)
        assert result == expected

    def test_no_ai_message_estimates_all(self):
        """Without AIMessage, all messages should be estimated with TikTokenCounter."""
        messages = [
            HumanMessage(content="Hello"),
            HumanMessage(content="How are you?"),
        ]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        expected = token_counter.count_tokens(messages, include_tool_tokens=False)
        assert result == expected

    def test_uses_latest_ai_message(self):
        """Should use the last AIMessage's total_tokens as baseline."""
        human1 = HumanMessage(content="First")
        ai1 = AIMessage(
            content="Response 1",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        human2 = HumanMessage(content="Second")
        ai2 = AIMessage(
            content="Response 2",
            usage_metadata={
                "input_tokens": 25,
                "output_tokens": 10,
                "total_tokens": 35,
            },
        )
        messages = [human1, ai1, human2, ai2]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        assert result == 35

    def test_multi_turn_with_trailing(self):
        """Multi-turn conversation with a new user message at the end."""
        human1 = HumanMessage(content="What is Python?")
        ai1 = AIMessage(
            content="Python is a programming language.",
            usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        )
        human2 = HumanMessage(content="Who created it?")
        ai2 = AIMessage(
            content="Guido van Rossum created Python.",
            usage_metadata={"input_tokens": 25, "output_tokens": 7, "total_tokens": 32},
        )
        human3 = HumanMessage(content="When was it released?")
        messages = [human1, ai1, human2, ai2, human3]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        expected = 32 + token_counter.count_tokens([human3], include_tool_tokens=False)
        assert result == expected

    def test_ai_message_without_usage_metadata_skipped(self):
        """AIMessage without usage_metadata should be treated like other messages."""
        human = HumanMessage(content="Hello")
        ai_no_metadata = AIMessage(content="Hi there!")
        messages = [human, ai_no_metadata]
        token_counter = TikTokenCounter("")

        result = _estimate_tokens_from_history(messages, token_counter)

        expected = token_counter.count_tokens(messages, include_tool_tokens=False)
        assert result == expected
