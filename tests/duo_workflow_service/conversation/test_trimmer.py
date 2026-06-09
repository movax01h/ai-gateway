# pylint: disable=line-too-long,too-many-lines,use-implicit-booleaness-not-comparison
from typing import List, cast
from unittest.mock import patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from duo_workflow_service.conversation.trimmer import (
    _pretrim_large_messages,
    apply_token_based_trim,
    restore_message_consistency,
)

# Error message for invalid tool call responses
INVALID_TOOL_ERROR_MESSAGE = (
    "While processing your request, GitLab Duo Chat encountered a problem making a call to the "
    "invalid_tool tool. Try again, or rephrase your request. If the problem "
    "persists, start a new chat, and/or select a different model for your request."
)


@patch("duo_workflow_service.conversation.trimmer._token_estimator.count")
def test_pretrim_large_messages(mock_count_tokens):
    max_single_messages_tokens = 100
    # Simulate token count
    mock_count_tokens.side_effect = lambda msgs, is_complete_history: (
        50 if "small" in msgs[0].content else 150
    )

    messages = [
        HumanMessage(content="This is a small message"),
        HumanMessage(content="This is a large message that exceeds the limit"),
    ]

    # Cast the list to List[BaseMessage] to satisfy mypy
    result = _pretrim_large_messages(
        cast(List[BaseMessage], messages), max_single_messages_tokens
    )

    assert len(result) == 2
    assert result[0].content == "This is a small message"
    assert (
        result[1].content
        == "Previous message was too large for context window and was omitted. Please respond based on the visible context."
    )


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
    result = restore_message_consistency(messages)

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

    result = restore_message_consistency(messages)

    # The ToolMessage before its parent gets converted to HumanMessage (orphaned),
    # and a synthetic ToolMessage is injected after the AIMessage since its tool_call
    # has no valid ToolMessage following it.
    assert len(result) == 5
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], HumanMessage)  # Converted from ToolMessage
    assert result[2].content == "tool response"
    assert isinstance(result[3], AIMessage)
    assert isinstance(result[4], ToolMessage)  # Synthetic ToolMessage injected
    assert result[4].tool_call_id == "tool-call-1"
    assert "interrupted" in result[4].content.lower()


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch(
    "duo_workflow_service.conversation.trimmer._token_estimator.count",
    return_value=999_999,
)
def test_apply_token_based_trim_preserves_tool_messages(
    _mock_count_tokens,
    mock_trim_messages,
):
    """Test that apply_token_based_trim passes tool messages through unchanged.

    Message consistency repair (orphaned ToolMessages, dangling AIMessages) is intentionally NOT done in
    apply_token_based_trim — it runs on the write path (state reducer) and its output is checkpointed.  Repairing there
    would persist synthetic ToolMessages into the checkpoint, causing the LLM to re-enter an infinite tool-call loop on
    every resume/retry.

    Consistency is repaired at read time in AgentPromptTemplate.invoke and ChatAgentPromptTemplate.invoke, just before
    the prompt is sent to the LLM.
    """

    original_messages = [
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
    # Capture identity and serialized form of each original object before the call
    original_ids = [id(m) for m in original_messages]
    original_serialized = [m.model_dump() for m in original_messages]

    mock_trim_messages.return_value = original_messages.copy()

    result = apply_token_based_trim(
        messages=original_messages, component_name="agent_a", max_context_tokens=400_000
    )

    # apply_token_based_trim must return messages unchanged —
    # no consistency repair, no message type conversion.
    assert [type(m) for m in result.messages] == [type(m) for m in original_messages]
    assert [m.model_dump() for m in result.messages] == original_serialized

    # Original message objects must not have been mutated in place.
    assert [id(m) for m in original_messages] == original_ids
    assert [m.model_dump() for m in original_messages] == original_serialized


def test_apply_token_based_trim_exceeding_context_limit():
    """Test that messages are trimmed when exceeding context limit."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        HumanMessage(content="second a message"),
        HumanMessage(content="third a message"),
    ]

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=22
    )

    # Should trim to fit within context limit
    # System message should be preserved
    assert any(isinstance(msg, SystemMessage) for msg in result.messages)
    # Should have fewer messages than original
    assert len(result.messages) <= len(messages)


def test_apply_token_based_trim_with_tool_messages_exceeding_context_limit_for_existing_message():
    """Test trimming with tool messages that exceed context limit."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first a message"),
        AIMessage(content="first ai message"),
        ToolMessage(content="first tool message", tool_call_id="tool-call-1"),
        AIMessage(content="second ai message"),
        ToolMessage(content="second tool message", tool_call_id="tool-call-2"),
    ]

    result = apply_token_based_trim(
        messages=messages,
        component_name="agent_a",
        max_context_tokens=20,  # Very small limit
    )

    # Should trim but maintain message consistency
    assert len(result.messages) > 0
    # System message should be preserved
    assert any(isinstance(msg, SystemMessage) for msg in result.messages)
    # Orphaned tool messages should be converted to HumanMessage
    for msg in result.messages:
        if isinstance(msg, ToolMessage):
            # If it's still a ToolMessage, it should have a valid parent
            assert msg.tool_call_id is not None


def test_apply_token_based_trim_single_message_too_large():
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
    result = apply_token_based_trim(
        messages=messages,
        component_name="agent_a",
        max_context_tokens=50_000,
    )

    # The very large message should be replaced with placeholder
    placeholder_found = any(
        "Previous message was too large for context window" in msg.content
        for msg in result.messages
        if isinstance(msg, HumanMessage)
    )
    assert placeholder_found


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch(
    "duo_workflow_service.conversation.trimmer._token_estimator.count",
    return_value=999_999,
)
def test_apply_token_based_trim_error_handling(_mock_count_tokens, mock_trim_messages):
    """Test fallback mechanism when trim_messages raises an exception."""

    mock_trim_messages.side_effect = ValueError("Simulated error in trim_messages")

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="second message"),
    ]

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (and raised)
    mock_trim_messages.assert_called_once()
    # Verify the fallback mechanism worked
    assert len(result.messages) > 0
    # System message should be retained
    assert any(isinstance(msg, SystemMessage) for msg in result.messages)
    # Should keep some recent messages (fallback uses min_recent=5)
    assert any(isinstance(msg, HumanMessage) for msg in result.messages)


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

    result = restore_message_consistency(messages)

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

    result = restore_message_consistency(messages)

    # The orphaned ToolMessage should be converted to HumanMessage
    assert len(result) == 3
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], HumanMessage)  # Converted from ToolMessage

    # The converted message should have the same content
    assert result[2].content == INVALID_TOOL_ERROR_MESSAGE


def test_restore_message_consistency_with_dangling_ai_tool_calls():
    """Test that AIMessages with tool_calls but no ToolMessages get synthetic ToolMessages injected.

    This happens when a workflow is retried after a crash: the execution node produces
    an AIMessage with tool_calls, but the tools_executor never ran to produce ToolMessages.
    """
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="I'll run that command now.",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "run_command",
                    "args": {"command": "bundle exec rake secret_tanuki"},
                }
            ],
        ),
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "tool-call-1"
    assert "interrupted" in result[3].content.lower()


def test_restore_message_consistency_with_dangling_multiple_tool_calls():
    """Test that multiple missing ToolMessages are injected for an AIMessage with multiple tool_calls."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="Let me read both files.",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "read_file",
                    "args": {"file_path": "a.rb"},
                },
                {
                    "id": "tool-call-2",
                    "name": "read_file",
                    "args": {"file_path": "b.rb"},
                },
            ],
        ),
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 5
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "tool-call-1"
    assert isinstance(result[4], ToolMessage)
    assert result[4].tool_call_id == "tool-call-2"


def test_restore_message_consistency_with_partial_tool_responses():
    """Test that only missing ToolMessages are injected when some are already present."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="Let me read both files.",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "read_file",
                    "args": {"file_path": "a.rb"},
                },
                {
                    "id": "tool-call-2",
                    "name": "read_file",
                    "args": {"file_path": "b.rb"},
                },
            ],
        ),
        ToolMessage(content="contents of a.rb", tool_call_id="tool-call-1"),
        # tool-call-2 has no ToolMessage
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 5
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    # ToolMessages are emitted in declared tool_call_id order.
    # tool-call-1 has a real ToolMessage, tool-call-2 gets a synthetic one.
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "tool-call-1"
    assert result[3].content == "contents of a.rb"
    assert isinstance(result[4], ToolMessage)
    assert result[4].tool_call_id == "tool-call-2"
    assert "interrupted" in result[4].content.lower()


def test_restore_message_consistency_with_dangling_invalid_tool_calls():
    """Test that AIMessages with invalid_tool_calls but no ToolMessages get synthetic ToolMessages."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="ai message with invalid tool call",
            invalid_tool_calls=[{"id": "invalid-call-1"}],
        ),
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 4
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "invalid-call-1"
    assert "interrupted" in result[3].content.lower()


def test_restore_message_consistency_preserves_complete_conversations():
    """Test that a complete conversation with proper tool_call/ToolMessage pairing is not modified."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="I'll run that command.",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "run_command",
                    "args": {"command": "ls"},
                }
            ],
        ),
        ToolMessage(content="file1.txt\nfile2.txt", tool_call_id="tool-call-1"),
        AIMessage(content="Here are the files."),
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 5
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].content == "file1.txt\nfile2.txt"
    assert isinstance(result[4], AIMessage)
    assert result[4].content == "Here are the files."


def test_restore_message_consistency_dangling_mid_conversation():
    """Test that a dangling AIMessage mid-conversation gets synthetic ToolMessages.

    This can happen when trimming removes ToolMessages but keeps the AIMessage.
    """
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="First tool call",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "read_file",
                    "args": {"file_path": "a.rb"},
                }
            ],
        ),
        # Missing ToolMessage for tool-call-1
        HumanMessage(content="What happened?"),
        AIMessage(content="Let me try again."),
    ]

    result = restore_message_consistency(messages)

    assert len(result) == 6
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert isinstance(result[2], AIMessage)
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == "tool-call-1"
    assert "interrupted" in result[3].content.lower()
    assert isinstance(result[4], HumanMessage)
    assert isinstance(result[5], AIMessage)


def test_restore_message_consistency_tool_message_not_immediately_after_ai():
    """Regression test: ToolMessage matching tool_call_id exists but is not immediately after AIMessage.

    The Anthropic API requires tool_result blocks to appear *immediately* after the
    tool_use block.  A ToolMessage that matches the tool_call_id but is separated by
    an intervening message must NOT be counted as resolved.  The AIMessage should
    receive a synthetic ToolMessage and the displaced ToolMessage should be treated as
    orphaned (converted to a HumanMessage).
    """
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="human message"),
        AIMessage(
            content="ai message with tool call",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "read_file",
                    "args": {"file_path": "a.rb"},
                }
            ],
        ),
        # Intervening message breaks the adjacency requirement
        HumanMessage(content="some other message"),
        # ToolMessage for tool-call-1 appears too late — not immediately after AIMessage
        ToolMessage(content="file contents", tool_call_id="tool-call-1"),
        AIMessage(content="Here is the result."),
    ]

    result = restore_message_consistency(messages)

    # The AIMessage at index 2 must be followed immediately by a synthetic ToolMessage
    # because the real ToolMessage was not adjacent.
    assert result[2].type == "ai"
    assert result[3].type == "tool"
    assert cast(ToolMessage, result[3]).tool_call_id == "tool-call-1"
    assert "interrupted" in result[3].content.lower()

    # The intervening HumanMessage passes through unchanged
    assert result[4].type == "human"
    assert result[4].content == "some other message"

    # The displaced ToolMessage is converted to a HumanMessage (orphaned)
    assert result[5].type == "human"
    assert result[5].content == "file contents"

    # Final AIMessage is unchanged
    assert result[6].type == "ai"
    assert result[6].content == "Here is the result."
    assert len(result) == 7


def test_restore_message_consistency_adds_missing_thinking_field():
    """Test that thinking blocks without a 'thinking' key get it set to empty string.

    This happens when claude-fable-5 returns a thinking block with only a signature and no thinking text. The Anthropic
    API requires the 'thinking' field to be present.
    """
    messages = [
        HumanMessage(content="who is the owner?"),
        AIMessage(
            content=[
                {"type": "thinking", "signature": "abc123"},
                {
                    "type": "tool_use",
                    "name": "gitlab_api_get",
                    "input": {"endpoint": "/api/v4/projects/1"},
                    "id": "toolu_1",
                },
            ],
            tool_calls=[
                {
                    "id": "toolu_1",
                    "name": "gitlab_api_get",
                    "args": {"endpoint": "/api/v4/projects/1"},
                }
            ],
        ),
        ToolMessage(content="project data", tool_call_id="toolu_1"),
    ]

    result = restore_message_consistency(messages)

    ai_msg = result[1]
    assert isinstance(ai_msg, AIMessage)
    thinking_block = ai_msg.content[0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == ""
    assert thinking_block["signature"] == "abc123"


def test_restore_message_consistency_preserves_existing_thinking_field():
    """Test that thinking blocks with an existing 'thinking' key are not modified."""
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "Let me think...",
                    "signature": "sig1",
                },
                {"type": "text", "text": "Here is my answer."},
            ],
        ),
    ]

    result = restore_message_consistency(messages)

    ai_msg = result[1]
    assert isinstance(ai_msg, AIMessage)
    thinking_block = ai_msg.content[0]
    assert thinking_block["thinking"] == "Let me think..."
    assert thinking_block["signature"] == "sig1"


@pytest.mark.parametrize(
    "content, tool_calls, expect_dropped",
    [
        pytest.param("", [], True, id="empty_string_dropped"),
        pytest.param("   ", [], True, id="whitespace_only_dropped"),
        pytest.param("hello", [], False, id="non_empty_kept"),
        pytest.param(
            "",
            [{"id": "tc-1", "name": "some_tool", "args": {}}],
            False,
            id="tool_calls_kept",
        ),
    ],
)
def test_restore_message_consistency_drops_blank_ai_messages(
    content, tool_calls, expect_dropped
):
    messages = [
        HumanMessage(content="user message"),
        AIMessage(content=content, tool_calls=tool_calls),
    ]
    if tool_calls:
        messages.append(ToolMessage(content="result", tool_call_id=tool_calls[0]["id"]))

    result = restore_message_consistency(messages)

    ai_messages = [m for m in result if m.type == "ai"]
    if expect_dropped:
        assert ai_messages == []
    else:
        assert len(ai_messages) == 1
        assert ai_messages[0].content == content


@patch("duo_workflow_service.conversation.trimmer.trim_messages")
@patch(
    "duo_workflow_service.conversation.trimmer._token_estimator.count",
    return_value=999_999,
)
def test_apply_token_based_trim_empty_result_handling(
    _mock_count_tokens, mock_trim_messages
):
    """Test fallback when trim_messages returns empty list."""

    mock_trim_messages.return_value = []

    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="first message"),
        HumanMessage(content="new message"),
    ]

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (and returned [])
    mock_trim_messages.assert_called_once()
    # Verify the fallback mechanism worked
    assert len(result.messages) > 0
    # At minimum should have system message
    assert any(isinstance(msg, SystemMessage) for msg in result.messages)
    # Should have at least one recent message (fallback uses min_recent=3)
    assert any(msg.content == "new message" for msg in result.messages)


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_apply_token_based_trim_without_warnings(mock_logger):
    """Test that no warnings are logged when trimming succeeds normally."""
    messages = [
        SystemMessage(content="system message"),
        HumanMessage(content="new message"),
    ]

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Should not log any warnings for normal operation
    mock_logger.warning.assert_not_called()
    # Should return the messages
    assert len(result.messages) == 2


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_apply_token_based_trim_single_human_message_no_unnecessary_fallback(
    mock_logger,
):
    """Test that single human message doesn't trigger unnecessary fallback.

    Reproduces production bug: single human message should not trigger fallback,
    which would lead to ['human', 'human'] pattern in subsequent calls.
    """

    # First call: trim a single human message
    messages_1 = [HumanMessage(content="first message")]

    result_1 = apply_token_based_trim(
        messages=messages_1, component_name="agent_a", max_context_tokens=400_000
    )

    assert len(result_1.messages) == 1
    assert result_1.messages[0].content == "first message"

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

    result_2 = apply_token_based_trim(
        messages=messages_2, component_name="agent_a", max_context_tokens=400_000
    )

    # Should have both messages
    assert len(result_2.messages) == 2
    assert [msg.type for msg in result_2.messages] == ["human", "human"]
    assert result_2.messages[0].content == "first message"
    assert result_2.messages[1].content == "second message"


@patch("duo_workflow_service.conversation.trimmer.logger")
def test_apply_token_based_trim_skips_when_under_budget(mock_logger):
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

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # All messages should be returned
    assert len(result.messages) == 3
    assert result.messages[0].content == "system message"
    assert result.messages[1].content == "hello"
    assert result.messages[2].content == "hi there"
    assert result.was_trimmed is False

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


@patch(
    "duo_workflow_service.conversation.trimmer._token_estimator.count",
    return_value=999_999,
)
@patch("duo_workflow_service.conversation.trimmer.trim_messages")
def test_apply_token_based_trim_trims_when_over_threshold(
    mock_trim_messages,
    _mock_count_tokens,
):
    """Verify full trimming kicks in when token utilization exceeds TRIM_THRESHOLD."""

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

    result = apply_token_based_trim(
        messages=input_messages, component_name="agent_a", max_context_tokens=400_000
    )

    # trim_messages should have been called (the expensive path ran)
    mock_trim_messages.assert_called_once()
    # Verify the token budget (0.7 * max_context_tokens) was passed, not raw max_context_tokens
    assert mock_trim_messages.call_args.kwargs["max_tokens"] == int(0.7 * 400_000)
    assert len(result.messages) == 2


def test_apply_token_based_trim_under_budget_returns_messages_unchanged():
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

    result = apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=400_000
    )

    # Under budget: no trimming applied, messages pass through as-is
    assert len(result.messages) == 5
    assert isinstance(result.messages[0], SystemMessage)
    assert isinstance(result.messages[1], HumanMessage)
    assert isinstance(result.messages[2], AIMessage)
    assert isinstance(result.messages[3], ToolMessage)
    assert isinstance(result.messages[4], HumanMessage)
    assert result.messages[3].tool_call_id == "call_1"
    assert "completed successfully" in result.messages[4].content


@pytest.mark.parametrize(
    "token_count, expect_trim_called",
    [
        # Just under budget (0.7 * 100_000 = 70_000) → fast path, no trimming
        (69_999, False),
        # Just over budget → expensive path, trimming runs
        (70_001, True),
    ],
)
@patch("duo_workflow_service.conversation.trimmer._token_estimator.count")
@patch("duo_workflow_service.conversation.trimmer.trim_messages")
def test_apply_token_based_trim_threshold_boundary(
    mock_trim_messages,
    mock_count_tokens,
    token_count,
    expect_trim_called,
):
    """Verify trimming triggers exactly at the TRIM_THRESHOLD boundary."""
    mock_count_tokens.return_value = token_count

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

    apply_token_based_trim(
        messages=messages, component_name="agent_a", max_context_tokens=100_000
    )

    if expect_trim_called:
        mock_trim_messages.assert_called_once()
    else:
        mock_trim_messages.assert_not_called()


class TestTrimResultFields:
    """Test that TrimResult has correct metadata."""

    def test_trim_result_was_trimmed_true_when_trimmed(self):
        messages = [
            SystemMessage(content="system message"),
            HumanMessage(content="first a message"),
            HumanMessage(content="second a message"),
            HumanMessage(content="third a message"),
        ]

        result = apply_token_based_trim(
            messages=messages, component_name="agent_a", max_context_tokens=22
        )

        assert result.was_trimmed is True
        assert result.tokens_before > 0
        assert result.messages_before == 4
        assert result.token_budget > 0
        assert result.max_context_tokens == 22

    def test_trim_result_was_trimmed_false_when_under_budget(self):
        messages = [HumanMessage(content="short")]

        result = apply_token_based_trim(
            messages=messages, component_name="agent_a", max_context_tokens=400_000
        )

        assert result.was_trimmed is False
        assert result.messages == messages

    def test_trim_result_empty_messages(self):
        result = apply_token_based_trim(
            messages=[], component_name="agent_a", max_context_tokens=400_000
        )

        assert result.was_trimmed is False
        assert result.messages == []
