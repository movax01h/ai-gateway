# pylint: disable=file-naming-for-tests
"""Tests for Anthropic server-side tool streaming in UserInterface._append_chunk_to_ui_chat_log."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
)
from duo_workflow_service.executor.outbox import Outbox
from lib.context import client_capabilities, gitlab_version


@pytest.fixture(name="outbox")
def outbox_fixture() -> Outbox:
    return MagicMock(spec=Outbox())


@pytest.fixture(name="gl_version_18_7")
def gl_version_18_7_fixture():
    """Set GitLab version to 18.7.0 for client capabilities support."""
    gitlab_version.set("18.7.0")
    yield
    gitlab_version.set(None)


@pytest.fixture(name="checkpoint_notifier")
def checkpoint_notifier_fixture(
    outbox, gl_version_18_7
):  # pylint: disable=unused-argument
    client_capabilities.set({"incremental_streaming"})
    return UserInterface(outbox=outbox, goal="test_goal")


@pytest.mark.parametrize(
    "tool_name",
    [
        "web_search",
        "web_fetch",
        "code_execution",
        "bash_code_execution",
    ],
)
def test_server_tool_use_block_creates_pending_tool_entry(
    checkpoint_notifier, tool_name
):
    """Any server_tool_use block should create a PENDING TOOL entry regardless of tool name."""
    chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtu_abc",
                "name": tool_name,
                "input": {"query": "latest AI research"},
                "index": 0,
            }
        ],
    )

    checkpoint_notifier._append_chunk_to_ui_chat_log(chunk)

    assert len(checkpoint_notifier.ui_chat_log) == 1
    entry = checkpoint_notifier.ui_chat_log[0]
    assert entry["message_type"] == MessageTypeEnum.TOOL
    assert entry["message_sub_type"] == tool_name
    assert entry["status"] == ToolStatus.PENDING
    assert entry["message_id"] == "srvtu_abc"
    assert entry["tool_info"]["name"] == tool_name
    assert entry["tool_info"]["args"] == {"query": "latest AI research"}
    assert "tool_response" not in entry["tool_info"]


@pytest.mark.parametrize(
    ("result_block_type", "result_content"),
    [
        # web_search_tool_result — list of result dicts
        (
            "web_search_tool_result",
            [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "AI in 2025",
                    "page_age": "2025-01-15",
                }
            ],
        ),
        # web_fetch_tool_result — single content object
        (
            "web_fetch_tool_result",
            {"type": "web_fetch", "url": "https://example.com", "content": "body"},
        ),
        # code_execution_tool_result — execution result dict
        (
            "code_execution_tool_result",
            {"type": "code_execution_result", "stdout": "hello\n", "stderr": ""},
        ),
        # hypothetical future result type
        (
            "future_fancy_tool_result",
            {"type": "fancy_result", "data": 42},
        ),
    ],
)
def test_server_tool_result_updates_pending_entry(
    checkpoint_notifier, result_block_type, result_content
):
    """Any *_tool_result block should update the preceding PENDING entry to SUCCESS with raw content."""
    tool_use_chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtu_abc",
                "name": "some_tool",
                "input": {"param": "value"},
                "index": 0,
            }
        ],
    )
    checkpoint_notifier._append_chunk_to_ui_chat_log(tool_use_chunk)

    result_chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": result_block_type,
                "tool_use_id": "srvtu_abc",
                "content": result_content,
                "index": 1,
            }
        ],
    )
    checkpoint_notifier._append_chunk_to_ui_chat_log(result_chunk)

    assert len(checkpoint_notifier.ui_chat_log) == 1
    entry = checkpoint_notifier.ui_chat_log[0]
    assert entry["status"] == ToolStatus.SUCCESS
    # Raw content is stored as-is — no provider-specific parsing.
    assert entry["tool_info"]["tool_response"] == result_content
    assert checkpoint_notifier._server_tool_log_index is None


def test_server_tool_result_without_prior_tool_use_does_not_crash(
    checkpoint_notifier,
):
    """Any *_tool_result arriving with no pending tool entry should not crash."""
    result_chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtu_orphan",
                "content": [],
                "index": 0,
            }
        ],
    )

    # Must not raise and must not modify the log.
    checkpoint_notifier._append_chunk_to_ui_chat_log(result_chunk)

    assert len(checkpoint_notifier.ui_chat_log) == 0


def test_text_before_and_after_server_tool_use(checkpoint_notifier):
    """Text blocks before and after a server tool use/result pair should be accumulated correctly."""
    with patch("duo_workflow_service.checkpointer.notifier.datetime") as mock_dt:
        mock_dt.now.return_value.isoformat.return_value = "2023-01-01T00:00:00+00:00"

        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(id="msg-1", content="I'll look that up.")
        )

        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(
                id="msg-1",
                content=[
                    {
                        "type": "server_tool_use",
                        "id": "srvtu_abc",
                        "name": "some_tool",
                        "input": {"param": "value"},
                        "index": 1,
                    }
                ],
            )
        )

        result_content = {"type": "some_result", "data": "output"}
        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(
                id="msg-1",
                content=[
                    {
                        "type": "some_tool_result",
                        "tool_use_id": "srvtu_abc",
                        "content": result_content,
                        "index": 2,
                    }
                ],
            )
        )

        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(
                id="msg-1",
                content=[{"type": "text", "text": " Based on results.", "index": 3}],
            )
        )

    assert len(checkpoint_notifier.ui_chat_log) == 2

    agent_entry = checkpoint_notifier.ui_chat_log[0]
    assert agent_entry["message_type"] == MessageTypeEnum.AGENT
    assert "I'll look that up." in agent_entry["content"]
    assert "Based on results." in agent_entry["content"]

    tool_entry = checkpoint_notifier.ui_chat_log[1]
    assert tool_entry["message_type"] == MessageTypeEnum.TOOL
    assert tool_entry["status"] == ToolStatus.SUCCESS
    assert tool_entry["tool_info"]["tool_response"] == result_content


@pytest.mark.asyncio
async def test_send_event_messages_stream_with_server_tool(checkpoint_notifier):
    """send_event with a server_tool_use chunk should add a TOOL entry and enqueue a checkpoint."""
    chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtu_xyz",
                "name": "some_tool",
                "input": {"param": "value"},
                "index": 0,
            }
        ],
    )

    await checkpoint_notifier.send_event("messages", (chunk, {}), stream=True)

    assert len(checkpoint_notifier.ui_chat_log) == 1
    entry = checkpoint_notifier.ui_chat_log[0]
    assert entry["message_type"] == MessageTypeEnum.TOOL
    assert entry["status"] == ToolStatus.PENDING
    assert checkpoint_notifier.outbox.put_action.call_count == 1


def test_server_tool_use_block_with_empty_input(checkpoint_notifier):
    """A server_tool_use block with no input should default to an empty dict in tool_info.args."""
    chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtu_noinput",
                "name": "some_tool",
                # no "input" key
                "index": 0,
            }
        ],
    )

    checkpoint_notifier._append_chunk_to_ui_chat_log(chunk)

    entry = checkpoint_notifier.ui_chat_log[0]
    assert entry["tool_info"]["args"] == {}


def test_multiple_server_tool_calls_in_sequence(checkpoint_notifier):
    """Multiple sequential server tool calls should each create their own TOOL entry."""
    result_payloads = [{"data": f"result_{i}"} for i in range(2)]

    for i in range(2):
        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(
                id=f"msg-{i}",
                content=[
                    {
                        "type": "server_tool_use",
                        "id": f"srvtu_{i}",
                        "name": "some_tool",
                        "input": {"param": f"value_{i}"},
                        "index": 0,
                    }
                ],
            )
        )
        checkpoint_notifier._append_chunk_to_ui_chat_log(
            AIMessageChunk(
                id=f"msg-{i}",
                content=[
                    {
                        "type": "some_tool_result",
                        "tool_use_id": f"srvtu_{i}",
                        "content": result_payloads[i],
                        "index": 1,
                    }
                ],
            )
        )

    assert len(checkpoint_notifier.ui_chat_log) == 2
    for i, entry in enumerate(checkpoint_notifier.ui_chat_log):
        assert entry["message_type"] == MessageTypeEnum.TOOL
        assert entry["status"] == ToolStatus.SUCCESS
        assert entry["tool_info"]["tool_response"] == result_payloads[i]


def test_non_dict_blocks_in_list_content_are_skipped(checkpoint_notifier):
    """Non-dict (string) items in a list content block should be silently skipped in block processing.

    The string item is not processed as a block (the ``continue`` branch is taken), but
    ``message.text()`` still aggregates all text content including bare strings.
    The important invariant is that only one AGENT entry is created (not one per block).
    """
    chunk = AIMessageChunk(
        id="msg-1",
        content=[
            "a plain string block",
            {"type": "text", "text": " actual text"},
        ],
    )

    checkpoint_notifier._append_chunk_to_ui_chat_log(chunk)

    # Only one AGENT entry should be created — the non-dict string block is skipped
    # at the block-processing level (the ``continue`` branch), so no duplicate entry.
    assert len(checkpoint_notifier.ui_chat_log) == 1
    assert checkpoint_notifier.ui_chat_log[0]["message_type"] == MessageTypeEnum.AGENT


def test_accumulate_text_chunk_with_empty_text_is_noop(checkpoint_notifier):
    """A chunk with empty string content should not create a log entry."""
    chunk = AIMessageChunk(
        id="msg-1",
        content="",
    )

    checkpoint_notifier._append_chunk_to_ui_chat_log(chunk)

    assert len(checkpoint_notifier.ui_chat_log) == 0


def test_server_tool_result_with_mismatched_tool_use_id_does_not_update(
    checkpoint_notifier,
):
    """A tool_result block whose tool_use_id does not match the pending entry should not update it."""
    use_chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "server_tool_use",
                "id": "srvtu_correct",
                "name": "some_tool",
                "input": {},
                "index": 0,
            }
        ],
    )
    checkpoint_notifier._append_chunk_to_ui_chat_log(use_chunk)

    result_chunk = AIMessageChunk(
        id="msg-1",
        content=[
            {
                "type": "some_tool_result",
                "tool_use_id": "srvtu_wrong",
                "content": {"data": "should not appear"},
                "index": 1,
            }
        ],
    )
    checkpoint_notifier._append_chunk_to_ui_chat_log(result_chunk)

    entry = checkpoint_notifier.ui_chat_log[0]
    assert entry["status"] == ToolStatus.PENDING
    assert "tool_response" not in entry["tool_info"]
