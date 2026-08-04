import pytest

from duo_workflow_service.entities.server_tool_blocks import (
    AgentTextSegment,
    ServerToolBoundary,
    build_anthropic_tool_ui_chat_log,
    is_anthropic_server_tool_result_block,
    is_anthropic_server_tool_use_block,
    split_content_around_server_tools,
)
from duo_workflow_service.entities.state import MessageTypeEnum, ToolStatus


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"}, True),
        ({"type": "tool_use", "id": "toolu_1", "name": "read_file"}, False),
        ({"type": "text", "text": "hello"}, False),
        ({"type": "web_search_tool_result", "tool_use_id": "srvtu_1"}, False),
        ({}, False),
        ("not-a-dict", False),
        (None, False),
    ],
)
def test_is_anthropic_server_tool_use_block(block, expected):
    assert is_anthropic_server_tool_use_block(block) is expected


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"type": "web_search_tool_result", "tool_use_id": "srvtu_1"}, True),
        ({"type": "web_fetch_tool_result", "tool_use_id": "srvtu_1"}, True),
        ({"type": "code_execution_tool_result", "tool_use_id": "srvtu_1"}, True),
        ({"type": "tool_result", "tool_use_id": "toolu_1"}, False),
        ({"type": "server_tool_use", "id": "srvtu_1"}, False),
        ({"type": "text", "text": "hello"}, False),
        ({}, False),
        ("not-a-dict", False),
    ],
)
def test_is_anthropic_server_tool_result_block(block, expected):
    assert is_anthropic_server_tool_result_block(block) is expected


def test_build_anthropic_tool_ui_chat_log_pending_without_result():
    use_block = {
        "type": "server_tool_use",
        "id": "srvtu_1",
        "name": "web_search",
        "input": {"query": "latest AI research"},
    }

    entry = build_anthropic_tool_ui_chat_log(use_block)

    assert entry["message_type"] == MessageTypeEnum.TOOL
    assert entry["message_sub_type"] == "web_search"
    assert entry["status"] == ToolStatus.PENDING
    assert entry["content"] == "Using web_search"
    assert entry["message_id"] == "srvtu_1"
    assert entry["tool_info"]["name"] == "web_search"
    assert entry["tool_info"]["args"] == {"query": "latest AI research"}
    # No result yet -> no tool_response key.
    assert "tool_response" not in entry["tool_info"]


def test_build_anthropic_tool_ui_chat_log_success_with_result():
    use_block = {
        "type": "server_tool_use",
        "id": "srvtu_1",
        "name": "web_search",
        "input": {"query": "gitlab duo"},
    }
    result_content = [
        {"type": "web_search_result", "url": "https://x", "title": "X"},
    ]
    result_block = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtu_1",
        "content": result_content,
    }

    entry = build_anthropic_tool_ui_chat_log(
        use_block, result_block, component_name="chat"
    )

    assert entry["status"] == ToolStatus.SUCCESS
    assert entry["message_id"] == "srvtu_1"
    assert entry["component_name"] == "chat"
    assert entry["tool_info"]["tool_response"] == result_content


def test_build_anthropic_tool_ui_chat_log_defaults_for_missing_fields():
    entry = build_anthropic_tool_ui_chat_log({"type": "server_tool_use"})

    assert entry["message_sub_type"] == "server_tool"
    assert entry["tool_info"]["name"] == "server_tool"
    assert entry["tool_info"]["args"] == {}
    assert entry["message_id"] is None


def test_build_anthropic_tool_ui_chat_log_redacts_secrets_in_result():
    leaked_token = "gh" + "p_" + "1234567890abcdefghijklmnopqrstuvwxyz"
    use_block = {"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"}
    result_block = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtu_1",
        "content": [{"type": "web_search_result", "snippet": f"token {leaked_token}"}],
    }

    entry = build_anthropic_tool_ui_chat_log(use_block, result_block)

    snippet = entry["tool_info"]["tool_response"][0]["snippet"]
    assert leaked_token not in snippet
    assert "[REDACTED]" in snippet


def test_split_content_around_server_tools_segments_and_boundaries():
    content = [
        {"type": "text", "text": "Let me "},
        "search.",  # bare string block -> merged into the leading segment
        {"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"},
        {"type": "web_search_tool_result", "tool_use_id": "srvtu_1"},  # skipped
        {"type": "text", "text": "Done."},
    ]

    segments = list(split_content_around_server_tools(content, "msg-1"))

    assert segments == [
        AgentTextSegment(key="msg-1", text="Let me search.", index=0),
        ServerToolBoundary(block=content[2], index=0),
        AgentTextSegment(key="msg-1:seg1", text="Done.", index=1),
    ]


def test_split_content_around_server_tools_skips_empty_segments():
    # Adjacent tool calls (no text between) yield no empty AGENT segment.
    content = [
        {"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"},
        {"type": "server_tool_use", "id": "srvtu_2", "name": "web_fetch"},
    ]

    segments = list(split_content_around_server_tools(content, "msg-1"))

    assert segments == [
        ServerToolBoundary(block=content[0], index=0),
        ServerToolBoundary(block=content[1], index=1),
    ]
