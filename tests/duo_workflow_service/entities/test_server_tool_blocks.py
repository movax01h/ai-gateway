import pytest
from structlog.testing import capture_logs

from duo_workflow_service.entities.server_tool_blocks import (
    AgentTextSegment,
    ServerToolBoundary,
    ServerToolResults,
    _is_anthropic_server_tool_result_block,
    _is_anthropic_server_tool_use_block,
    split_content_around_server_tools,
    warn_unmatched_server_tool_results,
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
    assert _is_anthropic_server_tool_use_block(block) is expected


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
    assert _is_anthropic_server_tool_result_block(block) is expected


def test_build_ui_chat_log_anthropic_pending_without_result():
    use_block = {
        "type": "server_tool_use",
        "id": "srvtu_1",
        "name": "web_search",
        "input": {"query": "latest AI research"},
    }

    entry = ServerToolResults([use_block]).build_ui_chat_log(use_block)

    assert entry["message_type"] == MessageTypeEnum.TOOL
    assert entry["message_sub_type"] == "web_search"
    assert entry["status"] == ToolStatus.PENDING
    assert entry["content"] == "Using web_search"
    assert entry["message_id"] == "srvtu_1"
    assert entry["tool_info"]["name"] == "web_search"
    assert entry["tool_info"]["args"] == {"query": "latest AI research"}
    # No result yet -> no tool_response key.
    assert "tool_response" not in entry["tool_info"]


def test_build_ui_chat_log_anthropic_success_with_result():
    result_content = [{"type": "web_search_result", "url": "https://x", "title": "X"}]
    content = [
        {
            "type": "server_tool_use",
            "id": "srvtu_1",
            "name": "web_search",
            "input": {"query": "gitlab duo"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtu_1",
            "content": result_content,
        },
    ]

    entry = ServerToolResults(content).build_ui_chat_log(
        content[0], component_name="chat"
    )

    assert entry["status"] == ToolStatus.SUCCESS
    assert entry["message_id"] == "srvtu_1"
    assert entry["component_name"] == "chat"
    assert entry["tool_info"]["tool_response"] == result_content


def test_build_ui_chat_log_anthropic_defaults_for_missing_fields():
    entry = ServerToolResults([]).build_ui_chat_log({"type": "server_tool_use"})

    assert entry["message_sub_type"] == "server_tool"
    assert entry["tool_info"]["name"] == "server_tool"
    assert entry["tool_info"]["args"] == {}
    assert entry["message_id"] is None


_PAIRED = [
    {"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"},
    {"type": "web_search_tool_result", "tool_use_id": "srvtu_1", "content": []},
]
_EMPTY_SEARCH = [
    {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "sources": []},
    }
]


@pytest.mark.parametrize(
    "content",
    [pytest.param(_PAIRED, id="anthropic"), pytest.param(_EMPTY_SEARCH, id="openai")],
)
def test_build_ui_chat_log_omits_an_empty_tool_response(content):
    entry = ServerToolResults(content).build_ui_chat_log(content[0])

    assert entry["status"] == ToolStatus.SUCCESS
    assert "tool_response" not in entry["tool_info"]


def test_build_ui_chat_log_redacts_secrets_in_result():
    leaked_token = "gh" + "p_" + "1234567890abcdefghijklmnopqrstuvwxyz"
    content = [
        {"type": "server_tool_use", "id": "srvtu_1", "name": "web_search"},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtu_1",
            "content": [
                {"type": "web_search_result", "snippet": f"token {leaked_token}"}
            ],
        },
    ]

    entry = ServerToolResults(content).build_ui_chat_log(content[0])

    snippet = entry["tool_info"]["tool_response"][0]["snippet"]
    assert leaked_token not in snippet
    assert "[REDACTED]" in snippet


@pytest.mark.parametrize(
    ("content", "expected_events"),
    [
        pytest.param(_PAIRED, [], id="a-paired-result-is-quiet"),
        pytest.param(
            [
                _PAIRED[0],
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtu_other",
                    "content": [],
                },
            ],
            [
                {
                    "event": "Server tool result has no matching server tool use block",
                    "log_level": "warning",
                    "tool_use_id": "srvtu_other",
                    "block_type": "web_search_tool_result",
                }
            ],
            id="an-unmatched-anthropic-result",
        ),
        pytest.param(
            [{"type": "web_search_call", "id": "ws_1", "action": {"type": "search"}}],
            [],
            id="a-mapped-openai-action-is-quiet",
        ),
        pytest.param(
            [{"type": "web_search_call", "id": "ws_1", "action": {"type": "unknown"}}],
            [
                {
                    "event": "Server tool call has an unmapped action type; card falls back to web_search",
                    "log_level": "warning",
                    "action_type": "unknown",
                }
            ],
            id="an-unmapped-openai-action",
        ),
    ],
)
def test_warn_unmatched_server_tool_results(content, expected_events):
    with capture_logs() as logs:
        warn_unmatched_server_tool_results(content)

    assert logs == expected_events


def test_split_content_around_server_tools_splits_on_openai_calls():
    content = [
        {"type": "reasoning", "id": "rs_1", "summary": []},  # skipped
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        {"type": "text", "text": "Done."},
    ]

    segments = list(split_content_around_server_tools(content, "resp_1"))

    assert segments == [
        ServerToolBoundary(block=content[1], index=0),
        AgentTextSegment(key="resp_1:seg1", text="Done.", index=1),
    ]


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
