import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_workflow_params import WorkflowConfigFetchError
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.session_context import (
    MAX_TOOL_RESPONSE_CHARS,
    MAX_UI_CHAT_LOG_ENTRIES,
    GetSessionContext,
    GetSessionContextInput,
)


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    return AsyncMock()


@pytest.fixture(name="tool")
def tool_fixture(gitlab_client):
    return GetSessionContext(
        metadata={
            "gitlab_client": gitlab_client,
            "gitlab_host": "gitlab.example.com",
            "workflow_id": "555",
        }
    )


def _make_checkpoint(channel_values: dict) -> dict:
    return {
        "checkpoint": {"channel_values": channel_values},
        "metadata": {"step": 1, "source": "loop", "writes": {}, "parents": {}},
    }


def _make_response(body) -> GitLabHttpResponse:
    return GitLabHttpResponse(status_code=200, body=body)


def _make_workflow_record(**fields) -> dict:
    record = {
        "id": 1,
        "title": None,
        "summary": None,
        "status": None,
        "workflow_definition": None,
    }
    record.update(fields)
    return record


def _route_aget(checkpoints=None, *, checkpoints_response=None, record=None):
    """Route the tool's up-to-two GET calls (workflow record, checkpoints) to the appropriate mocked response."""
    if record is None:
        record = _make_workflow_record()

    async def _aget(path, parse_json=True, **_kwargs):
        if path.endswith("/checkpoints?per_page=1"):
            if checkpoints_response is not None:
                return checkpoints_response
            return _make_response(checkpoints)
        return _make_response(record)

    return _aget


class TestGetSessionContextFormatDisplayMessage:
    def test_returns_session_id_label(self, tool):
        args = GetSessionContextInput(session_id=123)
        assert tool.format_display_message(args) == "Get context for session 123"


class TestGetSessionContextApiCall:
    @pytest.mark.asyncio
    async def test_calls_correct_endpoint(self, tool, gitlab_client):
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint({})])

        await tool._arun(session_id=42)

        called_paths = [
            call.kwargs.get("path") for call in gitlab_client.aget.call_args_list
        ]
        assert (
            "/api/v4/ai/duo_workflows/workflows/42/checkpoints?per_page=1"
            in called_paths
        )
        assert "/api/v4/ai/duo_workflows/workflows/42" in called_paths

    @pytest.mark.asyncio
    async def test_empty_checkpoint_list_raises(self, tool, gitlab_client):
        gitlab_client.aget.side_effect = _route_aget([])

        with pytest.raises(
            ToolException, match="Unable to find checkpoint for this session"
        ):
            await tool._arun(session_id=123)

    @pytest.mark.asyncio
    async def test_api_error_raises(self, tool, gitlab_client):
        gitlab_client.aget.side_effect = _route_aget(
            checkpoints_response=GitLabHttpResponse(
                status_code=404, body={"message": "not found"}
            )
        )

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(session_id=123)

    @pytest.mark.asyncio
    async def test_client_exception_propagates(self, tool, gitlab_client):
        async def _aget(path, parse_json=True, **_kwargs):
            if path.endswith("/checkpoints?per_page=1"):
                raise Exception("Connection error")
            return _make_response(_make_workflow_record())

        gitlab_client.aget.side_effect = _aget

        with pytest.raises(Exception, match="Connection error"):
            await tool._arun(session_id=123)


class TestGetSessionContextHappyPath:
    @pytest.mark.asyncio
    async def test_returns_status_goal_summary_and_activity(self, tool, gitlab_client):
        channel_values = {
            "status": "finished",
            "goal": "Fix the login bug",
            "ui_chat_log": [
                {
                    "message_type": "user",
                    "content": "Please fix the login bug",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "message_type": "agent",
                    "content": "I have fixed the login bug by patching auth.py",
                    "timestamp": "2024-01-01T00:01:00Z",
                },
            ],
        }
        gitlab_client.aget.side_effect = _route_aget(
            [_make_checkpoint(channel_values)],
            record=_make_workflow_record(
                title="Fix login",
                status="finished",
                summary="Fixed on error",
            ),
        )

        result = json.loads(await tool._arun(session_id=99))

        assert result["session_id"] == 99
        # Record-backed fields.
        assert result["title"] == "Fix login"
        assert result["status"] == "finished"
        assert result["summary"] == "Fixed on error"
        # Checkpoint-backed fields.
        assert result["goal"] == "Fix the login bug"
        assert (
            result["last_message"] == "I have fixed the login bug by patching auth.py"
        )
        assert len(result["recent_activity"]) == 2

    @pytest.mark.asyncio
    async def test_activity_entries_have_expected_fields(self, tool, gitlab_client):
        channel_values = {
            "status": "finished",
            "goal": "Do something",
            "ui_chat_log": [
                {
                    "message_type": "agent",
                    "content": "Done",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ],
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        entry = result["recent_activity"][0]

        assert entry["message_type"] == "agent"
        assert entry["content"] == "Done"
        assert entry["timestamp"] == "2024-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_tool_entries_include_tool_info(self, tool, gitlab_client):
        channel_values = {
            "status": "finished",
            "goal": "Read a file",
            "ui_chat_log": [
                {
                    "message_type": "tool",
                    "content": "read_file result",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "tool_info": {
                        "name": "read_file",
                        "args": {"path": "auth.py"},
                        "tool_response": "def login(): pass",
                    },
                }
            ],
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        entry = result["recent_activity"][0]

        assert entry["tool_info"]["name"] == "read_file"
        assert entry["tool_info"]["args"] == {"path": "auth.py"}
        assert entry["tool_info"]["tool_response"] == "def login(): pass"


class TestGetSessionContextTruncation:
    @pytest.mark.asyncio
    async def test_tool_response_truncated_when_over_limit(self, tool, gitlab_client):
        extra_chars = 100
        long_response = "x" * (MAX_TOOL_RESPONSE_CHARS + extra_chars)
        channel_values = {
            "ui_chat_log": [
                {
                    "message_type": "tool",
                    "content": "tool output",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "tool_info": {
                        "name": "read_file",
                        "args": {},
                        "tool_response": long_response,
                    },
                }
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        tool_response = result["recent_activity"][0]["tool_info"]["tool_response"]

        # Truncated to the limit, with an inline marker appended (mirrors
        # mr_discussions.py's _truncate_note_body convention) so the LLM doesn't
        # mistake a cut-off response for the complete output.
        assert tool_response.startswith("x" * MAX_TOOL_RESPONSE_CHARS)
        assert f"<TRUNCATED: {extra_chars} CHARACTERS DROPPED DUE TO SIZE LIMIT>" in (
            tool_response
        )

    @pytest.mark.asyncio
    async def test_tool_response_not_truncated_when_under_limit(
        self, tool, gitlab_client
    ):
        short_response = "x" * (MAX_TOOL_RESPONSE_CHARS - 10)
        channel_values = {
            "ui_chat_log": [
                {
                    "message_type": "tool",
                    "content": "tool output",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "tool_info": {
                        "name": "read_file",
                        "args": {},
                        "tool_response": short_response,
                    },
                }
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        tool_response = result["recent_activity"][0]["tool_info"]["tool_response"]

        assert tool_response == short_response
        assert "TRUNCATED" not in tool_response

    @pytest.mark.asyncio
    async def test_agent_content_not_truncated(self, tool, gitlab_client):
        long_content = "word " * 500  # well over any limit
        channel_values = {
            "ui_chat_log": [
                {
                    "message_type": "agent",
                    "content": long_content,
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        assert result["recent_activity"][0]["content"] == long_content

    @pytest.mark.asyncio
    async def test_only_last_n_entries_returned(self, tool, gitlab_client):
        total = MAX_UI_CHAT_LOG_ENTRIES + 5
        entries = [
            {
                "message_type": "agent",
                "content": f"message {i}",
                "timestamp": "2024-01-01T00:00:00Z",
            }
            for i in range(total)
        ]
        channel_values = {"ui_chat_log": entries}
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        activity = result["recent_activity"]

        # +1 for the leading system marker entry that signals omitted entries,
        # mirroring repository_files.py's "[Showing X of Y]" convention.
        assert len(activity) == MAX_UI_CHAT_LOG_ENTRIES + 1
        assert activity[0]["message_type"] == "system"
        assert (
            f"Showing last {MAX_UI_CHAT_LOG_ENTRIES} of {total}"
            in (activity[0]["content"])
        )
        # The remaining entries should be the actual last N entries.
        assert activity[1]["content"] == "message 5"
        assert activity[-1]["content"] == f"message {total - 1}"

    @pytest.mark.asyncio
    async def test_no_marker_entry_when_log_within_limit(self, tool, gitlab_client):
        entries = [
            {
                "message_type": "agent",
                "content": f"message {i}",
                "timestamp": "2024-01-01T00:00:00Z",
            }
            for i in range(MAX_UI_CHAT_LOG_ENTRIES)
        ]
        channel_values = {"ui_chat_log": entries}
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        activity = result["recent_activity"]

        assert len(activity) == MAX_UI_CHAT_LOG_ENTRIES
        assert all(entry["message_type"] != "system" for entry in activity)


class TestGetSessionContextLastMessageExtraction:
    @pytest.mark.asyncio
    async def test_last_message_is_last_agent_message(self, tool, gitlab_client):
        channel_values = {
            "ui_chat_log": [
                {
                    "message_type": "agent",
                    "content": "First agent message",
                    "timestamp": "t",
                },
                {"message_type": "tool", "content": "tool output", "timestamp": "t"},
                {
                    "message_type": "agent",
                    "content": "Final agent message",
                    "timestamp": "t",
                },
                {"message_type": "user", "content": "User follow-up", "timestamp": "t"},
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["last_message"] == "Final agent message"

    @pytest.mark.asyncio
    async def test_last_message_is_none_when_no_agent_messages(
        self, tool, gitlab_client
    ):
        channel_values = {
            "ui_chat_log": [
                {"message_type": "user", "content": "Hello", "timestamp": "t"},
                {"message_type": "tool", "content": "tool output", "timestamp": "t"},
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["last_message"] is None


class TestGetSessionContextMissingData:
    @pytest.mark.asyncio
    async def test_missing_channel_values_returns_nulls(self, tool, gitlab_client):
        checkpoint = {"checkpoint": {}, "metadata": {}}
        gitlab_client.aget.side_effect = _route_aget([checkpoint])

        result = json.loads(await tool._arun(session_id=123))

        assert result["session_id"] == 123
        assert result["status"] is None
        assert result["goal"] is None
        assert result["title"] is None
        assert result["summary"] is None
        assert result["workflow_definition"] is None
        assert result["last_message"] is None
        assert result["recent_activity"] == []

    @pytest.mark.asyncio
    async def test_missing_goal_returns_none(self, tool, gitlab_client):
        channel_values = {"status": "finished", "ui_chat_log": []}
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["goal"] is None

    @pytest.mark.asyncio
    async def test_missing_status_returns_none(self, tool, gitlab_client):
        channel_values = {"goal": "Do something", "ui_chat_log": []}
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["status"] is None

    @pytest.mark.asyncio
    async def test_missing_ui_chat_log_returns_empty_activity(
        self, tool, gitlab_client
    ):
        channel_values = {"status": "finished", "goal": "Do something"}
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["recent_activity"] == []

    @pytest.mark.asyncio
    async def test_non_dict_log_entries_skipped(self, tool, gitlab_client):
        channel_values = {
            "ui_chat_log": [
                "not-a-dict",
                {"message_type": "agent", "content": "valid", "timestamp": "t"},
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert len(result["recent_activity"]) == 1
        assert result["recent_activity"][0]["content"] == "valid"

    @pytest.mark.asyncio
    async def test_goal_read_from_context_dict_for_v1_flows(self, tool, gitlab_client):
        # v1 FlowState stores goal under channel_values["context"]["goal"],
        # not as a top-level channel_values["goal"] field.
        channel_values = {
            "status": "finished",
            "context": {"goal": "Distill AGENTS.md pitfalls"},
            "ui_chat_log": [],
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["goal"] == "Distill AGENTS.md pitfalls"

    @pytest.mark.asyncio
    async def test_top_level_goal_takes_precedence_over_context_dict(
        self, tool, gitlab_client
    ):
        # WorkflowState stores goal at top level; context dict should be ignored
        # when top-level goal is present.
        channel_values = {
            "status": "finished",
            "goal": "Top-level goal",
            "context": {"goal": "Nested goal"},
            "ui_chat_log": [],
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))

        assert result["goal"] == "Top-level goal"

    @pytest.mark.asyncio
    async def test_non_string_tool_response_passed_through(self, tool, gitlab_client):
        channel_values = {
            "ui_chat_log": [
                {
                    "message_type": "tool",
                    "content": "tool output",
                    "timestamp": "t",
                    "tool_info": {
                        "name": "some_tool",
                        "args": {},
                        "tool_response": {"key": "value"},
                    },
                }
            ]
        }
        gitlab_client.aget.side_effect = _route_aget([_make_checkpoint(channel_values)])

        result = json.loads(await tool._arun(session_id=1))
        tool_response = result["recent_activity"][0]["tool_info"]["tool_response"]

        assert tool_response == {"key": "value"}


class TestGetSessionContextWorkflowRecord:
    @pytest.mark.asyncio
    async def test_title_summary_definition_from_record(self, tool, gitlab_client):
        channel_values = {"status": "Completed", "goal": "g", "ui_chat_log": []}
        gitlab_client.aget.side_effect = _route_aget(
            [_make_checkpoint(channel_values)],
            record=_make_workflow_record(
                title="Fix login bug",
                summary="Failed on step 3",
                workflow_definition="software_development/v1",
                status="failed",
            ),
        )

        result = json.loads(await tool._arun(session_id=1))

        assert result["title"] == "Fix login bug"
        assert result["summary"] == "Failed on step 3"
        assert result["workflow_definition"] == "software_development/v1"

    @pytest.mark.asyncio
    async def test_status_prefers_record_over_checkpoint(self, tool, gitlab_client):
        # Checkpoint status is "Completed" but the canonical record status is
        # "finished"; the record value should win.
        channel_values = {"status": "Completed", "ui_chat_log": []}
        gitlab_client.aget.side_effect = _route_aget(
            [_make_checkpoint(channel_values)],
            record=_make_workflow_record(status="finished"),
        )

        result = json.loads(await tool._arun(session_id=1))

        assert result["status"] == "finished"

    @pytest.mark.asyncio
    async def test_status_falls_back_to_checkpoint_when_record_missing(
        self, tool, gitlab_client
    ):
        channel_values = {"status": "Completed", "ui_chat_log": []}
        gitlab_client.aget.side_effect = _route_aget(
            [_make_checkpoint(channel_values)],
            record=_make_workflow_record(status=None),
        )

        result = json.loads(await tool._arun(session_id=1))

        assert result["status"] == "Completed"

    @pytest.mark.asyncio
    async def test_fails_closed_when_record_fetch_returns_error_status(
        self, tool, gitlab_client
    ):
        # A non-2xx workflow-record fetch (e.g. 403 on a session the caller
        # can't access) must halt the tool entirely rather than degrading to
        # a partial, record-field-less response: the checkpoints endpoint
        # must never be reached.
        async def _aget(path, parse_json=True, **_kwargs):
            if path.endswith("/checkpoints?per_page=1"):
                return _make_response([_make_checkpoint({})])
            return GitLabHttpResponse(
                status_code=403, body={"message": "403 Forbidden"}
            )

        gitlab_client.aget.side_effect = _aget

        with pytest.raises(WorkflowConfigFetchError, match="HTTP 403"):
            await tool._arun(session_id=1)

        called_paths = [
            call.kwargs.get("path") for call in gitlab_client.aget.call_args_list
        ]
        assert not any(
            path.endswith("/checkpoints?per_page=1") for path in called_paths
        )

    @pytest.mark.asyncio
    async def test_propagates_when_record_fetch_raises(self, tool, gitlab_client):
        async def _aget(path, parse_json=True, **_kwargs):
            if path.endswith("/checkpoints?per_page=1"):
                return _make_response([_make_checkpoint({})])
            raise Exception("Connection error")

        gitlab_client.aget.side_effect = _aget

        with pytest.raises(Exception, match="Connection error"):
            await tool._arun(session_id=1)

    @pytest.mark.asyncio
    async def test_fails_closed_when_record_response_is_not_a_dict(
        self, tool, gitlab_client
    ):
        # A successful (2xx) but non-dict record response (e.g. malformed or
        # unexpected JSON shape) must halt the tool rather than silently
        # normalizing to an empty record and letting the checkpoints fetch
        # proceed regardless.
        async def _aget(path, parse_json=True, **_kwargs):
            if path.endswith("/checkpoints?per_page=1"):
                return _make_response([_make_checkpoint({})])
            return _make_response([])

        gitlab_client.aget.side_effect = _aget

        with pytest.raises(WorkflowConfigFetchError, match="expected object"):
            await tool._arun(session_id=1)

        called_paths = [
            call.kwargs.get("path") for call in gitlab_client.aget.call_args_list
        ]
        assert not any(
            path.endswith("/checkpoints?per_page=1") for path in called_paths
        )
