import json

import pytest

from duo_workflow_service.checkpointer.content_retention import (
    reduce_ui_chat_log,
    reduce_ui_chat_log_entry,
)

SECRET = "SECRET-MODEL-TEXT"

KEPT_ENTRY_FIELDS = {
    "message_id": "msg-1",
    "message_type": "tool",
    "message_sub_type": "read_file",
    "status": "success",
    "timestamp": "2026-10-06T00:00:00+00:00",
    "correlation_id": "corr-1",
    "component_name": "triage",
}


def _chat_entry(message_id="msg-1"):
    return {
        **KEPT_ENTRY_FIELDS,
        "message_id": message_id,
        "content": SECRET,
        "tool_info": {
            "name": "read_file",
            "args": {"file_path": SECRET},
            "tool_response": SECRET,
        },
        "additional_context": [{"content": SECRET}],
        "subsession_id": SECRET,
    }


def _reduced_entry(message_id="msg-1"):
    return {
        **KEPT_ENTRY_FIELDS,
        "message_id": message_id,
        "content": "",
        "additional_context": None,
        "tool_info": {"name": "read_file", "args": {}},
    }


def _assert_readers_accept(entry):
    """Mirror the checks the chat-log readers apply to each entry.

    duo-cli: gitlab-lsp packages/core/workflow_api/src/ui_chat_log.ts ``ChatLogSchema`` (zod 4, so a
    ``z.unknown()`` key must be present). Session page: duo-ui ``DuoToolMessage`` ``message`` prop validator
    ``hasToolInfo``.
    """
    nullable_str = (str, type(None))
    assert isinstance(entry["content"], str)
    assert isinstance(entry["timestamp"], str)
    assert isinstance(entry["message_sub_type"], nullable_str)
    assert isinstance(entry["status"], nullable_str)
    assert isinstance(entry["correlation_id"], nullable_str)
    assert "additional_context" in entry
    if "message_id" in entry:
        assert isinstance(entry["message_id"], str)
    assert isinstance(entry.get("component_name"), nullable_str)

    tool_info = entry["tool_info"]
    if tool_info is not None:
        assert isinstance(tool_info["name"], str)
        assert isinstance(tool_info["args"], dict)
    if entry["message_type"] in ("user", "agent"):
        assert tool_info is None
    elif entry["message_type"] == "request":
        assert tool_info is not None or entry["message_sub_type"] == "approval"
    elif entry["message_type"] == "tool":
        assert isinstance(tool_info, dict)  # DuoToolMessage hasToolInfo
    else:
        pytest.fail(f"unexpected message_type {entry['message_type']}")


class TestReduction:
    def test_reduces_every_chat_log_entry(self):
        assert reduce_ui_chat_log([_chat_entry("msg-0"), _chat_entry("msg-1")]) == [
            _reduced_entry("msg-0"),
            _reduced_entry("msg-1"),
        ]
        assert not reduce_ui_chat_log(None)

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"message_type": "tool", "tool_info": None, "message_sub_type": None},
            {"message_type": "agent", "tool_info": None, "message_sub_type": None},
            {"message_type": "user", "tool_info": None, "message_sub_type": None},
            {"message_type": "request", "message_sub_type": None},
            {
                "message_type": "request",
                "tool_info": None,
                "message_sub_type": "approval",
            },
            {"message_id": None, "correlation_id": None, "status": None},
        ],
    )
    def test_reduced_entry_passes_the_chat_log_readers(self, overrides):
        reduced = reduce_ui_chat_log_entry({**_chat_entry(), **overrides})

        _assert_readers_accept(reduced)
        assert SECRET not in json.dumps(reduced)

    def test_message_and_approval_entries_keep_no_tool_info(self):
        for message_type, sub_type in (("agent", None), ("request", "approval")):
            entry = {
                **_chat_entry(),
                "message_type": message_type,
                "message_sub_type": sub_type,
                "tool_info": None,
            }

            assert reduce_ui_chat_log_entry(entry)["tool_info"] is None

    def test_tool_entry_without_tool_info_gets_an_empty_one(self):
        entry = {**_chat_entry(), "tool_info": None}

        assert reduce_ui_chat_log_entry(entry)["tool_info"] == {"name": "", "args": {}}

    def test_missing_message_id_is_omitted_not_null(self):
        entry = {**_chat_entry(), "message_id": None}

        assert "message_id" not in reduce_ui_chat_log_entry(entry)
