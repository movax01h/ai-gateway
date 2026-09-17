"""The client's contract for a ui_chat_log tool card, mirrored for tests.

gitlab-lsp validates every ui_chat_log entry with a zod schema before the CLI or
the IDE renders it (`packages/core/workflow_api/src/ui_chat_log.ts`,
`ToolInfoSchema`), and one failing entry discards the whole checkpoint's chat
log for that client. The schema lives in that repo, so this is a mirror rather
than the source of truth: when it changes there, change it here.

Checked on the serialised form, because that is what crosses the wire.
"""

import json
from typing import Any

from duo_workflow_service.json_encoder.encoder import CustomEncoder

_TOOL_RESPONSE_STRING_FIELDS = ("content", "type", "name", "tool_call_id", "status")
_TOOL_RESPONSE_OBJECT_FIELDS = ("additional_kwargs", "response_metadata")


def assert_client_valid_tool_info(tool_info: Any) -> None:
    """Assert *tool_info* would pass the client's `ToolInfoSchema`."""
    data = json.loads(json.dumps(tool_info, cls=CustomEncoder))

    assert isinstance(data.get("name"), str), "tool_info.name must be a string"
    assert isinstance(data.get("args"), dict), "tool_info.args must be an object"
    if "suggested_patterns" in data:
        patterns = data["suggested_patterns"]
        assert isinstance(patterns, list), (
            f"tool_info.suggested_patterns must be an array, got {type(patterns).__name__}"
        )
        assert all(isinstance(p, str) for p in patterns)
    if "tool_response" not in data:
        return

    response = data["tool_response"]
    if isinstance(response, str):
        return

    # ToolResponseSchema: a serialised ToolMessage.
    assert isinstance(response, dict), (
        "tool_response must be a string or a serialised ToolMessage, "
        f"got {type(response).__name__}: {response!r}"
    )
    for key in _TOOL_RESPONSE_STRING_FIELDS:
        assert isinstance(response.get(key), str), (
            f"tool_response.{key} must be a string, got {response.get(key)!r}"
        )
    for key in _TOOL_RESPONSE_OBJECT_FIELDS:
        assert isinstance(response.get(key), dict), (
            f"tool_response.{key} must be an object"
        )
    # `id` is nullable but not optional in the schema.
    assert "id" in response and (
        response["id"] is None or isinstance(response["id"], str)
    )
