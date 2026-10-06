"""Content retention: what a session keeps of the text it produced.

A flow whose config sets ``content_retention: metadata`` keeps the shape of the run (which steps and tools ran, their
status and timing) but none of the model or user text. The checkpointer, the live stream and the audit collector apply
it at their exits, so it is an allowlist: a field not named here is dropped.
"""

from typing import Any, Literal, Mapping, Optional

ContentRetention = Literal["full", "metadata"]

METADATA_RETENTION: ContentRetention = "metadata"


class SessionNotResumableError(Exception):
    """A metadata-retention session was asked to resume; its saved state cannot rebuild the run."""


# The only channels kept, and the only chat-log fields kept per entry.
_KEPT_CHANNELS = ("status", "ui_chat_log")
_KEPT_CHAT_LOG_FIELDS = (
    "message_type",
    "message_sub_type",
    "status",
    "timestamp",
    "correlation_id",
    "component_name",
)


def reduce_ui_chat_log_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the identity, type, status and tool name of a chat-log entry; empty its text.

    The result still has the shape the chat-log readers validate. duo-cli (gitlab-lsp ``ChatLogSchema``) requires
    ``additional_context`` to be present and ``message_id`` to be a string when present. The session page
    (duo-ui ``DuoToolMessage``) requires every ``tool`` entry to carry a ``tool_info`` object.
    """
    reduced: dict[str, Any] = {key: entry.get(key) for key in _KEPT_CHAT_LOG_FIELDS}
    if entry.get("message_id") is not None:
        reduced["message_id"] = entry["message_id"]
    reduced["content"] = ""
    reduced["additional_context"] = None
    tool_info: Optional[Mapping[str, Any]] = entry.get("tool_info")
    if tool_info or entry.get("message_type") == "tool":
        reduced["tool_info"] = {"name": (tool_info or {}).get("name") or "", "args": {}}
    else:
        reduced["tool_info"] = None
    return reduced


def reduce_ui_chat_log(entries: Optional[list]) -> list[dict[str, Any]]:
    return [reduce_ui_chat_log_entry(entry) for entry in entries or []]


def reduce_state_for_metadata_retention(
    channel_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the channel values a metadata-retention session may persist: ``status`` and a reduced chat log."""
    reduced = {
        key: channel_values[key] for key in _KEPT_CHANNELS if key in channel_values
    }
    if "ui_chat_log" in reduced:
        reduced["ui_chat_log"] = reduce_ui_chat_log(reduced["ui_chat_log"])
    return reduced


def reduce_checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep LangGraph's bookkeeping keys; drop ``writes`` and anything else that can carry node output."""
    return {
        key: metadata[key]
        for key in ("source", "step", "parents", "run_id")
        if key in metadata
    }
