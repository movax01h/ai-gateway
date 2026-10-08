"""Content retention: what a session exposes of the text it produced.

A flow whose config sets ``content_retention: metadata`` exposes the shape of the run (which steps and tools ran, their
status and timing) but none of the model or user text. The live stream and the audit collector apply it at their exits,
so it is an allowlist: a field not named here is dropped. Saved checkpoints stay complete so the session can resume;
GitLab reduces them when it serves them to anyone but Duo Workflow Service.
"""

from typing import Any, Literal, Mapping, Optional

ContentRetention = Literal["full", "metadata"]

METADATA_RETENTION: ContentRetention = "metadata"


def is_metadata_only(content_retention: Optional[str]) -> bool:
    """Whether a flow with this ``content_retention`` keeps only the metadata of its run."""
    return content_retention == METADATA_RETENTION


# The only chat-log fields kept per entry.
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


def reduce_ui_chat_log(
    entries: Optional[list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Reduce every chat-log entry to its metadata; see ``reduce_ui_chat_log_entry``."""
    return [reduce_ui_chat_log_entry(entry) for entry in entries or []]
