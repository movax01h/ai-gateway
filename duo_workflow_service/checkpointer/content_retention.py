"""Content retention: what a session exposes of the text it produced.

A registry flow listed in ``METADATA_ONLY_FLOWS`` exposes the shape of the run (which steps and tools ran, their status
and timing) but none of the model or user text; a fixed failure notice keeps its text. The live stream and the audit
collector apply it at their exits, so it is an allowlist: a field not named here is dropped. Saved checkpoints stay
complete so the session can resume; GitLab reduces them when it serves them to anyone but Duo Workflow Service.
"""

from typing import Any, Mapping, Optional

# The only chat-log fields kept per entry.
_KEPT_CHAT_LOG_FIELDS = (
    "message_type",
    "message_sub_type",
    "status",
    "timestamp",
    "correlation_id",
    "component_name",
)

# message_id prefix of a failure notice the service writes itself from fixed text (a NotifiableAgentException
# ui_message, or the generic error message with at most an exception class name), never model, tool or user text.
# Such a notice keeps its text.
# Keep in sync with ee/lib/ai/duo_workflows/content_retention.rb.
ERROR_NOTICE_ID_PREFIX = "error-notice-"


def _is_error_notice(entry: Mapping[str, Any]) -> bool:
    return (
        entry.get("message_type") == "agent"
        and entry.get("status") == "failure"
        and str(entry.get("message_id") or "").startswith(ERROR_NOTICE_ID_PREFIX)
    )


def reduce_ui_chat_log_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the identity, type, status and tool name of a chat-log entry; empty its text unless it is a failure notice.

    The result still has the shape the chat-log readers validate. duo-cli (gitlab-lsp ``ChatLogSchema``) requires
    ``additional_context`` to be present and ``message_id`` to be a string when present. The session page
    (duo-ui ``DuoToolMessage``) requires every ``tool`` entry to carry a ``tool_info`` object.
    """
    reduced: dict[str, Any] = {key: entry.get(key) for key in _KEPT_CHAT_LOG_FIELDS}
    if entry.get("message_id") is not None:
        reduced["message_id"] = entry["message_id"]
    notice = _is_error_notice(entry)
    reduced["content"] = str(entry.get("content") or "") if notice else ""
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
    """Reduce every chat-log entry to its metadata, keeping a fixed failure notice's text.

    See ``reduce_ui_chat_log_entry``.
    """
    return [reduce_ui_chat_log_entry(entry) for entry in entries or []]
