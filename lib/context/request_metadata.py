"""Request metadata context variables shared between services.

These context variables track request-level metadata for instrumentation, logging, and metrics collection.
"""

from contextvars import ContextVar
from enum import Enum
from typing import Optional

import structlog

from lib.language_server import LanguageServerVersion

log = structlog.stdlib.get_logger("request_metadata")

# Context variables for request metadata
client_type: ContextVar[Optional[str]] = ContextVar("client_type", default=None)
gitlab_realm: ContextVar[Optional[str]] = ContextVar("gitlab_realm", default=None)
gitlab_version: ContextVar[Optional[str]] = ContextVar("gitlab_version", default=None)
language_server_version: ContextVar[Optional[LanguageServerVersion]] = ContextVar(
    "language_server_version", default=None
)
is_gitlab_team_member: ContextVar[Optional[bool]] = ContextVar(
    "is_gitlab_team_member", default=None
)

# client_capabilities is used to make backwards compatible changes to our
# communication protocol. This is needed usually when we're adding new
# protobuf fields and changing behaviour of the gRPC communication in
# non-backwards compatible ways.
#
# It is passed through the original client (e.g.
# VSCode extension, language server, Rails) via
# protobuf metadata and set for each request.
client_capabilities: ContextVar[set[str]] = ContextVar(
    "client_capabilities", default=set()
)


def _language_server_version_label() -> str:
    lsp_version = language_server_version.get()
    if lsp_version:
        return str(lsp_version.version)
    return "unknown"


def _gitlab_version_label() -> str:
    # pylint: disable=import-outside-toplevel
    from packaging.version import InvalidVersion, Version

    try:
        gl_version = Version(gitlab_version.get() or "")
        return str(gl_version)
    except (InvalidVersion, TypeError):
        return "unknown"


def _client_type_label() -> str:
    client_type_value = client_type.get()
    if client_type_value:
        return str(client_type_value)
    return "unknown"


def _gitlab_realm_label() -> str:
    gitlab_realm_value = gitlab_realm.get()
    if gitlab_realm_value:
        return str(gitlab_realm_value)
    return "unknown"


def _is_gitlab_team_member_label() -> str:
    value = is_gitlab_team_member.get()
    if value is None:
        return "unknown"
    return "yes" if value else "no"


_METADATA_LABEL_GETTERS = {
    "lsp_version": _language_server_version_label,
    "gitlab_version": _gitlab_version_label,
    "client_type": _client_type_label,
    "gitlab_realm": _gitlab_realm_label,
    "is_gitlab_team_member": _is_gitlab_team_member_label,
}

METADATA_LABELS = list(_METADATA_LABEL_GETTERS.keys())


def build_metadata_labels() -> dict[str, str]:
    """Build a dictionary of metadata labels from current context values."""
    return {key: getter() for key, getter in _METADATA_LABEL_GETTERS.items()}


class LLMFinishReason(str, Enum):
    """Enum representing possible finish reasons from LLM responses."""

    STOP = "stop"
    END_TURN = "end_turn"
    LENGTH = "length"  # Hit max_tokens limit
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_CALLS = "tool_calls"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"

    @classmethod
    def values(cls):
        """Return all enum values as a list."""
        return [e.value for e in cls]

    @classmethod
    def truncation_values(cls) -> list["LLMFinishReason"]:
        """Return finish reasons that indicate the response was truncated by the output token limit.

        These reasons mean the model had more to say but was cut off, making the response incomplete. They are distinct
        from content-filter stops or context-window errors.
        """
        return [cls.LENGTH, cls.MAX_TOKENS]

    @classmethod
    def abnormal_values(cls) -> list["LLMFinishReason"]:
        """Return abnormal finish reasons."""
        return [
            cls.LENGTH,
            cls.CONTENT_FILTER,
            cls.GUARDRAIL_INTERVENED,
            cls.MAX_TOKENS,
            cls.MODEL_CONTEXT_WINDOW_EXCEEDED,
        ]


def extract_finish_reason(metadata: dict) -> Optional[LLMFinishReason]:
    """Extract the finish reason from LangChain response_metadata.

    Different providers use different keys:
    - Anthropic: "stop_reason"
    - OpenAI / Vertex / Mistral: "finish_reason"
    - Bedrock Converse: "stopReason"
    """
    for key in ("stop_reason", "finish_reason", "stopReason"):
        if value := metadata.get(key):
            try:
                return LLMFinishReason(value)
            except ValueError:
                log.warning(
                    "Unknown LLM finish reason encountered",
                    finish_reason=value,
                    metadata_key=key,
                )
                return None
    return None
