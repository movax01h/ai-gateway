"""Identify the tool that calls the proxy routes, for token-usage events.

Proxy callers (OpenCode, Claude Code, raw SDKs, ...) don't send the `X-Gitlab-Client-*` headers that the standard
context is normally filled from, but they do send a `User-Agent`. The resolver below maps it to a small, bounded set of
tool names, so the raw string (which can carry OS details or free text) never reaches the event.

The `User-Agent` is set by the client. Treat the result as an indicator only, never for enforcement or billing.
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ai_gateway.api.middleware.headers import (
    X_GITLAB_CLIENT_NAME,
    X_GITLAB_CLIENT_TYPE,
    X_GITLAB_CLIENT_VERSION,
)

__all__ = [
    "CLIENT_NAME_OTHER",
    "CallerIdentity",
    "resolve_caller_identity",
]

CLIENT_NAME_OTHER = "other"

_CLIENT_TYPE_CLI = "cli"
_CLIENT_TYPE_SDK = "sdk"
_CLIENT_TYPE_OTHER = "other"

# Hard limits so client-controlled input can't inflate the event.
_MAX_USER_AGENT_LENGTH = 512
_MAX_HEADER_VALUE_LENGTH = 64
_MAX_VERSION_LENGTH = 32

_VERSION_PATTERN = re.compile(r"^v?\d+(?:[.\-+_][0-9A-Za-z]+)*$")
_COMMENT_PATTERN = re.compile(r"\([^)]*\)")


@dataclass(frozen=True)
class _KnownTool:
    name: str
    client_type: str


# Products that wrap another client. The first non-wrapper product decides the name, so
# `opencode/1.2.3 gitlab-ai-provider/6.0.0 (darwin; arm64)` is OpenCode. A wrapper on its own is reported as itself.
_WRAPPER_TOOLS: dict[str, _KnownTool] = {
    "gitlab-ai-provider": _KnownTool("gitlab_ai_provider", _CLIENT_TYPE_SDK),
}

# Keyed by the lowercase product token of the User-Agent. Keep the set of `name` values small and stable: it is the
# cardinality of `client_name` in the warehouse.
_KNOWN_TOOLS: dict[str, _KnownTool] = {
    "opencode": _KnownTool("opencode", _CLIENT_TYPE_CLI),
    "claude-cli": _KnownTool("claude_code", _CLIENT_TYPE_CLI),
    "claude-code": _KnownTool("claude_code", _CLIENT_TYPE_CLI),
    "codex": _KnownTool("codex", _CLIENT_TYPE_CLI),
    "codex_exec": _KnownTool("codex", _CLIENT_TYPE_CLI),
    "codex_cli_rs": _KnownTool("codex", _CLIENT_TYPE_CLI),
    "codex-cli": _KnownTool("codex", _CLIENT_TYPE_CLI),
    "pi": _KnownTool("pi", _CLIENT_TYPE_CLI),
    "kilo": _KnownTool("kilo", _CLIENT_TYPE_CLI),
    "kilocode": _KnownTool("kilo", _CLIENT_TYPE_CLI),
    # The Python SDKs send the client class name, so each client class is its own product token.
    "anthropic": _KnownTool("anthropic_sdk", _CLIENT_TYPE_SDK),
    "asyncanthropic": _KnownTool("anthropic_sdk", _CLIENT_TYPE_SDK),
    "anthropicvertex": _KnownTool("anthropic_sdk", _CLIENT_TYPE_SDK),
    "asyncanthropicvertex": _KnownTool("anthropic_sdk", _CLIENT_TYPE_SDK),
    "openai": _KnownTool("openai_sdk", _CLIENT_TYPE_SDK),
    "asyncopenai": _KnownTool("openai_sdk", _CLIENT_TYPE_SDK),
    "python-httpx": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "python-requests": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "python-urllib3": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "aiohttp": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "axios": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "node-fetch": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "undici": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "okhttp": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "go-http-client": _KnownTool("http_library", _CLIENT_TYPE_SDK),
    "curl": _KnownTool("http_library", _CLIENT_TYPE_SDK),
}


@dataclass(frozen=True)
class CallerIdentity:
    """The calling client as recorded on the standard context."""

    client_name: str | None = None
    client_version: str | None = None
    client_type: str | None = None

    def to_event_context(self) -> dict[str, str]:
        """Return the non-null fields, so they only override what is known."""
        return {key: value for key, value in vars(self).items() if value is not None}


def _clean_version(value: str | None) -> str | None:
    if not value:
        return None

    value = value[:_MAX_VERSION_LENGTH]
    return value if _VERSION_PATTERN.match(value) else None


def _header_value(headers: Mapping[str, Any], name: str) -> str | None:
    value = headers.get(name)
    if not isinstance(value, str):
        return None

    return value.strip()[:_MAX_HEADER_VALUE_LENGTH] or None


def _identity_from_user_agent(user_agent: str | None) -> CallerIdentity:
    if not user_agent or not user_agent.strip():
        return CallerIdentity()

    tokens = _COMMENT_PATTERN.sub(" ", user_agent[:_MAX_USER_AGENT_LENGTH]).split()
    wrapper: CallerIdentity | None = None

    for index, token in enumerate(tokens):
        product, _, version = token.partition("/")
        key = product.lower()

        if (wrapper_tool := _WRAPPER_TOOLS.get(key)) is not None:
            wrapper = wrapper or CallerIdentity(
                client_name=wrapper_tool.name,
                client_version=_clean_version(version),
                client_type=wrapper_tool.client_type,
            )
            continue

        tool = _KNOWN_TOOLS.get(key)
        if tool is None:
            # The first product decides, so a free-text suffix can't change the result.
            break

        parsed_version = _clean_version(version)
        if parsed_version is None and tool.client_type == _CLIENT_TYPE_SDK:
            # SDKs report `Anthropic/JS 0.39.0`: the version follows the language.
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            parsed_version = _clean_version(next_token)

        return CallerIdentity(
            client_name=tool.name,
            client_version=parsed_version,
            client_type=tool.client_type,
        )

    return wrapper or CallerIdentity(
        client_name=CLIENT_NAME_OTHER, client_type=_CLIENT_TYPE_OTHER
    )


def resolve_caller_identity(headers: Mapping[str, Any]) -> CallerIdentity:
    """Resolve the calling client of a proxy request.

    An explicit `X-Gitlab-Client-Name` header wins, and its sibling headers are used as sent without mixing in values
    parsed from the User-Agent. Otherwise the User-Agent is mapped to a known tool, `other` when it isn't known, and
    nothing when it is missing. Never raises.
    """
    header_name = _header_value(headers, X_GITLAB_CLIENT_NAME)
    header_version = _header_value(headers, X_GITLAB_CLIENT_VERSION)
    header_type = _header_value(headers, X_GITLAB_CLIENT_TYPE)

    if header_name:
        return CallerIdentity(
            client_name=header_name,
            client_version=header_version,
            client_type=header_type,
        )

    parsed = _identity_from_user_agent(headers.get("user-agent"))

    return CallerIdentity(
        client_name=parsed.client_name,
        client_version=header_version or parsed.client_version,
        client_type=header_type or parsed.client_type,
    )
