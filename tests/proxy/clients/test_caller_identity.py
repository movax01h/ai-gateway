import pytest
from starlette.datastructures import Headers

from ai_gateway.proxy.clients.caller_identity import (
    CallerIdentity,
    resolve_caller_identity,
)


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        (
            "opencode/1.2.3 gitlab-ai-provider/6.0.1 (darwin; arm64)",
            CallerIdentity("opencode", "1.2.3", "cli"),
        ),
        (
            "gitlab-ai-provider/6.0.1 opencode/1.2.3",
            CallerIdentity("opencode", "1.2.3", "cli"),
        ),
        (
            "gitlab-ai-provider/6.0.1 (darwin; arm64)",
            CallerIdentity("gitlab_ai_provider", "6.0.1", "sdk"),
        ),
        (
            "claude-cli/2.0.5 (external, cli)",
            CallerIdentity("claude_code", "2.0.5", "cli"),
        ),
        (
            "claude-cli/2.0.5 (external, sdk-cli)",
            CallerIdentity("claude_code", "2.0.5", "cli"),
        ),
        (
            "codex_exec/0.46.0 (Mac OS 15; arm64)",
            CallerIdentity("codex", "0.46.0", "cli"),
        ),
        ("pi/1.4.0", CallerIdentity("pi", "1.4.0", "cli")),
        ("kilo/4.1.0", CallerIdentity("kilo", "4.1.0", "cli")),
        ("Anthropic/JS 0.39.0", CallerIdentity("anthropic_sdk", "0.39.0", "sdk")),
        ("OpenAI/JS 5.1.0", CallerIdentity("openai_sdk", "5.1.0", "sdk")),
        ("OpenAI/Python 1.2.3", CallerIdentity("openai_sdk", "1.2.3", "sdk")),
        ("OpenAI/Python", CallerIdentity("openai_sdk", None, "sdk")),
        (
            "AsyncOpenAI/Python 2.54.0",
            CallerIdentity("openai_sdk", "2.54.0", "sdk"),
        ),
        (
            "Anthropic/Python 0.116.0",
            CallerIdentity("anthropic_sdk", "0.116.0", "sdk"),
        ),
        (
            "AsyncAnthropic/Python 0.116.0",
            CallerIdentity("anthropic_sdk", "0.116.0", "sdk"),
        ),
        (
            "AnthropicVertex/Python 0.116.0",
            CallerIdentity("anthropic_sdk", "0.116.0", "sdk"),
        ),
        (
            "AsyncAnthropicVertex/Python 0.116.0",
            CallerIdentity("anthropic_sdk", "0.116.0", "sdk"),
        ),
        ("python-httpx/0.27.0", CallerIdentity("http_library", "0.27.0", "sdk")),
        ("python-httpx", CallerIdentity("http_library", None, "sdk")),
        ("Mozilla/5.0 (X11; Linux x86_64)", CallerIdentity("other", None, "other")),
        ("my-secret-tool/1.0 token=abc", CallerIdentity("other", None, "other")),
        ("opencode/not a version", CallerIdentity("opencode", None, "cli")),
        ("(((", CallerIdentity("other", None, "other")),
        ("", CallerIdentity()),
        ("   ", CallerIdentity()),
        (None, CallerIdentity()),
    ],
)
def test_user_agent(user_agent, expected):
    headers = {} if user_agent is None else {"user-agent": user_agent}

    assert resolve_caller_identity(Headers(headers)) == expected


def test_malformed_input_never_raises():
    headers = {"user-agent": "\x00\xff/" * 1000, "X-Gitlab-Client-Version": 5}

    identity = resolve_caller_identity(headers)

    assert identity.client_name == "other"
    assert identity.client_version is None


def test_explicit_headers_win_over_user_agent():
    headers = {
        "X-Gitlab-Client-Name": "gitlab-ai-provider",
        "X-Gitlab-Client-Version": "6.1.0",
        "X-Gitlab-Client-Type": "sdk",
        "user-agent": "opencode/1.2.3",
    }

    assert resolve_caller_identity(headers) == CallerIdentity(
        "gitlab-ai-provider", "6.1.0", "sdk"
    )


def test_header_version_and_type_apply_to_parsed_name():
    headers = {
        "X-Gitlab-Client-Version": "9.9.9",
        "X-Gitlab-Client-Type": "ide",
        "user-agent": "opencode/1.2.3",
    }

    assert resolve_caller_identity(headers) == CallerIdentity(
        "opencode", "9.9.9", "ide"
    )


def test_explicit_header_values_are_truncated():
    headers = {"X-Gitlab-Client-Name": "a" * 500}

    assert resolve_caller_identity(headers).client_name == "a" * 64


def test_to_event_context_omits_unknown_fields():
    assert CallerIdentity("opencode", None, "cli").to_event_context() == {
        "client_name": "opencode",
        "client_type": "cli",
    }
    assert not CallerIdentity().to_event_context()
