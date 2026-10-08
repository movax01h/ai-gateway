"""Fixtures for the litellm bump checks.

These tests run litellm's real request-building and response-parsing code. Only the last step, the HTTP transport, is
replaced. No request leaves the process and no provider credentials are needed.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import httpx
import litellm
import pytest

ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 5},
}

OPENAI_CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}

OPENAI_TEXT = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}

GEMINI = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "hi"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 3,
        "candidatesTokenCount": 5,
        "totalTokenCount": 8,
    },
}

BEDROCK_CONVERSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 5, "totalTokens": 8},
    "metrics": {"latencyMs": 1},
}


@dataclass
class CapturedRequest:
    method: str
    url: httpx.URL
    headers: httpx.Headers
    body: Any


@dataclass
class FakeProviders:
    """Records every outgoing request and answers from canned responses keyed by URL substring."""

    requests: list[CapturedRequest] = field(default_factory=list)
    responses: dict[str, Callable[[], dict]] = field(default_factory=dict)

    def respond(self, url_part: str, payload: dict) -> None:
        self.responses[url_part] = lambda: payload

    def handle(self, request: httpx.Request) -> httpx.Response:
        content = request.content
        try:
            body = json.loads(content) if content else None
        except ValueError:
            body = content
        self.requests.append(
            CapturedRequest(request.method, request.url, request.headers, body)
        )
        for url_part, make_payload in self.responses.items():
            if url_part in str(request.url):
                return httpx.Response(200, json=make_payload(), request=request)
        raise AssertionError(
            f"Unexpected outgoing request: {request.method} {request.url}"
        )

    @property
    def last(self) -> CapturedRequest:
        assert self.requests, "litellm sent no HTTP request"
        return self.requests[-1]


@pytest.fixture(name="providers")
def providers_fixture(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeProviders]:
    fake = FakeProviders()

    async def handle_async(_self, request: httpx.Request) -> httpx.Response:
        return fake.handle(request)

    def handle_sync(_self, request: httpx.Request) -> httpx.Response:
        return fake.handle(request)

    # litellm can use an aiohttp transport that bypasses httpx. Turn it off and drop clients cached with it.
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_sync)

    yield fake

    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(name="fake_aws")
def fake_aws_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(name="fake_vertex_auth")
def fake_vertex_auth_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the Google OAuth token exchange, which litellm does outside httpx.

    This replaces two private litellm methods. If litellm renames them, this fixture fails with an AttributeError. That
    is a signal to review the Vertex auth path, not a regression in the service.
    """
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    def project_of(args: tuple, kwargs: dict) -> str:
        project_id = kwargs.get("project_id", args[1] if len(args) > 1 else None)
        return project_id or "fake-project"

    async def token_async(_self, *args, **kwargs):
        return "fake-token", project_of(args, kwargs)

    def token_sync(_self, *args, **kwargs):
        return "fake-token", project_of(args, kwargs)

    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", token_async)
    monkeypatch.setattr(VertexBase, "_ensure_access_token", token_sync)


@pytest.fixture(name="update_schemas")
def update_schemas_fixture() -> bool:
    """True when the schema files should be rewritten instead of compared.

    Set UPDATE_LITELLM_SCHEMAS=1.
    """
    return bool(os.environ.get("UPDATE_LITELLM_SCHEMAS"))  # pylint: disable=direct-environment-variable-reference
