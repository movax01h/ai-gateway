"""A fake network for the litellm parameter tests.

litellm builds and validates every request with its real code. This module replaces only the last step. It answers
requests from canned payloads, records everything litellm sends, and blocks real connections.
"""

import asyncio
import json
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Awaitable, TypeVar

import httpx
import litellm
import pytest

from litellm_param_tests import responses


class OutboundConnectionBlocked(BaseException):
    """Raised when something tries to open a real socket.

    This derives from BaseException on purpose. litellm wraps Exception subclasses into retryable errors, and the prompt
    retry would then spend about 40 seconds on each case before reporting the failure.
    """


@dataclass(frozen=True)
class Route:
    """One fake endpoint.

    The host and the path must both match.
    """

    method: str
    host: str
    path: str
    status: int
    payload: Any


@dataclass
class Call:
    method: str
    url: str
    host: str
    path: str
    header_names: list[str]
    body: Any


ROUTES = [
    Route(
        "POST",
        "api.fireworks.ai",
        r"/inference/v1/chat/completions$",
        200,
        responses.OPENAI_CHAT,
    ),
    Route(
        "POST",
        "api.fireworks.ai",
        r"/inference/v1/completions$",
        200,
        responses.OPENAI_TEXT,
    ),
    Route(
        "POST",
        "bedrock-mantle",
        r"/openai/v1/responses$",
        200,
        responses.OPENAI_RESPONSES,
    ),
    Route(
        "POST", "api.openai.com", r"/v1/embeddings$", 200, responses.OPENAI_EMBEDDING
    ),
    Route(
        "POST",
        "aiplatform.googleapis.com",
        r"publishers/google/models/text-embedding-005:predict$",
        200,
        responses.VERTEX_EMBEDDING,
    ),
    Route(
        "POST",
        "aiplatform.googleapis.com",
        r"publishers/anthropic/models/[^/]+:rawPredict$",
        200,
        responses.ANTHROPIC_MESSAGE,
    ),
    Route(
        "POST",
        "aiplatform.googleapis.com",
        r"publishers/mistralai/models/[^/]+:rawPredict$",
        200,
        responses.OPENAI_CHAT,
    ),
    Route("POST", "bedrock-runtime", r"/converse$", 200, responses.BEDROCK_CONVERSE),
    Route("POST", "bedrock-runtime", r"/invoke$", 200, responses.ANTHROPIC_MESSAGE),
    # litellm fetches the Codestral chat template from HuggingFace. A 404 makes it fall back.
    Route("GET", "huggingface.co", r".*", 404, {}),
    Route(
        "POST",
        "fake-selfhosted.invalid",
        r"/v1/chat/completions$",
        200,
        responses.OPENAI_CHAT,
    ),
]


@dataclass
class FakeNet:
    """Everything litellm sent during one test."""

    calls: list[Call] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    socket_attempts: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.calls.clear()
        self.unexpected.clear()
        self.socket_attempts.clear()

    def handle(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = None
        self.calls.append(
            Call(
                method=request.method,
                url=str(request.url),
                host=request.url.host,
                path=request.url.path,
                header_names=sorted(name.lower() for name in request.headers.keys()),
                body=body,
            )
        )

        for route in ROUTES:
            if (
                route.method == request.method
                and route.host in request.url.host
                and re.search(route.path, request.url.path)
            ):
                return httpx.Response(route.status, json=route.payload, request=request)

        # Do not raise here. An exception is retried by the prompt for about 40 seconds. A 400 is not retried.
        self.unexpected.append(f"{request.method} {request.url}")
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": f"unexpected request in litellm_param_tests: {request.url}"
                }
            },
            request=request,
        )

    def matching(self, method: str, host: str, path: str) -> list[Call]:
        """Return the recorded calls whose method, host and path match the given route."""
        return [
            call
            for call in self.calls
            if call.method == method
            and host in call.host
            and re.search(path, call.path)
        ]


def install(mp: pytest.MonkeyPatch, net: FakeNet) -> None:
    """Route every httpx request to the fake and block real sockets."""

    async def handle_async(_self: Any, request: httpx.Request) -> httpx.Response:
        return net.handle(request)

    def handle_sync(_self: Any, request: httpx.Request) -> httpx.Response:
        return net.handle(request)

    def block_connect(_self: Any, address: Any) -> None:
        net.socket_attempts.append(str(address))
        raise OutboundConnectionBlocked(f"blocked outbound connection to {address!r}")

    # litellm can use an aiohttp transport that bypasses httpx. Turn it off and drop clients cached with it.
    mp.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    mp.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    mp.setattr(httpx.HTTPTransport, "handle_request", handle_sync)
    mp.setattr(socket.socket, "connect", block_connect)


T = TypeVar("T")


async def run_guarded(awaitable: Awaitable[T], net: FakeNet, timeout: float) -> T:
    """Await a case, and stop it the moment something tries to open a real connection.

    Libraries react to a blocked socket in different ways, and some leave the event loop waiting. Watching the guard
    from outside makes the case fail at once, with the address, whatever the library does.
    """
    task = asyncio.ensure_future(awaitable)
    deadline = asyncio.get_running_loop().time() + timeout

    while not task.done():
        if net.socket_attempts:
            task.cancel()
            raise OutboundConnectionBlocked(
                f"blocked outbound connection to {net.socket_attempts}"
            )
        if asyncio.get_running_loop().time() > deadline:
            task.cancel()
            raise asyncio.TimeoutError(f"no result after {timeout} seconds")
        await asyncio.wait({task}, timeout=0.05)

    try:
        return task.result()
    except BaseException as error:
        if net.socket_attempts:
            raise OutboundConnectionBlocked(
                f"blocked outbound connection to {net.socket_attempts}"
            ) from error
        raise
