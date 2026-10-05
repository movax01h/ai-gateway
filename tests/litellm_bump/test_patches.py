"""Check that every place the repo patches, assigns over, or reads from litellm internals still works.

Assigning to a renamed attribute does not raise in Python. It creates a new attribute that litellm never reads, so the
patch silently stops working. Each test here checks both sides: litellm still uses the name we patch, and our patch
behaves as intended.
"""

import asyncio
import inspect

import httpx
import litellm
import pytest
from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.pass_through_endpoints.llm_provider_handlers import (
    openai_passthrough_logging_handler,
)

import ai_gateway.proxy.clients  # noqa: F401  # pylint: disable=unused-import  # applies the patches under test
from ai_gateway.vendor.langchain_litellm.litellm import _create_usage_metadata
from tests.litellm_bump.conftest import OPENAI_CHAT, FakeProviders

MESSAGES = [{"role": "user", "content": "hello"}]


def test_pass_through_route_patches_apply_and_are_still_called_by_litellm():
    helpers = pass_through_endpoints.InitPassThroughEndpointHelpers

    assert helpers.is_registered_pass_through_route("/anything") is True
    assert helpers.get_registered_pass_through_route("/anything", "POST") == {}

    # If litellm stops calling these helpers, our overrides are dead code and its real route check runs instead.
    source = inspect.getsource(pass_through_endpoints)
    assert "is_registered_pass_through_route" in source
    assert "get_registered_pass_through_route" in source


def test_chunk_parser_patch_copies_model_and_usage():
    iterator = OpenAIChatCompletionResponseIterator(
        streaming_response=None, sync_stream=False
    )

    chunk = iterator.chunk_parser(
        {
            "id": "resp_1",
            "choices": [],
            "response": {
                "model": "gpt-5",
                "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
            },
        }
    )

    assert chunk.model == "gpt-5"
    assert chunk.usage.total_tokens == 8


def test_chunk_parser_is_still_used_by_litellm_pass_through_logging():
    source = inspect.getsource(openai_passthrough_logging_handler)

    assert "OpenAIChatCompletionResponseIterator" in source
    assert "chunk_parser" in inspect.getsource(OpenAIChatCompletionResponseIterator)


@pytest.mark.asyncio
async def test_success_callback_receives_usage_in_the_shape_we_read(
    providers: FakeProviders, monkeypatch: pytest.MonkeyPatch
):
    providers.respond("api.fireworks.ai", OPENAI_CHAT)
    received: list[tuple] = []

    async def spy(kwargs, completion_obj, start_time, end_time):
        received.append((kwargs, completion_obj, start_time, end_time))

    monkeypatch.setattr(litellm, "_async_success_callback", [spy])

    await litellm.acompletion(
        model="fireworks_ai/accounts/fireworks/models/codestral-2508",
        messages=MESSAGES,
        api_key="fake",
    )
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.1)

    assert received, (
        "litellm did not call the async success callback with four arguments"
    )
    _kwargs, completion_obj, _start, _end = received[0]
    # ai_gateway/proxy/clients/base.py reads completion_obj.get("usage") and then converts it.
    usage_metadata = _create_usage_metadata(completion_obj.get("usage"))
    assert usage_metadata["total_tokens"] == 8


@pytest.mark.asyncio
async def test_vertex_project_and_location_globals_are_still_read(
    providers: FakeProviders, fake_vertex_auth, monkeypatch: pytest.MonkeyPatch
):
    from tests.litellm_bump.conftest import GEMINI

    providers.respond("aiplatform.googleapis.com", GEMINI)
    # ai_gateway.config.setup_litellm assigns these two module globals.
    monkeypatch.setattr(litellm, "vertex_project", "global-project")
    monkeypatch.setattr(litellm, "vertex_location", "europe-west4")

    await litellm.acompletion(model="vertex_ai/gemini-2.5-flash", messages=MESSAGES)

    assert providers.last.url.host == "europe-west4-aiplatform.googleapis.com"
    assert "/projects/global-project/locations/europe-west4/" in providers.last.url.path


@pytest.mark.asyncio
async def test_mtls_client_session_hook_is_still_used_for_openai_compatible_calls(
    providers: FakeProviders, monkeypatch: pytest.MonkeyPatch
):
    # ai_gateway.config._setup_litellm_mtls assigns litellm.aclient_session so a client certificate is sent upstream.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OPENAI_CHAT, request=request)

    monkeypatch.setattr(
        litellm,
        "aclient_session",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    await litellm.acompletion(
        model="openai/gpt-4o",
        messages=MESSAGES,
        api_key="fake",
        api_base="https://models.internal/v1",
    )

    assert seen, (
        "litellm no longer uses litellm.aclient_session, so mTLS would be silently dropped"
    )
    assert not providers.requests
