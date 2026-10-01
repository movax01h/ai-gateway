"""Check what litellm sends to, and reads from, each provider on the startup probe path.

The startup probe (`/monitoring/ready`) calls Fireworks, Vertex AI and Bedrock through litellm. A litellm change that
breaks one of these paths stops instances from becoming ready.
"""

import litellm
import pytest

from tests.litellm_bump.conftest import (
    ANTHROPIC_MESSAGE,
    BEDROCK_CONVERSE,
    GEMINI,
    OPENAI_CHAT,
    OPENAI_TEXT,
    FakeProviders,
)

MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_fireworks_chat(providers: FakeProviders):
    providers.respond("api.fireworks.ai", OPENAI_CHAT)

    response = await litellm.acompletion(
        model="fireworks_ai/accounts/fireworks/models/codestral-2508",
        messages=MESSAGES,
        api_key="fake",
        max_tokens=16,
    )

    request = providers.last
    assert request.url.host == "api.fireworks.ai"
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == "Bearer fake"
    assert request.body["messages"] == MESSAGES
    assert request.body["max_tokens"] == 16
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_fireworks_text_completion(providers: FakeProviders):
    providers.respond("api.fireworks.ai", OPENAI_TEXT)

    response = await litellm.atext_completion(
        model="fireworks_ai/accounts/fireworks/models/codestral-2508",
        prompt="def add(",
        api_key="fake",
        max_tokens=16,
    )

    request = providers.last
    assert request.url.path.endswith("/completions")
    assert not request.url.path.endswith("/chat/completions")
    assert request.body["prompt"] == "def add("
    assert response.choices[0].text == "hi"


@pytest.mark.asyncio
async def test_vertex_gemini(providers: FakeProviders, fake_vertex_auth):
    providers.respond("aiplatform.googleapis.com", GEMINI)

    response = await litellm.acompletion(
        model="vertex_ai/gemini-2.5-flash",
        messages=MESSAGES,
        max_tokens=16,
        vertex_project="p1",
        vertex_location="us-east5",
    )

    request = providers.last
    assert request.url.host == "us-east5-aiplatform.googleapis.com"
    assert "/projects/p1/locations/us-east5/" in request.url.path
    assert request.url.path.endswith("gemini-2.5-flash:generateContent")
    assert request.headers["authorization"] == "Bearer fake-token"
    assert request.body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert 16 in request.body["generationConfig"].values()
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_vertex_claude(providers: FakeProviders, fake_vertex_auth):
    providers.respond("aiplatform.googleapis.com", ANTHROPIC_MESSAGE)

    response = await litellm.acompletion(
        model="vertex_ai/claude-haiku-4-5@20251001",
        messages=MESSAGES,
        max_tokens=16,
        vertex_project="p1",
        vertex_location="us-east5",
    )

    request = providers.last
    assert "/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict" in str(
        request.url
    )
    assert request.headers["authorization"] == "Bearer fake-token"
    assert request.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert request.body["max_tokens"] == 16
    assert "anthropic_version" in request.body
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_bedrock_claude(providers: FakeProviders, fake_aws):
    providers.respond("bedrock-runtime", BEDROCK_CONVERSE)

    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=MESSAGES,
        max_tokens=16,
    )

    request = providers.last
    assert request.url.host == "bedrock-runtime.us-east-1.amazonaws.com"
    assert request.url.path.endswith("/converse")
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert request.body["messages"] == [
        {"role": "user", "content": [{"text": "hello"}]}
    ]
    assert request.body["inferenceConfig"]["maxTokens"] == 16
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 8
