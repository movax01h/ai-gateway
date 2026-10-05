"""One litellm call per provider path on the startup probe.

Each scenario sends a fixed request through the real litellm code and returns the parsed response text.
"""

from typing import Awaitable, Callable

import litellm

# Production applies this patch when it loads the models. The scenario must run with it, like production does.
import ai_gateway.models.v2.litellm_empty_text_patch  # noqa: F401  # pylint: disable=unused-import
from tests.litellm_bump.conftest import (
    ANTHROPIC_MESSAGE,
    BEDROCK_CONVERSE,
    GEMINI,
    OPENAI_CHAT,
    OPENAI_TEXT,
    FakeProviders,
)

MESSAGES = [{"role": "user", "content": "hello"}]
FIREWORKS_MODEL = "fireworks_ai/accounts/fireworks/models/codestral-2508"


async def fireworks_chat(providers: FakeProviders) -> str:
    providers.respond("api.fireworks.ai", OPENAI_CHAT)
    response = await litellm.acompletion(
        model=FIREWORKS_MODEL, messages=MESSAGES, api_key="fake", max_tokens=16
    )
    return response.choices[0].message.content


async def fireworks_text(providers: FakeProviders) -> str:
    providers.respond("api.fireworks.ai", OPENAI_TEXT)
    response = await litellm.atext_completion(
        model=FIREWORKS_MODEL, prompt="def add(", api_key="fake", max_tokens=16
    )
    return response.choices[0].text


async def vertex_gemini(providers: FakeProviders) -> str:
    providers.respond("aiplatform.googleapis.com", GEMINI)
    response = await litellm.acompletion(
        model="vertex_ai/gemini-2.5-flash",
        messages=MESSAGES,
        max_tokens=16,
        vertex_project="p1",
        vertex_location="us-east5",
    )
    return response.choices[0].message.content


async def vertex_claude(providers: FakeProviders) -> str:
    providers.respond("aiplatform.googleapis.com", ANTHROPIC_MESSAGE)
    response = await litellm.acompletion(
        model="vertex_ai/claude-haiku-4-5@20251001",
        messages=MESSAGES,
        max_tokens=16,
        vertex_project="p1",
        vertex_location="us-east5",
    )
    return response.choices[0].message.content


async def vertex_claude_empty_assistant_text(providers: FakeProviders) -> str:
    """An assistant turn with a tool call and empty text, as agent conversations produce."""
    providers.respond("aiplatform.googleapis.com", ANTHROPIC_MESSAGE)
    response = await litellm.acompletion(
        model="vertex_ai/claude-haiku-4-5@20251001",
        messages=[
            {"role": "user", "content": "list the files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "description": "List files",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        max_tokens=16,
        vertex_project="p1",
        vertex_location="us-east5",
    )
    return response.choices[0].message.content


async def bedrock_claude(providers: FakeProviders) -> str:
    providers.respond("bedrock-runtime", BEDROCK_CONVERSE)
    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=MESSAGES,
        max_tokens=16,
    )
    return response.choices[0].message.content


SCENARIOS: dict[str, Callable[[FakeProviders], Awaitable[str]]] = {
    "fireworks_chat": fireworks_chat,
    "fireworks_text": fireworks_text,
    "vertex_gemini": vertex_gemini,
    "vertex_claude": vertex_claude,
    "vertex_claude_empty_assistant_text": vertex_claude_empty_assistant_text,
    "bedrock_claude": bedrock_claude,
}
