import json

import pytest
from litellm.litellm_core_utils.prompt_templates import factory as prompt_factory

from ai_gateway.models.v2 import (
    litellm_empty_text_patch,  # noqa: F401  (applies the monkey-patch)
)

# Sourced from litellm.litellm_core_utils.prompt_templates.factory._sanitize_empty_text_content
PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


@pytest.mark.parametrize("content", ["", "   ", "\t\n"])
def test_sanitize_empty_text_content_keeps_empty_text_alongside_tool_call(content):
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "toolu_1", "type": "function"}],
    }
    result = prompt_factory._sanitize_empty_text_content(message)
    assert result["content"] == content


def test_sanitize_empty_text_content_still_replaces_truly_empty_message():
    message = {"role": "assistant", "content": ""}

    result = prompt_factory._sanitize_empty_text_content(message)

    assert result["content"] == PLACEHOLDER


def test_patch_is_reached_through_the_anthropic_transform():
    """The tests above call the patched function directly, which proves its logic but not that litellm still routes
    through it.

    `_sanitize_empty_text_content` is private, so a litellm bump can rename it or stop calling it and
    the placeholder would silently start leaking again. Drive the real Anthropic transform so that
    regression fails here instead.
    """
    messages = [
        {"role": "user", "content": "read the config file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    ]

    result = prompt_factory.anthropic_messages_pt(
        messages=messages,
        model="claude-sonnet-4-5",
        llm_provider="anthropic",
    )

    assert PLACEHOLDER not in json.dumps(result)
