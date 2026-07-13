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
