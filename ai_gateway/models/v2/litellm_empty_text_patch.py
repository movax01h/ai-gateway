from typing import Any

from litellm.litellm_core_utils.prompt_templates import factory as _prompt_factory

# litellm >=1.85.0 unconditionally rewrites empty assistant text content to a
# human-readable placeholder before building Anthropic requests, even when a
# tool_call is present and the empty text is meaningless. That placeholder then
# leaks into the conversation and gets echoed back by the model on later turns
# (https://github.com/BerriAI/litellm/issues/24498). litellm's own block-builder
# already drops empty text blocks safely when a tool_call exists, so skip the
# placeholder rewrite for that case and defer to upstream for everything else.
_original_sanitize_empty_text_content = _prompt_factory._sanitize_empty_text_content


def _sanitize_empty_text_content(message: Any) -> Any:
    if message.get("role") == "assistant" and message.get("tool_calls"):
        content = message.get("content")
        if isinstance(content, str) and not content.strip():
            return message

    return _original_sanitize_empty_text_content(message)


_prompt_factory._sanitize_empty_text_content = _sanitize_empty_text_content
