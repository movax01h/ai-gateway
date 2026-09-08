# Claude 4.6+ rejects requests that end with an assistant turn (prefill).
# Anthropic has stated the removal is permanent: no future model is expected
# to support assistant prefill.
# https://platform.claude.com/docs/en/about-claude/models/migration-guide#breaking-changes

from typing import Any, Optional

from ai_gateway.model_selection.model_selection_config import ModelSelectionConfig

PREVIOUS_ASSISTANT_CONTEXT_PREFIX = "[Previous assistant context]: "


def supports_assistant_prefill(model: Optional[str]) -> bool:
    """Return whether `model` accepts an assistant message as the final turn.

    Source of truth is the `supports_assistant_prefill` flag on each model
    definition in `ai_gateway/model_selection/models.yml`.
    """
    if not model or "claude" not in model:
        return True
    for defn in ModelSelectionConfig.instance().get_resolved_llm_definitions().values():
        if defn.params.model == model:
            return defn.supports_assistant_prefill
    return False


def remove_trailing_assistant_message(payload: dict) -> dict:
    """Rewrite a trailing assistant message as a user message.

    The original prefill content is preserved (prefixed) so the model still sees it as context.
    """
    messages = payload.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return payload

    last_text = _extract_text(messages[-1].get("content", ""))
    payload["messages"] = [
        *messages[:-1],
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{PREVIOUS_ASSISTANT_CONTEXT_PREFIX}{last_text}",
                }
            ],
        },
    ]
    return payload


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "".join(parts)
    return ""


def normalize_image_blocks(messages: list[dict]) -> list[dict]:
    """Rewrite LangChain standard image blocks into the OpenAI/LiteLLM shape.

    ``langchain-anthropic`` translates standard content blocks
    (``{"type": "image", "base64": ..., "mime_type": ...}``) natively, but the
    LiteLLM adapter passes ``message.content`` through verbatim, so the blocks
    have to be converted to OpenAI's ``image_url`` data-URL form here.

    Blocks that are already in OpenAI form, carry a plain ``url``, or are not
    images are left untouched, so this is safe to run over every payload.

    Args:
        messages: Message dicts as produced by the LiteLLM message converter.

    Returns:
        A new list of message dicts with image blocks normalized. Messages
        without inline image data are returned unchanged (same object).
    """
    normalized = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(
            _is_standard_image_block(block) for block in content
        ):
            normalized.append(message)
            continue

        normalized.append(
            {
                **message,
                "content": [_to_openai_image_block(block) for block in content],
            }
        )
    return normalized


def _is_standard_image_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "image"
        and bool(block.get("base64") or block.get("url"))
    )


def _to_openai_image_block(block: Any) -> Any:
    if not _is_standard_image_block(block):
        return block

    if url := block.get("url"):
        return {"type": "image_url", "image_url": {"url": url}}

    mime_type = block.get("mime_type", "image/png")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{block['base64']}"},
    }
