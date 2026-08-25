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
