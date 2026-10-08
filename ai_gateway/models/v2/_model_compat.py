# Claude 4.6+ rejects requests that end with an assistant turn (prefill).
# Anthropic has stated the removal is permanent: no future model is expected
# to support assistant prefill.
# https://platform.claude.com/docs/en/about-claude/models/migration-guide#breaking-changes

from typing import Any, Optional

import litellm
import structlog

from ai_gateway.model_selection.model_selection_config import ModelSelectionConfig
from lib.context import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)

log = structlog.stdlib.get_logger("model_compat")

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


# An empty tool slot reads like a failed tool call to the model.
TOOL_IMAGE_PLACEHOLDER = "(see attached image)"
TOOL_IMAGES_BANNER = "Attached image(s) from tool result:"

# Providers whose litellm transform passes tool content verbatim to an
# OpenAI-shape endpoint, where the tool role has no image slot. Anthropic
# providers stay out: litellm converts their tool images natively, and
# Anthropic prompt caching anchors on the final message, which a hoisted user
# message would displace.
IMAGE_HOIST_PROVIDERS = frozenset(
    {"openai", "custom_openai", "fireworks_ai", "hosted_vllm"}
)

# Gemini and Claude share custom_llm_provider="vertex_ai"; only Gemini rejects
# tool-result images ("function_response.parts is not supported"), so the
# model name decides within these providers.
_GEMINI_HOIST_PROVIDERS = frozenset({"vertex_ai", "gemini"})


def hoist_tool_result_images(
    messages: list[dict], custom_llm_provider: Optional[str], model: Optional[str]
) -> list[dict]:
    """Lift tool-result images into a synthetic user message, for providers whose tool role cannot carry them.

    Each run of consecutive tool messages has its image blocks removed (other blocks keep their positions, an
    images-only result gets ``TOOL_IMAGE_PLACEHOLDER``) and collected into one user message appended after the run.
    Request-time only, never persisted.

    The synthetic message can land after a user turn or before an assistant turn. Templates strict enough to reject
    that also reject the tool role itself, so nothing reaching this path fits them anyway; merging into the adjacent
    user turn or inserting a filler assistant message are the known fixes if a tool-capable template ever needs one.

    Args:
        messages: Message dicts from the LiteLLM converter, image blocks already in OpenAI ``image_url`` form.
        custom_llm_provider: litellm provider tag used for the gate.
        model: The resolved model name; decides within Gemini-hosting providers.

    Returns:
        A new list with tool-result images hoisted, or ``messages`` itself when the gate is closed or there is
        nothing to hoist.
    """
    if not _should_hoist_tool_images(custom_llm_provider, model):
        return messages
    if not any(
        message.get("role") == "tool"
        and isinstance(message.get("content"), list)
        and any(_is_openai_image_block(block) for block in message["content"])
        for message in messages
    ):
        return messages

    hoisted: list[dict] = []
    pending_images: list[dict] = []

    def flush() -> None:
        if pending_images:
            hoisted.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TOOL_IMAGES_BANNER},
                        *pending_images,
                    ],
                }
            )
            pending_images.clear()

    for message in messages:
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, list):
                kept, images = _split_tool_images(content)
                if images:
                    pending_images.extend(images)
                    message = {
                        **message,
                        "content": kept
                        or [{"type": "text", "text": TOOL_IMAGE_PLACEHOLDER}],
                    }
            # A bare-string tool result is still part of the run: no flush.
            hoisted.append(message)
        else:
            flush()
            hoisted.append(message)
    flush()
    return hoisted


def _should_hoist_tool_images(
    custom_llm_provider: Optional[str], model: Optional[str]
) -> bool:
    if custom_llm_provider in IMAGE_HOIST_PROVIDERS:
        return True
    return (
        custom_llm_provider in _GEMINI_HOIST_PROVIDERS
        and "gemini" in (model or "").lower()
    )


def _is_openai_image_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "image_url"
        and bool(
            isinstance(block.get("image_url"), dict) and block["image_url"].get("url")
        )
    )


def _split_tool_images(content: list) -> tuple[list, list[dict]]:
    kept: list = []
    images: list[dict] = []
    for block in content:
        if _is_openai_image_block(block):
            images.append(block)
        else:
            kept.append(block)
    return kept, images


# No model identifier: the model repeats what it reads, and a deployment path is noise to the user.
IMAGE_OMITTED_NOTICE = (
    "[image omitted: the selected model does not support image input]"
)


def strip_image_blocks_for_non_vision_model(
    messages: list[dict], model: Optional[str], custom_llm_provider: Optional[str]
) -> list[dict]:
    """Replace image blocks with a text notice when the model is known not to see images.

    Only a positive "no" strips; unknown passes through so self-hosted vision models keep working.
    """
    if not model:
        return messages
    if not any(
        isinstance(message.get("content"), list)
        and any(_is_gated_image_block(block) for block in message["content"])
        for message in messages
    ):
        return messages
    if _model_supports_vision(model, custom_llm_provider) is not False:
        return messages

    log.info(
        "Replaced image blocks for non-vision model",
        model=model,
        custom_llm_provider=custom_llm_provider,
        verdict_source="flag"
        if _declared_vision_support(model) is not None
        else "litellm",
    )
    notice_text = IMAGE_OMITTED_NOTICE
    return [
        (
            message
            if not isinstance(message.get("content"), list)
            else {
                **message,
                "content": _replace_images_with_notice(message["content"], notice_text),
            }
        )
        for message in messages
    ]


def _replace_images_with_notice(blocks: list, notice_text: str) -> list:
    """Swap image blocks for a text notice, one per run of consecutive images."""
    replaced: list = []
    previous_was_notice = False
    for block in blocks:
        if _is_gated_image_block(block):
            if not previous_was_notice:
                replaced.append({"type": "text", "text": notice_text})
            previous_was_notice = True
            continue
        replaced.append(block)
        previous_was_notice = False
    return replaced


def _is_gated_image_block(block: Any) -> bool:
    """Either image shape: the strip runs before normalize, but an already-normalized block must not slip past."""
    return _is_standard_image_block(block) or _is_openai_image_block(block)


def _model_supports_vision(
    model: str, custom_llm_provider: Optional[str]
) -> Optional[bool]:
    """Return the vision verdict for ``model``: True, False, or None when nobody knows.

    The definition's ``supports_vision`` flag wins, then litellm's registry under
    the provider (how it files Fireworks models), then litellm by bare name. The
    flag comes first because litellm is wrong for some deployments: Minimax M3
    reads images live while the registry says it cannot. An unannotated registry
    entry counts as unknown, not as "no vision".
    """
    declared = _declared_vision_support(model)
    if declared is not None:
        return declared
    if custom_llm_provider:
        verdict = _litellm_vision_verdict(model, custom_llm_provider)
        if verdict is not None:
            return verdict
    return _litellm_vision_verdict(model, None)


def current_model_supports_vision() -> Optional[bool]:
    """Return ``False`` only when every model this request can run on is known not to see images.

    Components pick their model by tag, so the request's default alone is not enough. Any failure reads as unknown: this
    decides one advisory line, never whether the read succeeds.
    """
    try:
        verdicts = {_metadata_vision_verdict(m) for m in _request_models()}
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.debug(
            "vision verdict lookup failed", error=str(e), error_type=type(e).__name__
        )
        return None
    if not verdicts or None in verdicts:
        return None
    if all(verdicts):
        return True
    return False if not any(verdicts) else None


def _request_models() -> list[Any]:
    by_tag = current_model_metadata_with_size_context.get()
    if by_tag is not None:
        return [by_tag.default, *by_tag.by_tag.values()]
    metadata = current_model_metadata_context.get()
    return [metadata] if metadata is not None else []


def _metadata_vision_verdict(metadata: Any) -> Optional[bool]:
    # Fireworks and self-hosted metadata name the deployment in their request params; plain
    # GitLab-managed metadata does not, so the definition's params are the fallback.
    request_params = metadata.to_params() if hasattr(metadata, "to_params") else {}
    definition_params = getattr(
        getattr(metadata, "llm_definition", None), "params", None
    )
    model = (
        request_params.get("model")
        or getattr(definition_params, "identifier", None)
        or getattr(definition_params, "model", None)
    )
    if not model:
        return None
    declared = getattr(
        getattr(metadata, "llm_definition", None), "supports_vision", None
    )
    if declared is not None:
        return declared
    provider = request_params.get("custom_llm_provider") or getattr(
        definition_params, "custom_llm_provider", None
    )
    if provider and (verdict := _litellm_vision_verdict(model, provider)) is not None:
        return verdict
    return _litellm_vision_verdict(model, None)


def _declared_vision_support(model: str) -> Optional[bool]:
    """Return the context definition's ``supports_vision`` flag, only if that definition is for ``model``.

    A tag-routed component can run on a model other than the context's default, so the names have to match before the
    flag is trusted.
    """
    metadata = current_model_metadata_context.get()
    definition = getattr(metadata, "llm_definition", None)
    declared = getattr(definition, "supports_vision", None)
    if definition is None or declared is None:
        return None
    params = definition.params
    if model not in (
        getattr(params, "model", None),
        getattr(params, "identifier", None),
    ):
        return None
    return declared


def _litellm_vision_verdict(
    model: str, custom_llm_provider: Optional[str]
) -> Optional[bool]:
    """Return litellm's explicit ``supports_vision`` boolean for ``model``, else ``None``."""
    try:
        if custom_llm_provider:
            info = litellm.get_model_info(
                model, custom_llm_provider=custom_llm_provider
            )
        else:
            info = litellm.get_model_info(model)
    # litellm raises a plain Exception for unmapped models; only other failures are logged.
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "isn't mapped yet" not in str(e):
            log.debug(
                "litellm model info lookup failed",
                model=model,
                custom_llm_provider=custom_llm_provider,
                error=str(e),
                error_type=type(e).__name__,
            )
        return None
    supports = info.get("supports_vision")
    return supports if isinstance(supports, bool) else None
