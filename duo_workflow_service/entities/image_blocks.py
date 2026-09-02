"""Image content blocks, independent of how the image got into the conversation.

An image can enter a turn from more than one direction — a file the user
attached (see :mod:`duo_workflow_service.entities.attachments`), or a tool
result such as ``read_file`` returning a screenshot. Everything in this module
works on the *content block* alone and never on the producing entity, so both
directions share one shape, one token estimate, and one checkpoint policy.

Shape
-----
LangChain standard blocks::

    {"type": "image", "id": ..., "base64": "<payload>", "mime_type": "image/png"}

Each provider integration translates these into its own native form, so
producers must not emit provider-specific shapes. Build them with
:func:`image_content_block` rather than by hand.

Lifetime
--------
:func:`strip_image_payloads` drops inline payloads before a message is
checkpointed, leaving a short placeholder. The placeholder deliberately says
nothing about where the image came from, because this module cannot know: a
block carries no provenance. A producer that *can* offer a recovery path should
emit its own text block next to the image (e.g. the path a tool read the image
from). That block is not an image block, so it survives stripping untouched and
tells the model on a later turn how to get the image back.
"""

from typing import Any, Optional

from langchain_core.messages.content import create_image_block

__all__ = [
    "IMAGE_BLOCK_TOKEN_ESTIMATE",
    "image_content_block",
    "is_image_content_block",
    "strip_image_payloads",
]

# Flat per-image token estimate used by `TikTokenCounter`. Real cost is
# resolution-dependent (Anthropic bills roughly `width * height / 750`); this is
# a deliberate over-estimate of a typical 1024x1024 screenshot so that context
# budgeting errs towards compacting early rather than overflowing the window.
# Anything is better than tiktoken-encoding the base64 string as prose, which
# over-counts by two orders of magnitude.
IMAGE_BLOCK_TOKEN_ESTIMATE = 1600


def image_content_block(base64: str, mime_type: str) -> dict[str, Any]:
    """Build the LangChain standard image block for an inline payload.

    The single constructor for image blocks. Going through it keeps every producer on the standard shape, which each
    provider integration knows how to translate; a hand-rolled provider-specific block would reach exactly one provider
    intact.
    """
    return dict(create_image_block(base64=base64, mime_type=mime_type))


def is_image_content_block(block: Any) -> bool:
    """Return whether *block* is an image content block carrying inline data.

    Only inline payloads match. A block referencing a remote ``url`` costs
    nothing to keep, so it is neither stripped nor charged the token estimate.
    """
    return (
        isinstance(block, dict)
        and block.get("type") == "image"
        and bool(block.get("base64"))
    )


def strip_image_payloads(content: Any) -> Any:
    """Replace inline image payloads in message *content* with text placeholders.

    Called when serialising any message for a checkpoint, whatever its role.
    Keeping base64 out of the checkpoint bounds its size — checkpoints are
    re-read on every resume — at the cost of the model no longer seeing the
    image on later turns.

    Args:
        content: ``BaseMessage.content`` — a string or a list of content blocks.

    Returns:
        *content* unchanged when it holds no inline image data, otherwise a new
        list with each image block replaced by a ``text`` block placeholder.
    """
    if not isinstance(content, list):
        return content
    if not any(is_image_content_block(block) for block in content):
        return content

    stripped: list[Any] = []
    for block in content:
        if is_image_content_block(block):
            stripped.append(
                {
                    "type": "text",
                    "text": _placeholder(block.get("mime_type")),
                }
            )
        else:
            stripped.append(block)
    return stripped


def _placeholder(mime_type: Optional[str]) -> str:
    """Return the text that stands in for a stripped image.

    Provenance-neutral on purpose: the same text has to read correctly for a
    file the user attached (which cannot be fetched again) and for a tool result
    (which can). Naming either one would mislead the model about the other.
    """
    return f"[{mime_type or 'image'} omitted from history]"
