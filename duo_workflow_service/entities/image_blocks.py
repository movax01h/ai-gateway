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
checkpointed, leaving a short placeholder.

An image is therefore visible to the model **until the first checkpoint write of
the turn it arrived on**, not for the whole turn and not for later ones. Any
interrupt writes a checkpoint and resumes by reloading history from it
(``aget_tuple`` -> ``checkpoint_decoder``), and a tool-call approval is exactly
that. So "screenshot + fix the bug it shows" followed by an approval finishes
the turn without the image. This is a known v1 limitation, accepted because the
payload cannot live in the checkpoint as things stand: 2.5 MB decoded is ~3.3 MB
of base64 against a 4 MiB message cap, before any history. The durable fix is to
strip at next-turn ingestion rather than on every write, so the payload survives
the turn it belongs to; see the follow-up issue referenced from the module that
produces attachments.

The placeholder deliberately says nothing about where the image came from,
because this module cannot know: a block carries no provenance. A producer that
can say something useful should emit its own text block next to the image -- the
path a tool read it from, or the filename a user attached. That block is not an
image block, so it survives stripping untouched and is what lets the model say
which file it can no longer see rather than guessing.
"""

from typing import Any, Optional

from langchain_core.messages.content import create_image_block

__all__ = [
    "IMAGE_BLOCK_TOKEN_ESTIMATE",
    "image_content_block",
    "is_image_content_block",
    "strip_image_payloads",
]

# Flat per-image token cost charged wherever a conversation is budgeted.
#
# Not a magic number: it is the ceiling of what an image can cost, derived from
# Anthropic's published behaviour. They downscale anything larger to roughly
# 1.15 megapixels and bill about `width * height / 750` tokens, so the most any
# single image can cost is `1_150_000 / 750` ~= 1533; 1600 rounds that up.
#
# Being the ceiling is the point. A budget estimate that can only ever be too
# high makes context management compact early, which is recoverable. One that
# can be too low overflows the model's window mid-turn, which is not. The
# alternative of measuring the base64 string as if it were prose over-counts by
# two orders of magnitude and would compact away most of the conversation.
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
    Keeping base64 out of the checkpoint bounds its size, which matters because
    checkpoints are re-read on every resume and re-emitted whole on each new
    thread group.

    The cost is that the image is visible only until the *first* checkpoint write
    of its own turn. Because every interrupt writes a checkpoint and then resumes
    from it, a tool-call approval mid-turn is enough to lose it -- this is not
    limited to later turns. See the module docstring for why that is accepted for
    now and what the durable fix looks like.

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
