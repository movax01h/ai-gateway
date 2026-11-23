"""Shared user-message ingestion stages, consumed by every chat engine.

The legacy ``chat.Workflow`` and the engine-owned chat-partial path build user
turns from the same raw inputs: text, additional context, and (once the
attachments contract lands) multimodal content blocks. These helpers are the
single place that message shape is defined; the per-engine callers stay thin.
"""

from typing import Any, Optional, Union

from langchain_core.messages import HumanMessage

from duo_workflow_service.workflows.type_definitions import AdditionalContext

__all__ = [
    "ATTACHMENTS_KWARG",
    "ContentBlock",
    "assemble_user_message",
    "non_blank_text_block",
    "split_text_blocks",
]

# Where a user message keeps the records of its session files. The checkpoint keeps
# ``additional_kwargs``, and the prompt layer renders only ``additional_context``.
ATTACHMENTS_KWARG = "attachments"

# Element type of a langchain message content list. The ``str`` arm exists for
# compatibility with ``HumanMessage(content=...)`` (``list`` is invariant);
# these helpers only ever produce dict blocks.
ContentBlock = Union[str, dict[str, Any]]


def non_blank_text_block(text: str) -> list[ContentBlock]:
    """Wrap text as a one-element text-block list, or empty if blank.

    Providers reject degenerate text blocks, so every path that builds list content must apply the same blankness rule;
    this is that rule's one home.
    """
    return [{"type": "text", "text": text}] if text.strip() else []


def assemble_user_message(
    text: str,
    additional_context: Optional[list[AdditionalContext]] = None,
    content_blocks: Optional[list[dict[str, Any]]] = None,
    attachments: Optional[list[dict[str, Any]]] = None,
) -> HumanMessage:
    """Build the user-turn ``HumanMessage`` a chat engine appends to history.

    Without ``content_blocks`` this reproduces the legacy shape exactly: string
    content plus the additional-context metadata the prompt layer renders. With
    blocks, content becomes a list carrying the text block first, so multimodal
    payloads ride the message without any per-engine format. ``attachments`` are
    the turn's session file records, kept under :data:`ATTACHMENTS_KWARG`.
    """
    additional_kwargs: dict[str, Any] = {"additional_context": additional_context}
    if attachments:
        additional_kwargs[ATTACHMENTS_KWARG] = list(attachments)
    if not content_blocks:
        return HumanMessage(content=text, additional_kwargs=additional_kwargs)

    blocks: list[ContentBlock] = non_blank_text_block(text)
    blocks.extend(content_blocks)
    return HumanMessage(content=blocks, additional_kwargs=additional_kwargs)


def split_text_blocks(content: list[Any]) -> tuple[str, list[Any]]:
    """Split list content into its leading text and the blocks that follow it.

    Returns the text pieces before the first non-text block joined with newlines, and every block from there on in its
    original order. Callers that re-render text through a jinja template use this to keep non-text blocks (e.g. images)
    out of the template path, which would otherwise flatten them to their repr. Text after the first image stays where
    it is, so a label written just before its image (see ``attachment_content_blocks``) still sits next to it, and the
    model can tell several images apart.
    """
    texts: list[str] = []
    for index, block in enumerate(content):
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        else:
            return "\n".join(texts), list(content[index:])
    return "\n".join(texts), []
