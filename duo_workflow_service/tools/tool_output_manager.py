import json
from enum import Enum
from textwrap import dedent
from typing import Any

import structlog
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from duo_workflow_service.entities.image_blocks import (
    block_text,
    is_image_block,
    with_block_text,
)
from duo_workflow_service.security.tool_output_security import (
    TRUNCATED_TOOL_OUTPUT_TAG,
    TRUNCATION_NOTICE_TAG,
)
from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter

logger = structlog.get_logger("tools_executor")

token_counter = TikTokenCounter("planner")

# Stands in for the output inside a block list's truncation notice. The kept
# text stays in its own blocks; repeating it in the notice would double the
# payload the truncation exists to bound.
_TRUNCATED_BLOCKS_POINTER = "(the output kept is in the content blocks above)"


class TruncationDirection(str, Enum):
    """Direction for truncation."""

    FROM_START = "from_start"  # Keep the beginning (default)
    FROM_END = "from_end"  # Keep the end (most recent)


class TruncationConfig(BaseModel):
    """Configuration for tool output truncation limits."""

    max_bytes: int = 200 * 1024  # 200 KiB
    truncated_size: int = 100 * 1024  # 100 KiB
    direction: TruncationDirection = (
        TruncationDirection.FROM_START
    )  # Default to keeping the beginning


def _add_truncation_instruction(
    truncated_text: str, original_token_size: int, truncated_token_size: int
) -> str:
    """Create a formatted truncation notice message.

    ``truncated_text`` is the output itself for a string response. For a
    content-block list it is ``_TRUNCATED_BLOCKS_POINTER``, because the kept
    text stays in its own blocks and the notice rides along as a trailing one.
    """
    percentage = (truncated_token_size / original_token_size) * 100
    return dedent(f"""
        <{TRUNCATION_NOTICE_TAG}>
        IMPORTANT: This tool output has been truncated due to size limits.

        <truncation_details>
        - Original size: {original_token_size} tokens
        - Displayed size: {truncated_token_size} tokens
        - Percentage shown: {percentage:.1f}%
        </truncation_details>

        <{TRUNCATED_TOOL_OUTPUT_TAG}>
        {truncated_text}
        </{TRUNCATED_TOOL_OUTPUT_TAG}>

        <instructions>
        When generating a response based on truncated tool output, explicitly inform the user by including a note such as: "Note: This response is based on truncated tool output and may be incomplete."

        If you need information that might be in the missing portion, please try one of these actions:
        1. Refine your tool call to request a specific subset or filter the data
        2. Use alternative approaches to gather the necessary information
        </instructions>
        </{TRUNCATION_NOTICE_TAG}>
        """)


def truncate_string(
    text: str, tool_name: str, truncation_config: TruncationConfig
) -> str:
    """Truncate string if it exceeds the configured byte limits."""

    max_bytes = truncation_config.max_bytes
    truncated_size = truncation_config.truncated_size
    direction = truncation_config.direction

    encoded = text.encode("utf-8")

    if len(encoded) <= max_bytes:
        return text

    if direction == TruncationDirection.FROM_END:
        # Keep the end (most recent content)
        truncated_text = encoded[-truncated_size:].decode("utf-8", errors="ignore")
    else:
        # Keep the beginning (default behavior)
        truncated_text = encoded[:truncated_size].decode("utf-8", errors="ignore")

    # Log token size to be consistent
    original_token_size = token_counter.count_string_content(text)
    truncated_token_size = token_counter.count_string_content(truncated_text)

    logger.info(
        "Tool response exceeds max size and will be truncated",
        tool_name=tool_name,
        original_token_size=original_token_size,
        truncated_token_size=truncated_token_size,
        max_bytes=max_bytes,
        truncated_size=truncated_size,
        direction=direction.value,
    )

    truncated_output = _add_truncation_instruction(
        truncated_text=truncated_text,
        original_token_size=original_token_size,
        truncated_token_size=truncated_token_size,
    )

    return truncated_output


def _budgeted_text(block: Any) -> str:
    """The text *block* is charged to the response budget as.

    Image blocks are exempt, in the two shapes :func:`is_image_block` accepts.
    Slicing base64 destroys the image, the producer is expected to cap each one
    before building the block (#2819), and a block carrying only a remote
    ``url`` costs nothing to keep. Anything else that calls itself an image pays
    its JSON like any other block, so a payload under a key this module does
    not know cannot ride for free. A text block or bare string counts its text.
    Any other block counts its JSON, so a block this module cannot render still
    costs what it weighs.
    """
    if is_image_block(block):
        return ""
    text = block_text(block)
    if text is not None:
        return text
    return json.dumps(block)


def _cut_block(block: Any, text: str) -> Any:
    """*block* reduced to *text*.

    A text block or bare string keeps its shape. A block that was budgeted by its JSON becomes a text block carrying
    what survived, which is what the string path does to a whole response.
    """
    if block_text(block) is not None:
        return with_block_text(block, text)
    return {"type": "text", "text": text}


def _truncate_content_blocks(
    blocks: list, tool_name: str, truncation_config: TruncationConfig
) -> list:
    """Truncate a content-block list that carries an image.

    ``max_bytes`` budgets the whole response, as on the string path. Image
    blocks are exempt. Everything else shares the one budget: a text block by
    its text, any other block by its JSON (see :func:`_budgeted_text`). When
    the budget runs out inside a block, that block is cut; a non-text block
    that is cut becomes a text block with what survived (:func:`_cut_block`).
    Blocks that fit pass through untouched.

    The kept text stays in the blocks it came from, so block order and each
    block's position relative to an image survive. A block carrying no text is
    dropped rather than kept empty, whether this loop emptied it or it arrived
    that way: providers reject an empty text block, and the notice already says
    the output was cut. The notice is appended as its own trailing block.

    Returns ``blocks`` itself when nothing was cut.
    """
    texts = [_budgeted_text(block) for block in blocks]
    encoded = [text.encode("utf-8") for text in texts]
    if sum(len(item) for item in encoded) <= truncation_config.max_bytes:
        return blocks

    from_end = truncation_config.direction == TruncationDirection.FROM_END
    # Spend the budget from whichever end the caller wants to keep, so the
    # surviving text is the same text `truncate_string` would have kept.
    order = reversed(range(len(blocks))) if from_end else range(len(blocks))

    truncated: list[Any] = list(blocks)
    dropped: set[int] = set()
    remaining = truncation_config.truncated_size
    cut = False
    for index in order:
        size = len(encoded[index])
        if not size:
            if block_text(blocks[index]) is not None:
                # Carries text and weighs nothing: it arrived empty. Providers
                # reject an empty text block, so it goes the same way as one
                # this loop empties.
                dropped.add(index)
            continue
        if size <= remaining:
            remaining -= size
            continue
        cut = True
        if not remaining:
            dropped.add(index)
            continue
        kept_bytes = (
            encoded[index][-remaining:] if from_end else encoded[index][:remaining]
        )
        kept = kept_bytes.decode("utf-8", errors="ignore")
        remaining = 0
        if not kept:
            # The budget ended inside one multibyte character.
            dropped.add(index)
            continue
        truncated[index] = _cut_block(blocks[index], kept)

    if not cut:
        # Only a config with truncated_size above max_bytes gets here. Nothing
        # was cut, so no notice, and the list goes back untouched.
        return blocks

    truncated = [block for index, block in enumerate(truncated) if index not in dropped]
    original_token_size = token_counter.count_string_content("".join(texts))
    truncated_token_size = token_counter.count_string_content(
        "".join(_budgeted_text(block) for block in truncated)
    )

    logger.info(
        "Tool response exceeds max size and will be truncated",
        tool_name=tool_name,
        original_token_size=original_token_size,
        truncated_token_size=truncated_token_size,
        max_bytes=truncation_config.max_bytes,
        truncated_size=truncation_config.truncated_size,
        direction=truncation_config.direction.value,
    )

    truncated.append(
        {
            "type": "text",
            "text": _add_truncation_instruction(
                truncated_text=_TRUNCATED_BLOCKS_POINTER,
                original_token_size=original_token_size,
                truncated_token_size=truncated_token_size,
            ),
        }
    )
    return truncated


def _truncate_content(
    content: Any, tool_name: str, truncation_config: TruncationConfig
) -> Any:
    """Truncate message *content* of any shape; the same object comes back when nothing was cut.

    A lone image block passes through: it has no text to budget, and slicing the payload or the url would destroy it.
    A block list carrying an image, by the budget's own definition (:func:`is_image_block`), is kept as a list and goes
    through :func:`_truncate_content_blocks`, so a list is routed there exactly when it holds a block the string path
    must not slice. Everything else is the string path: a string as it is, anything else as its JSON.
    """
    if is_image_block(content):
        return content
    if isinstance(content, list) and any(is_image_block(block) for block in content):
        return _truncate_content_blocks(
            content, tool_name=tool_name, truncation_config=truncation_config
        )
    text = content if isinstance(content, str) else json.dumps(content)
    truncated = truncate_string(
        text, tool_name=tool_name, truncation_config=truncation_config
    )
    return content if truncated == text else truncated


def truncate_tool_response(
    tool_response: Any, tool_name: str, truncation_config: TruncationConfig
) -> Any:
    """Truncate tool response if it exceeds token limit."""

    try:
        # Skip the Command objects
        if isinstance(tool_response, Command):
            logger.info("Skip truncation for Command tool response")
            return tool_response

        if isinstance(tool_response, ToolMessage):
            # Unwrap, truncate the content like any other, copy the message
            # back only when something changed.
            content = _truncate_content(
                tool_response.content,
                tool_name=tool_name,
                truncation_config=truncation_config,
            )
            if content is tool_response.content:
                return tool_response
            return tool_response.model_copy(update={"content": content})

        return _truncate_content(
            tool_response, tool_name=tool_name, truncation_config=truncation_config
        )

    except Exception as e:
        logger.error(f"Abort tool response truncation due to unexpected error: {e}")
        return tool_response
