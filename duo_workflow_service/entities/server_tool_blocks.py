"""Project server-side tool content blocks (e.g. web search) to UiChatLog."""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Iterator, Optional, TypeGuard, Union

import structlog

from duo_workflow_service.entities import _openai_web_search as openai_web_search
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    build_tool_info,
)

log = structlog.stdlib.get_logger("server_tool_blocks")


def _is_anthropic_server_tool_use_block(block: Any) -> TypeGuard[dict]:
    """Matches any ``*_tool_use`` type; bare ``tool_use`` excluded by the ``_`` prefix."""
    return isinstance(block, dict) and str(block.get("type", "")).endswith("_tool_use")


def _is_anthropic_server_tool_result_block(block: Any) -> TypeGuard[dict]:
    """Matches any ``*_tool_result`` type; bare ``tool_result`` excluded by the ``_`` prefix."""
    return isinstance(block, dict) and str(block.get("type", "")).endswith(
        "_tool_result"
    )


def _is_server_tool_call_block(block: Any) -> TypeGuard[dict]:
    return _is_anthropic_server_tool_use_block(
        block
    ) or openai_web_search.is_call_block(block)


def text_segment_id(message_id: Optional[str], tool_count: int) -> Optional[str]:
    """Message id for the text segment after ``tool_count`` tool calls."""
    return message_id if tool_count == 0 else f"{message_id}:seg{tool_count}"


@dataclass
class AgentTextSegment:
    """A run of assistant text between two server-tool boundaries."""

    key: Optional[str]
    text: str
    index: int  # number of tool boundaries seen before this segment


@dataclass
class ServerToolBoundary:
    """A server-tool call block, in its position within the content stream."""

    block: dict
    index: int  # 0-based position among tool boundaries


ServerToolSegment = Union[AgentTextSegment, ServerToolBoundary]


def split_content_around_server_tools(
    content: list, message_id: Optional[str]
) -> Iterator[ServerToolSegment]:
    """Yield ordered text segments and the server-tool blocks that split them.

    Sole source of segmentation and keying, shared by the streaming and final-message paths.
    """
    tool_count = 0
    parts: list[str] = []

    def emit_accumulated_text() -> Iterator[AgentTextSegment]:
        """Emit the text accumulated since the last tool boundary, if any."""
        text = "".join(parts)
        if text:
            yield AgentTextSegment(
                text_segment_id(message_id, tool_count), text, tool_count
            )

    for block in content:
        if _is_server_tool_call_block(block):
            yield from emit_accumulated_text()
            parts = []
            yield ServerToolBoundary(block, tool_count)
            tool_count += 1
        elif _is_anthropic_server_tool_result_block(block):
            continue
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    yield from emit_accumulated_text()


def warn_unmatched_server_tool_results(content: Any) -> None:
    """Log server-tool blocks we could not place: a result with no call, an unknown action type."""
    blocks: list = content if isinstance(content, list) else []
    use_ids = {
        block.get("id")
        for block in blocks
        if _is_anthropic_server_tool_use_block(block)
    }
    for block in blocks:
        if (
            _is_anthropic_server_tool_result_block(block)
            and block.get("tool_use_id") not in use_ids
        ):
            log.warning(
                "Server tool result has no matching server tool use block",
                tool_use_id=block.get("tool_use_id"),
                block_type=block.get("type"),
            )
        elif openai_web_search.is_call_block(
            block
        ) and not openai_web_search.is_known_action(block):
            log.warning(
                "Server tool call has an unmapped action type; card falls back to web_search",
                action_type=(block.get("action") or {}).get("type"),
            )


class ServerToolResults:
    """The results behind each server-tool call in a message, and the TOOL card built from them.

    Anthropic pairs every call with a ``*_tool_result`` block; OpenAI emits none, so its
    results are reassembled from the whole message.
    """

    def __init__(self, content: Any):
        self._blocks: list = content if isinstance(content, list) else []

    @cached_property
    def _anthropic(self) -> dict[str, dict]:
        return {
            block["tool_use_id"]: block
            for block in self._blocks
            if _is_anthropic_server_tool_result_block(block)
            and block.get("tool_use_id")
        }

    @cached_property
    def _openai(self) -> dict[str, list[dict]]:
        """OpenAI results per call id, computed on first use: Anthropic messages never ask."""
        return openai_web_search.results_by_call_id(self._blocks)

    def build_ui_chat_log(
        self, call_block: dict, *, component_name: Optional[str] = None
    ) -> UiChatLog:
        if openai_web_search.is_call_block(call_block):
            name, args, status = openai_web_search.card_fields(call_block)
            tool_response = self._openai.get(call_block.get("id", ""))
        else:
            result = self._anthropic.get(call_block.get("id", ""))
            name = call_block.get("name") or "server_tool"
            args = call_block.get("input") or {}
            status = ToolStatus.SUCCESS if result else ToolStatus.PENDING
            tool_response = (result or {}).get("content")

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_sub_type=name,
            content=f"Using {name}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            correlation_id=None,
            # Empty means no key on both paths; `status` already says the search ran.
            tool_info=build_tool_info(name, args, tool_response or None),
            additional_context=None,
            message_id=call_block.get("id"),
            component_name=component_name,
        )
