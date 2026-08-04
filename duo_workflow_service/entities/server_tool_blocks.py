"""Project server-side tool content blocks (e.g. web search) to UiChatLog."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, TypeGuard, Union

from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    build_tool_info,
)


def is_anthropic_server_tool_use_block(block: Any) -> TypeGuard[dict]:
    """Matches any ``*_tool_use`` type; bare ``tool_use`` excluded by the ``_`` prefix."""
    return isinstance(block, dict) and str(block.get("type", "")).endswith("_tool_use")


def is_anthropic_server_tool_result_block(block: Any) -> TypeGuard[dict]:
    """Matches any ``*_tool_result`` type; bare ``tool_result`` excluded by the ``_`` prefix."""
    return isinstance(block, dict) and str(block.get("type", "")).endswith(
        "_tool_result"
    )


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
        if is_anthropic_server_tool_use_block(block):
            yield from emit_accumulated_text()
            parts = []
            yield ServerToolBoundary(block, tool_count)
            tool_count += 1
        elif is_anthropic_server_tool_result_block(block):
            continue
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    yield from emit_accumulated_text()


def _build_tool_card(
    *,
    name: str,
    args: dict,
    message_id: Optional[str],
    status: ToolStatus,
    tool_response: Any = None,
    component_name: Optional[str] = None,
) -> UiChatLog:
    """Build a TOOL ``UiChatLog`` entry."""
    return UiChatLog(
        message_type=MessageTypeEnum.TOOL,
        message_sub_type=name,
        content=f"Using {name}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
        correlation_id=None,
        tool_info=build_tool_info(name, args, tool_response),
        additional_context=None,
        message_id=message_id,
        component_name=component_name,
    )


def build_anthropic_tool_ui_chat_log(
    use_block: dict,
    result_block: Optional[dict] = None,
    *,
    component_name: Optional[str] = None,
) -> UiChatLog:
    """TOOL card for an Anthropic server-tool call: PENDING without ``result_block``,
    SUCCESS with."""
    tool_response = result_block.get("content") if result_block is not None else None
    status = ToolStatus.SUCCESS if result_block is not None else ToolStatus.PENDING

    return _build_tool_card(
        name=use_block.get("name") or "server_tool",
        args=use_block.get("input") or {},
        message_id=use_block.get("id"),
        status=status,
        tool_response=tool_response,
        component_name=component_name,
    )
