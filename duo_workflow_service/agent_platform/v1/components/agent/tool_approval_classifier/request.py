"""Build the backend-independent input for one classifier review from the calling agent's state."""

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall

from duo_workflow_service.agent_platform.v1.components.agent.tool_approval_classifier.types import (
    ToolApprovalRequest,
)
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey

__all__ = ["build_tool_approval_request"]

# Bounds keep a huge request or a file write with a large body from blowing up the backend's input.
MAX_GOAL_CHARS = 4_000
MAX_ARGS_CHARS = 8_000
# Bounding each string argument first keeps a large body from pushing short arguments, such as a file path, out of
# the overall cut.
MAX_ARG_VALUE_CHARS = 2_000
MAX_DISPLAY_MESSAGE_CHARS = 2_000
MAX_USER_MESSAGES = 4
MAX_USER_MESSAGE_CHARS = 1_000
MAX_RECENT_ACTIONS = 6
MAX_ACTION_ARGS_CHARS = 300
MAX_ACTION_ARG_VALUE_CHARS = 100


def build_tool_approval_request(
    *,
    state: FlowState,
    tool_call: ToolCall,
    calling_agent: str,
    conversation_history_key: BaseIOKey,
    goal_key: BaseIOKey,
    display_message: Optional[str] = None,
) -> ToolApprovalRequest:
    """Build what a classifier backend sees of one pending tool call.

    Args:
        state: The calling component's flow state.
        tool_call: The pending call.
        calling_agent: The calling component's name.
        conversation_history_key: The calling node's history key. A supervisor-bound subagent's resolves to its
            subsession's history.
        goal_key: The calling node's goal key. A supervisor-bound subagent's resolves to its delegation prompt.
        display_message: How the call is shown to the human, when the tool renders one.

    Returns:
        The request, with every free-text field bounded.
    """
    goal = _value_or_none(goal_key, state)
    history = list(_value_or_none(conversation_history_key, state) or [])
    tool_args = dict(tool_call.get("args") or {})

    return ToolApprovalRequest(
        tool_name=tool_call["name"],
        tool_args_json=_truncate(
            _to_json(_bound_values(tool_args, MAX_ARG_VALUE_CHARS), indent=2),
            MAX_ARGS_CHARS,
        ),
        calling_agent=calling_agent,
        goal=_truncate(str(goal), MAX_GOAL_CHARS) if goal else None,
        display_message=(
            _truncate(display_message, MAX_DISPLAY_MESSAGE_CHARS)
            if display_message
            else None
        ),
        user_messages=_recent_user_messages(history),
        recent_actions=_recent_actions(history, pending_call_id=tool_call.get("id")),
    )


def _value_or_none(key: BaseIOKey, state: FlowState) -> Any:
    # A flow without a goal, or an agent before its first turn, must still be reviewable.
    try:
        return key.value_from_state(state)
    except KeyError:
        return None


def _recent_user_messages(history: list[BaseMessage]) -> tuple[str, ...]:
    texts = [
        _truncate(text, MAX_USER_MESSAGE_CHARS)
        for message in history
        if isinstance(message, HumanMessage) and (text := _text(message.content))
    ]
    return tuple(texts[-MAX_USER_MESSAGES:])


def _recent_actions(
    history: list[BaseMessage], pending_call_id: Optional[str]
) -> tuple[str, ...]:
    # The pending call's AIMessage is already in the history, and listing the call would read as if it had run.
    actions = [
        f"{call['name']} {_action_args(call.get('args') or {})}"
        for message in history
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if not pending_call_id or call.get("id") != pending_call_id
    ]
    return tuple(actions[-MAX_RECENT_ACTIONS:])


def _action_args(args: dict[str, Any]) -> str:
    bounded = _bound_values(args, MAX_ACTION_ARG_VALUE_CHARS)
    return _truncate(_to_json(bounded), MAX_ACTION_ARGS_CHARS)


def _text(content: Any) -> str:
    """The plain text of a message ``content``: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "\n".join(parts).strip()
    return ""


def _bound_values(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, limit)
    if isinstance(value, dict):
        return {key: _bound_values(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bound_values(item, limit) for item in value]
    return value


def _to_json(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, default=str)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… (truncated, {len(text) - limit} more characters)"
