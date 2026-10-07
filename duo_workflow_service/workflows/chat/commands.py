"""Client-forced tool calls for chat.

A slash command in the UI has to run one exact tool with fixed arguments every
time, so dispatch cannot go through the LLM. The client states the intent in an
internal ``additional_context`` envelope, which this module turns into a tool
call for the graph to execute directly.

The envelope rides ``additional_context`` because a new field on
``StartWorkflowRequest`` would not survive Workhorse: it decodes the client
event with ``protojson`` and ``DiscardUnknown``, which drops unknown fields
rather than preserving them. ``additional_context`` is an established internal
control channel (see ``orbit_context``).
"""

import json
from typing import Any, NamedTuple, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage
from structlog import get_logger

from duo_workflow_service.tools.start_flow import CATALOG_FLOW_NAME
from duo_workflow_service.workflows.type_definitions import AdditionalContext

__all__ = [
    "CHAT_COMMAND_CATEGORY",
    "FORCED_CALL_ID_PREFIX",
    "ForcedToolCall",
    "forced_tool_call_message_id",
    "is_forced_tool_call_message",
    "parse_forced_tool_call",
    "strip_command_context",
]

logger = get_logger("chat.commands")

# Internal envelope category. Rails must also list this in
# INTERNAL_CONTEXT_CATEGORIES, or reading the checkpoint back fails on the
# non-null AdditionalContextCategory GraphQL enum.
CHAT_COMMAND_CATEGORY = "duo_chat_command"

# Marks the assistant turn carrying a tool call the client chose rather than the
# model. The graph reads it back to tell whose call the tools node just ran, so
# it is a contract rather than a debugging aid: the prefix travels in the
# message id, which survives the checkpoint round-trip.
FORCED_CALL_ID_PREFIX = "forced-"

_FLOW_COMMAND = "flow"


def forced_tool_call_message_id() -> str:
    """Message id marking an assistant turn as client-forced."""
    return f"{FORCED_CALL_ID_PREFIX}{uuid4()!s}"


def is_forced_tool_call_message(message: BaseMessage) -> bool:
    """Whether this assistant turn carries a tool call the client chose.

    Read off the message rather than off the workflow, because the workflow's ``_forced_tool_call`` is scoped to the
    session and says only that *some* turn was forced. Routing needs to know whether *this* turn was, or every later
    turn in the same session inherits the answer.
    """
    return (
        isinstance(message, AIMessage)
        and bool(message.id)
        and str(message.id).startswith(FORCED_CALL_ID_PREFIX)
    )


class ForcedToolCall(NamedTuple):
    """A tool call the client has chosen on the user's behalf."""

    name: str
    args: dict[str, Any]


def _build_flow_call(payload: dict[str, Any]) -> Optional[ForcedToolCall]:
    consumer_id = payload.get("ai_catalog_item_consumer_id")
    if not isinstance(consumer_id, int) or isinstance(consumer_id, bool):
        logger.warning(
            "Ignoring flow command without a usable consumer id",
            consumer_id=consumer_id,
        )
        return None

    flow_args: dict[str, Any] = {
        "name": CATALOG_FLOW_NAME,
        "ai_catalog_item_consumer_id": consumer_id,
    }

    # An absent goal is omitted rather than sent as null so that
    # Ai::Catalog::Flows::ExecuteService falls back to the flow's description.
    goal = payload.get("goal")
    if goal is not None and not isinstance(goal, str):
        logger.warning("Ignoring flow command with a non-string goal")
        return None
    if goal:
        flow_args["goal"] = goal

    return ForcedToolCall(name="start_flow", args={"flow": flow_args})


# Routing straight to the tool node bypasses ChatAgent._get_approvals. That is the
# intent, not an oversight: typing the command *is* the approval, and asking the user
# to confirm an action they just spelled out is a prompt with one sensible answer.
# `start_flow` is deliberately not in the session's pre-approved privileges, so do not
# read this list as one of tools that are approved anyway.
#
# What makes the bypass safe is that the call is settled before the model sees it:
# every argument comes from the envelope, so a builder must never leave one for the
# model to fill, and the tool it names must do only what the command says.
#
# Governance still binds. ToolsRegistry.toolset drops denied tools, so a denied tool is
# absent from the toolset, the tools node cannot run it, and the turn falls through to
# the model to explain (see Workflow._turn_completed_a_forced_call). A deny rule outranks
# a command; only the per-call approval prompt is skipped.
_COMMAND_BUILDERS = {_FLOW_COMMAND: _build_flow_call}


def parse_forced_tool_call(
    additional_context: Optional[Sequence[AdditionalContext]],
) -> Optional[ForcedToolCall]:
    """The tool call requested by a chat command, if the client sent one."""
    for item in additional_context or []:
        if item.category != CHAT_COMMAND_CATEGORY or not item.content:
            continue

        try:
            payload = json.loads(item.content)
        except ValueError:
            logger.warning("Ignoring chat command with unparsable content")
            continue

        if not isinstance(payload, dict):
            logger.warning("Ignoring chat command that is not an object")
            continue

        command = payload.get("command")
        builder = _COMMAND_BUILDERS.get(command) if isinstance(command, str) else None
        if builder is None:
            logger.warning("Ignoring unknown chat command", command=command)
            continue

        return builder(payload)

    return None


def strip_command_context(
    additional_context: Optional[Sequence[AdditionalContext]],
) -> Optional[list[AdditionalContext]]:
    """Additional context without the command envelope.

    The chat prompt renders every context item verbatim, and the envelope is machine addressing rather than something
    the user supplied.

    Absence stays absence. ``None`` passes through, and a list left empty by dropping the envelope normalises back to
    ``None``, so a command-only turn looks like a turn that never carried additional context at all — the same
    reasoning as an attachments-only turn in ``Workflow._turn_context``. Handing back ``[]`` instead would change what
    every workflow reports, because ``with_attachment_references`` distinguishes the two.
    """
    if additional_context is None:
        return None

    return [
        item for item in additional_context if item.category != CHAT_COMMAND_CATEGORY
    ] or None
