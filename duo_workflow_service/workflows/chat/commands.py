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

from structlog import get_logger

from duo_workflow_service.tools.start_flow import CATALOG_FLOW_NAME
from duo_workflow_service.workflows.type_definitions import AdditionalContext

__all__ = [
    "CHAT_COMMAND_CATEGORY",
    "ForcedToolCall",
    "parse_forced_tool_call",
    "strip_command_context",
]

logger = get_logger("chat.commands")

# Internal envelope category. Rails must also list this in
# INTERNAL_CONTEXT_CATEGORIES, or reading the checkpoint back fails on the
# non-null AdditionalContextCategory GraphQL enum.
CHAT_COMMAND_CATEGORY = "duo_chat_command"

_FLOW_COMMAND = "flow"


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


# Routing straight to the tool node bypasses ChatAgent._get_approvals, so only
# commands whose tool is unconditionally pre-approved may be built here.
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
