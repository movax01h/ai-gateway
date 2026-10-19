"""Detection of consecutive identical tool-call batches in conversation history."""

import json
from typing import NamedTuple, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

__all__ = [
    "IDENTICAL_CALL_NUDGE",
    "IDENTICAL_CALL_SKIPPED_UI_MESSAGE",
    "IDENTICAL_CALL_WRAP_UP",
    "IdenticalCallChain",
    "inspect_identical_call_chain",
    "is_guard_nudge",
    "nudge_message",
    "tool_calls_fingerprint",
]

IDENTICAL_CALL_NUDGE = (
    "You have issued this exact {tool_name} call {count} times in a row, and every "
    "execution returned an identical result. This call was not executed again. "
    "Use the result you already have and move on to the next step of the task."
)

IDENTICAL_CALL_WRAP_UP = (
    "You are stuck repeating the same tool call. This call was not executed. "
    "Do not make any more tool calls. Summarize what you have accomplished so far "
    "and what blocked you, and provide that as your final answer."
)

# Shown to the user in place of the tool result for a skipped call.
IDENTICAL_CALL_SKIPPED_UI_MESSAGE = (
    "Skipped repeated {tool_name} call: the same call already returned this result."
)

# Marks a ToolMessage the guard synthesized rather than a real tool result. Set in
# ``additional_kwargs`` because it round-trips through checkpoint serialization and
# is dropped when the history is rendered for the model, so the flag is invisible
# to the agent. Without it a nudge is indistinguishable from a tool result, and
# ``inspect_identical_call_chain`` would read the nudge text as a *changed* result,
# concluding the loop had broken and letting the repeated call run again.
_NUDGE_MARKER = "identical_call_guard_nudge"


def nudge_message(content: str, tool_call_id: Optional[str]) -> ToolMessage:
    """Build a guard nudge/wrap-up ToolMessage, marked so later scans can spot it."""
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        additional_kwargs={_NUDGE_MARKER: True},
    )


def is_guard_nudge(message: BaseMessage) -> bool:
    """Return True for a ToolMessage this guard synthesized."""
    return isinstance(message, ToolMessage) and bool(
        message.additional_kwargs.get(_NUDGE_MARKER)
    )


def tool_calls_fingerprint(message: BaseMessage) -> Optional[str]:
    """Return a stable fingerprint of a message's (tool, args) batch, or None without tool calls."""
    if not isinstance(message, AIMessage) or not message.tool_calls:
        return None
    return json.dumps(
        sorted(
            (
                tool_call["name"],
                json.dumps(tool_call.get("args", {}), sort_keys=True, default=str),
            )
            for tool_call in message.tool_calls
        )
    )


class IdenticalCallChain(NamedTuple):
    """What the trailing run of one repeated tool-call batch looks like.

    Attributes:
        repeats: Consecutive AIMessages issuing that batch, including the latest one.
        results_identical: Whether the two most recent *executed* runs returned the
            same results. False until two have executed, so the guard never trips
            on repetition alone.
        nudges: Nudges the guard has already sent within this run.
    """

    repeats: int
    results_identical: bool
    nudges: int


def inspect_identical_call_chain(
    history: Sequence[BaseMessage],
) -> IdenticalCallChain:
    """Measure the trailing run of consecutive identical tool-call batches.

    Walks back from the last message until something other than a matching AIMessage
    or a ToolMessage appears (a text-only reply, a different call, an injected
    HumanMessage). That boundary resets the guard and scopes ``nudges`` to the
    current run.
    """
    if not history:
        return IdenticalCallChain(repeats=0, results_identical=False, nudges=0)

    fingerprint = tool_calls_fingerprint(history[-1])
    if fingerprint is None:
        return IdenticalCallChain(repeats=0, results_identical=False, nudges=0)

    # Responses to the latest AIMessage don't exist yet, hence repeats starts at
    # 1 with no group of its own.
    repeats = 1
    nudges = 0
    executed_results: list[tuple[str, ...]] = []
    pending: list[ToolMessage] = []

    for message in reversed(history[:-1]):
        if isinstance(message, ToolMessage):
            pending.append(message)
            continue
        if tool_calls_fingerprint(message) != fingerprint:
            # Not part of the run; `pending` answered *this* message, so it is
            # discarded along with it.
            break

        repeats += 1
        responses = list(reversed(pending))
        pending = []

        nudges += sum(1 for response in responses if is_guard_nudge(response))
        # A skipped cycle produced only nudges, so it has no results to compare.
        # Counting its empty group would make it differ from a real one and read
        # as a changed result, which is the loop-breaking signal inverted.
        results = tuple(
            str(response.content)
            for response in responses
            if not is_guard_nudge(response)
        )
        if results and len(executed_results) < 2:
            executed_results.append(results)

    results_identical = (
        len(executed_results) == 2 and executed_results[0] == executed_results[1]
    )
    return IdenticalCallChain(
        repeats=repeats, results_identical=results_identical, nudges=nudges
    )
