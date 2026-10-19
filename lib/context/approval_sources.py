"""Per-tool-call approval-source tracking context variable.

The ``ToolInvokedEvent`` audit event is emitted from a LangChain callback
(``on_tool_start``) and from tool-execution nodes that have no visibility into
the approval decision that authorized the call. The decision (and its source)
is known earlier and elsewhere: the v1 tool-approval node, the chat agent, and
the client-sent ``Approved`` proto handled in the workflow.

This module bridges that gap with a workflow-scoped registry keyed by tool call
id. Producers record the authorizing ``ApprovalSource`` when a call is approved
(explicitly, by pre-approval, or by silent reuse of a prior/session approval);
the audit emission points look it up so every ``ai_tool_invoked`` event carries
the source that authorized it.

The registry is a single mutable dict stored in a ContextVar (same pattern as
``tool_executions``): it is initialized once per workflow invocation and mutated
in place, so writes made inside one LangGraph node are visible to the callback
handler and to sibling nodes that share the copied context.

All record/get helpers below no-op (return ``None``) when the registry is
uninitialized, the ``tool_call_id`` is missing, or the value is empty, so
callers can record and look up unconditionally without guarding.
"""

from contextvars import ContextVar
from typing import NotRequired, Optional, TypedDict


class ApprovalAttribution(TypedDict):
    """Per tool-call-id attribution: the authorizing approval source and the
    optional policy provenance the client supplied. Both keys are recorded
    independently, so either may be absent."""

    source: NotRequired[str]
    policy_ref: NotRequired[dict[str, str]]


type ApprovalSourceRegistry = dict[str, ApprovalAttribution]

approval_sources: ContextVar[Optional[ApprovalSourceRegistry]] = ContextVar(
    "approval_sources", default=None
)


def init_approval_sources() -> None:
    """Initialize the approval-source registry with an empty dict.

    Called once at the start of a workflow invocation. Safe to call again; it resets the registry for a fresh run.
    """
    approval_sources.set({})


def _entry_for(tool_call_id: str) -> Optional[ApprovalAttribution]:
    registry = approval_sources.get()
    if registry is None:
        return None
    return registry.setdefault(tool_call_id, {})


def record_approval_source(tool_call_id: Optional[str], source: Optional[str]) -> None:
    """Record the approval source that authorized a specific tool call.

    Args:
        tool_call_id: The LangChain tool call id (``tool_call["id"]``).
        source: The ``ApprovalSource`` value (its string form) that authorized
            the call. ``ApprovalSource`` is a ``StrEnum``, so members serialize
            to their snake_case string automatically.
    """
    if not tool_call_id or not source:
        return
    entry = _entry_for(tool_call_id)
    if entry is None:
        return
    entry["source"] = str(source)


def record_approval_policy_ref(
    tool_call_id: Optional[str], policy_ref: Optional[dict[str, str]]
) -> None:
    """Record the policy provenance the client supplied for a tool call.

    An empty ``policy_ref`` dict (the client sent no meaningful provenance) is
    treated the same as missing.
    """
    if not tool_call_id or not policy_ref:
        return
    entry = _entry_for(tool_call_id)
    if entry is None:
        return
    entry["policy_ref"] = policy_ref


def get_approval_source(tool_call_id: Optional[str]) -> Optional[str]:
    """Look up the approval source recorded for a tool call."""
    if not tool_call_id:
        return None
    registry = approval_sources.get()
    if registry is None:
        return None
    entry = registry.get(tool_call_id)
    if not entry:
        return None
    return entry.get("source")


def get_approval_policy_ref(tool_call_id: Optional[str]) -> Optional[dict[str, str]]:
    """Look up the policy provenance recorded for a tool call."""
    if not tool_call_id:
        return None
    registry = approval_sources.get()
    if registry is None:
        return None
    entry = registry.get(tool_call_id)
    if not entry:
        return None
    return entry.get("policy_ref")
