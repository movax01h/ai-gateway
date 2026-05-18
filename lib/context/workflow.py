"""Workflow context variables shared between ai_gateway and duo_workflow_service."""

from contextvars import ContextVar

_workflow_id: ContextVar[str] = ContextVar("workflow_id", default="undefined")


def set_workflow_id(workflow_id: str) -> None:
    """Set the current workflow ID in the context."""
    _workflow_id.set(workflow_id)


def get_workflow_id() -> str | None:
    """Get the current workflow ID, or None if not set."""
    wf_id = _workflow_id.get()
    return wf_id if wf_id != "undefined" else None
