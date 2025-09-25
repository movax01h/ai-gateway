from contextvars import ContextVar
from typing import Any, Optional

_current_workflow_checkpointer: ContextVar[Optional[Any]] = ContextVar(
    "current_workflow_checkpointer", default=None
)


def set_workflow_checkpointer(checkpointer: Any) -> None:
    """Set the current workflow checkpointer for LLM tracking."""
    _current_workflow_checkpointer.set(checkpointer)


def get_workflow_checkpointer() -> Optional[Any]:
    """Get the current workflow checkpointer for LLM tracking."""
    return _current_workflow_checkpointer.get()


def clear_workflow_checkpointer() -> None:
    """Clear the current workflow checkpointer."""
    _current_workflow_checkpointer.set(None)
