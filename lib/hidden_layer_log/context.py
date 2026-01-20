from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from lib.internal_events import current_event_context

__all__ = [
    "HiddenLayerLogContext",
    "current_hidden_layer_log_context",
    "set_hidden_layer_log_context",
]


@dataclass
class HiddenLayerLogContext:
    """Context for HiddenLayer logging of tool invocations.

    This context should be set before tool execution and is automatically used
    by HiddenLayer security scanning for logging ONLY.

    Attributes:
        tool_name: Name of the tool being invoked.
        tool_args: Arguments passed to the tool invocation.
        project_id: GitLab project ID associated with the invocation.
    """

    tool_name: str
    tool_args: Optional[Dict[str, Any]] = None
    project_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation with non-None fields only.

        Used for HiddenLayer logging and tracking purposes.

        Returns:
            Dictionary containing only fields with non-None values.
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }


current_hidden_layer_log_context: ContextVar[HiddenLayerLogContext] = ContextVar(
    "current_hidden_layer_log_context",
    default=HiddenLayerLogContext(tool_name="unknown"),
)


def set_hidden_layer_log_context(
    tool_name: str, tool_args: Optional[Dict[str, Any]] = None
) -> None:
    """Set HiddenLayer log context with data from current event context.

    This helper automatically populates project_id from the current event context,
    reducing boilerplate at call sites. The context is used for HiddenLayer security
    scanning and logging.

    Unlike other context setters that are called once per request in interceptors,
    this is called multiple times per request (once per tool invocation).

    Args:
        tool_name: Name of the tool being invoked.
        tool_args: Arguments passed to the tool invocation.
    """
    event_context = current_event_context.get()
    current_hidden_layer_log_context.set(
        HiddenLayerLogContext(
            tool_name=tool_name,
            tool_args=tool_args,
            project_id=event_context.project_id if event_context else None,
        )
    )
