from __future__ import annotations

from pydantic import ValidationError

# The user-facing agent errors live in the shared errors package so the legacy ``components``
# tree can raise them without importing ``agent_platform``. Re-exported here for existing callers.
from duo_workflow_service.errors.typing import (
    TOOL_BLOCKED_BY_ADMINISTRATOR_RESULT,
    NotifiableAgentException,
    ToolApprovalUnavailableError,
    ToolBlockedByAdministratorError,
)

__all__ = [
    "TOOL_BLOCKED_BY_ADMINISTRATOR_RESULT",
    "FlowValidationError",
    "MissingAdditionalContextError",
    "NotifiableAgentException",
    "ToolApprovalUnavailableError",
    "ToolBlockedByAdministratorError",
]


class FlowValidationError(ValueError):
    """Raised when flow config validation fails with one or more errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))

    @classmethod
    def from_pydantic(cls, exc: ValidationError) -> FlowValidationError:
        """Create from a Pydantic ``ValidationError``.

        Formats each error as ``field.path: message`` or just ``message``
        when the error has no field location (e.g. model-level validators).
        """
        errors = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            msg = err["msg"]
            errors.append(f"{loc}: {msg}" if loc else msg)
        return cls(errors)


class MissingAdditionalContextError(KeyError):
    """Raised when a required flow input reads an additional context category the caller did not supply.

    Additional context is exposed under ``context:inputs.<category>``, so a missing category means the request lacked
    input the flow requires, not a service fault. Subclasses ``KeyError`` so existing handling of missing inputs keeps
    working.

    Args:
        message: The full missing-input message.
        category: The additional context category that was not supplied.
    """

    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
