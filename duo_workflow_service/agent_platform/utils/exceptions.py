from __future__ import annotations

from pydantic import ValidationError


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


class NotifiableAgentException(Exception):
    """Exception raised in the agent platform whose safe, user-facing message is surfaced in the UI.

    Use this exception when you need to communicate a safe, human-readable error to the end user
    while keeping sensitive debugging context (tokens, internal hostnames, stack traces, tool
    outputs) strictly server-side.

    Args:
        ui_message: A static or templated string that is safe to display to end users.
            Must **not** contain interpolated values from network responses, tool outputs,
            secrets, or LLM responses.
        internal_detail: Optional additional context logged server-side only.
            This value is never sent to the UI.
    """

    def __init__(self, ui_message: str, internal_detail: str | None = None) -> None:
        super().__init__(ui_message)
        self.ui_message = ui_message
        self.internal_detail = internal_detail


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
