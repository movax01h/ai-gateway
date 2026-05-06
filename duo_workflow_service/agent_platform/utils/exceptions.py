from __future__ import annotations

from pydantic_core import ValidationError


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
