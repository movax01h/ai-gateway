"""Errors raised while binding catalog items, split by who caused them.

All are ``ValueError`` subclasses, like the platform's other config errors.
"""

from pydantic import ValidationError

__all__ = ["CatalogItemConfigError", "CatalogItemError", "CatalogItemsError"]


class CatalogItemError(ValueError):
    """Base for catalog item failures."""


class CatalogItemConfigError(CatalogItemError):
    """A flow's ``include`` section, or how a component claims it, is wrong.

    No request can bind to the flow until the config is fixed.
    """


class CatalogItemsError(CatalogItemError):
    """The items the client sent cannot be used with this flow."""

    @classmethod
    def from_validation_error(cls, exc: ValidationError) -> "CatalogItemsError":
        """Flatten a ``ValidationError`` into one error.

        Args:
            exc: The error raised while validating an item payload.

        Returns:
            One error listing every failure as ``field.path: message``. The paths are
            kept because they are what a caller reports back.
        """
        details = "; ".join(
            (
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                if err["loc"]
                else err["msg"]
            )
            for err in exc.errors()
        )
        return cls(f"Invalid catalog items: {details}")
