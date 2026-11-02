"""Errors raised while fetching or binding catalog items, split by who caused them.

The binding errors are ``ValueError`` subclasses, like the platform's other config errors.
"""

from typing import Optional

from langchain_core.tools import ToolException
from pydantic import ValidationError

from duo_workflow_service.errors.typing import NotifiableAgentException

__all__ = [
    "CatalogAgentsUnavailableError",
    "CatalogItemConfigError",
    "CatalogItemError",
    "CatalogItemsError",
    "CatalogItemsFetchError",
]


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


class CatalogItemsFetchError(ToolException):
    """GitLab refused or could not serve the catalog items request.

    Attributes:
        status_code: GitLab's HTTP status, or ``None`` when GitLab answered with a body that is not an object.
    """

    # A client error that a later identical request can still pass.
    _RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """Whether the same request can succeed later.

        Any other 4xx, like a 404 for a deleted flow, gets the same answer on every turn. A ``None`` status means
        GitLab answered 2xx with a body that is not an object: an API-contract defect an identical retry reproduces,
        so it is not retryable either.
        """
        if self.status_code is None:
            return False
        return (
            self.status_code >= 500
            or self.status_code in self._RETRYABLE_CLIENT_STATUSES
        )


class CatalogAgentsUnavailableError(NotifiableAgentException):
    """The run cannot use the AI Catalog agents its flow references.

    The UI message is static. GitLab's answer is a network response, so it stays in the server log.
    """

    REFUSED_MESSAGE = (
        "This flow stopped because it uses an AI Catalog agent that cannot be used. "
        "Check that the agent is published and that you have access to it."
    )
    FETCH_FAILED_MESSAGE = (
        "This flow stopped because GitLab could not load the AI Catalog agents it uses. "
        "Try again."
    )

    @classmethod
    def from_error(
        cls, error: CatalogItemsFetchError | CatalogItemsError
    ) -> "CatalogAgentsUnavailableError":
        """Wrap a fetch or binding error, with its detail kept server-side."""
        if isinstance(error, CatalogItemsFetchError):
            return cls(cls.FETCH_FAILED_MESSAGE, internal_detail=str(error))
        return cls(cls.REFUSED_MESSAGE, internal_detail=str(error))
