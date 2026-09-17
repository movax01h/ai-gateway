"""Which failures a fanned-out item keeps to itself, and which end the flow."""

from typing import Any

from langgraph.errors import GraphRecursionError

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.errors.error_handler import ModelError
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
from duo_workflow_service.security.exceptions import SecurityException
from lib.usage_quota.errors import UsageQuotaError
from lib.usage_quota.service import InsufficientCredits

__all__ = [
    "ITEM_ERROR_SUBKEY",
    "TERMINAL_EXCEPTIONS",
    "AllItemsFailedError",
    "failed_item_errors",
    "item_error_record",
]

#: Sub-key an error record occupies inside one item's results entry. Namespaced
#: the way ``for_each_index`` is, because a bare ``error`` collides with what a
#: wrapped component publishes -- ``DeterministicStepComponent`` has one.
ITEM_ERROR_SUBKEY = "for_each_error"

# Failures that are not attributable to the item being processed. Recording one
# as that item's outcome describes a fault every sibling is about to hit -- and
# that whatever runs after the fan-out hits too -- as one item's bad luck.
TERMINAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    # The step limit, which the flow reports to the user as such.
    GraphRecursionError,
    # A rejected request, which deliberately leaves the flow out of FAILED.
    InvalidRequestException,
    # A provider failure past `ModelErrorHandler`'s retries, or a non-retryable
    # one: auth, permission, invalid request, context too large.
    ModelError,
    # The only exceptions allowed to put their own message in front of the user.
    NotifiableAgentException,
    NotifiableException,
    # Prompt injection and friends: never feed the result back to a model.
    SecurityException,
    # Out of credits/entitlements, or the quota check itself is unavailable.
    InsufficientCredits,
    UsageQuotaError,
)


class AllItemsFailedError(Exception):
    """Every item of a fan-out failed, so it published nothing anyone can read."""

    @classmethod
    def for_component(
        cls, component_name: str, errors: dict[str, dict[str, Any]]
    ) -> "AllItemsFailedError":
        """Report the whole fan-out's failure through one item's error.

        The reader needs a cause, not a count: a fault reaching every item is usually in what the fan-out runs.
        """
        index = min(errors, key=int)
        record = errors[index]
        kinds = {error["type"] for error in errors.values()}
        return cls(
            f"for_each on component '{component_name}' failed on all "
            f"{len(errors)} of its items, so it published no result a later "
            f"component can read. The items raised {len(kinds)} distinct error "
            f"type(s); item {index} raised {record['type']}: "
            f"{record['message']}. A fault that hits every item is usually in "
            f"the wrapped component's own configuration -- its prompt, tools or "
            f"inputs -- rather than in the items."
        )


def item_error_record(error: Exception) -> dict[str, Any]:
    """Render a contained failure as the entry that item's results slot holds."""
    return {ITEM_ERROR_SUBKEY: {"type": type(error).__name__, "message": str(error)}}


def failed_item_errors(entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pick the error records out of the per-item entries, keyed by item index.

    One place decides what counts as a failure, rather than every reader separately.
    """
    return {
        index: entry[ITEM_ERROR_SUBKEY]
        for index, entry in entries.items()
        if isinstance(entry, dict) and ITEM_ERROR_SUBKEY in entry
    }
