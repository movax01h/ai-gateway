"""Declarative ``for_each`` fan-out.

Fan-out is a wrapper around a component rather than a field on it, so a
component type needs no knowledge of ``for_each`` to be fanned out.
"""

from duo_workflow_service.agent_platform.experimental.components.for_each.component import (
    BRANCHES_SUBKEY,
    ERRORS_SUBKEY,
    FAILED_SUBKEY,
    ITEM_INDEX_CONTEXT_KEY,
    PROCESSED_ITEMS_SUBKEY,
    PUBLISHED_SUBKEYS,
    RESULTS_SUBKEY,
    SUCCEEDED_SUBKEY,
    TOTAL_ITEMS_SUBKEY,
    TRUNCATED_SUBKEY,
    ForEachComponent,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.config import (
    MAX_CONCURRENCY_CEILING,
    MAX_ITEMS_CEILING,
    ForEachConfig,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.errors import (
    ITEM_ERROR_SUBKEY,
    TERMINAL_EXCEPTIONS,
    AllItemsFailedError,
    failed_item_errors,
    item_error_record,
)
from duo_workflow_service.agent_platform.experimental.components.for_each.unit import (
    TerminalRouter,
    compile_as_unit,
)

__all__ = [
    "BRANCHES_SUBKEY",
    "ERRORS_SUBKEY",
    "FAILED_SUBKEY",
    "ITEM_ERROR_SUBKEY",
    "ITEM_INDEX_CONTEXT_KEY",
    "MAX_CONCURRENCY_CEILING",
    "MAX_ITEMS_CEILING",
    "PROCESSED_ITEMS_SUBKEY",
    "PUBLISHED_SUBKEYS",
    "RESULTS_SUBKEY",
    "SUCCEEDED_SUBKEY",
    "TERMINAL_EXCEPTIONS",
    "TOTAL_ITEMS_SUBKEY",
    "TRUNCATED_SUBKEY",
    "AllItemsFailedError",
    "ForEachComponent",
    "ForEachConfig",
    "TerminalRouter",
    "compile_as_unit",
    "failed_item_errors",
    "item_error_record",
]
