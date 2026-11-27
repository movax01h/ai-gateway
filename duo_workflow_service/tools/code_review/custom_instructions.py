"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.custom_instructions`` instead.
"""

import warnings

from ai.features.review.code_review.components.custom_instructions import (
    CUSTOM_INSTRUCTION_FORMAT_HINT,
    INSTRUCTIONS_FILE_PATH,
    filter_matching_instructions,
    format_instructions,
    parse_instructions,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.custom_instructions has moved to "
    "ai.features.review.code_review.components.custom_instructions",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CUSTOM_INSTRUCTION_FORMAT_HINT",
    "INSTRUCTIONS_FILE_PATH",
    "filter_matching_instructions",
    "format_instructions",
    "parse_instructions",
]
