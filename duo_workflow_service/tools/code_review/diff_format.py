"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.diff_format`` instead.
"""

import warnings

from ai.features.review.code_review.components.diff_format import (
    format_diff_lines,
    format_file_diff,
    format_file_diffs,
    format_renamed_files,
    is_binary_diff,
    split_unified_diff,
    walk_diff_lines,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.diff_format has moved to "
    "ai.features.review.code_review.components.diff_format",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "format_diff_lines",
    "format_file_diff",
    "format_file_diffs",
    "format_renamed_files",
    "is_binary_diff",
    "split_unified_diff",
    "walk_diff_lines",
]
