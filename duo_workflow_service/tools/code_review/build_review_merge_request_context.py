"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.build_review_merge_request_context`` instead.
"""

import warnings

from ai.features.review.code_review.components.build_review_merge_request_context import (
    FULL_REVIEW_GUIDANCE,
    SCOPE_DELTA,
    SCOPE_DELTA_GUIDANCE,
    SCOPE_FAILED,
    SCOPE_NO_NEW_LINES,
    SCOPE_TRUNCATED,
    SHA_PATTERN,
    BuildReviewMergeRequestContext,
    BuildReviewMergeRequestContextInput,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.build_review_merge_request_context has moved to "
    "ai.features.review.code_review.components.build_review_merge_request_context",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FULL_REVIEW_GUIDANCE",
    "SCOPE_DELTA",
    "SCOPE_DELTA_GUIDANCE",
    "SCOPE_FAILED",
    "SCOPE_NO_NEW_LINES",
    "SCOPE_TRUNCATED",
    "SHA_PATTERN",
    "BuildReviewMergeRequestContext",
    "BuildReviewMergeRequestContextInput",
]
