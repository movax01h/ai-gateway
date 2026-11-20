"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.detect_review_context_truncation`` instead.
"""

import warnings

from ai.features.review.code_review.components.detect_review_context_truncation import (
    DetectReviewContextTruncation,
    DetectReviewContextTruncationInput,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.detect_review_context_truncation has moved to "
    "ai.features.review.code_review.components.detect_review_context_truncation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DetectReviewContextTruncation",
    "DetectReviewContextTruncationInput",
]
