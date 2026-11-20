"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.post_duo_code_review`` instead.
"""

import warnings

from ai.features.review.code_review.components.post_duo_code_review import (
    PostDuoCodeReview,
    PostDuoCodeReviewInput,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.post_duo_code_review has moved to "
    "ai.features.review.code_review.components.post_duo_code_review",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PostDuoCodeReview",
    "PostDuoCodeReviewInput",
]
