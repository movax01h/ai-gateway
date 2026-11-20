"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.finalize_code_review_findings`` instead.
"""

import warnings

from ai.features.review.code_review.components.finalize_code_review_findings import (
    FinalizeCodeReviewFindings,
    FinalizeCodeReviewFindingsInput,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.finalize_code_review_findings has moved to "
    "ai.features.review.code_review.components.finalize_code_review_findings",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FinalizeCodeReviewFindings",
    "FinalizeCodeReviewFindingsInput",
]
