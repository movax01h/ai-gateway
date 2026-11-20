"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.post_duo_code_review_findings`` instead.
"""

import warnings

from ai.features.review.code_review.components.post_duo_code_review_findings import (
    SEVERITY_ORDER,
    PostDuoCodeReviewFindings,
    PostDuoCodeReviewFindingsInput,
    build_summary,
    render_previous_findings,
    select_findings,
    truncation_notice,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.post_duo_code_review_findings has moved to "
    "ai.features.review.code_review.components.post_duo_code_review_findings",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SEVERITY_ORDER",
    "PostDuoCodeReviewFindings",
    "PostDuoCodeReviewFindingsInput",
    "build_summary",
    "render_previous_findings",
    "select_findings",
    "truncation_notice",
]
