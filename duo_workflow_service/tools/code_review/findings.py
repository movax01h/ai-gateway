"""Deprecated import location. Moved into the code_review feature package.

Import from ``ai.features.review.code_review.components.findings`` instead.
"""

import warnings

from ai.features.review.code_review.components.findings import (
    NEEDS_ATTENTION,
    PREVIOUS_FINDING_LABELS,
    PREVIOUS_FINDING_ORDER,
    SEVERITY_ORDER,
    anchor_fields,
    attribute_message,
    build_review_payload,
    build_summary,
    render_posted_finding,
    render_previous_findings,
    render_structured_finding,
    resolved_discussion_ids,
    select_findings,
    severity_counts,
)

warnings.warn(
    "duo_workflow_service.tools.code_review.findings has moved to "
    "ai.features.review.code_review.components.findings",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "NEEDS_ATTENTION",
    "PREVIOUS_FINDING_LABELS",
    "PREVIOUS_FINDING_ORDER",
    "SEVERITY_ORDER",
    "anchor_fields",
    "attribute_message",
    "build_review_payload",
    "build_summary",
    "render_posted_finding",
    "render_previous_findings",
    "render_structured_finding",
    "resolved_discussion_ids",
    "select_findings",
    "severity_counts",
]
