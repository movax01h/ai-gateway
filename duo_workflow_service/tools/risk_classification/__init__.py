"""Deprecated import location. Moved into the risk_classification feature package.

Import from ``ai.features.review.risk_classification.components`` instead.
"""

from .submit_merge_request_risk_classification import (
    RiskClassificationClaimInput,
    SubmitMergeRequestRiskClassification,
    SubmitMergeRequestRiskClassificationInput,
)

__all__ = [
    "RiskClassificationClaimInput",
    "SubmitMergeRequestRiskClassification",
    "SubmitMergeRequestRiskClassificationInput",
]
