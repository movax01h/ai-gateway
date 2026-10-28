"""Deprecated import location. Moved into the risk_classification feature package.

Import from ``ai.features.review.risk_classification.components`` instead.
"""

import warnings

from ai.features.review.risk_classification.components.submit_merge_request_risk_classification import (
    RESULTS_PATH,
    RiskClassificationClaimInput,
    SubmitMergeRequestRiskClassification,
    SubmitMergeRequestRiskClassificationInput,
)

warnings.warn(
    "duo_workflow_service.tools.risk_classification has moved to "
    "ai.features.review.risk_classification.components",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RESULTS_PATH",
    "RiskClassificationClaimInput",
    "SubmitMergeRequestRiskClassification",
    "SubmitMergeRequestRiskClassificationInput",
]
