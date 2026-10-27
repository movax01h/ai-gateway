"""Feature-owned tools for the risk_classification flow."""

from langchain.tools import BaseTool

from .submit_merge_request_risk_classification import (
    RiskClassificationClaimInput,
    SubmitMergeRequestRiskClassification,
    SubmitMergeRequestRiskClassificationInput,
)

__all__ = [
    "FEATURE_TOOLS",
    "RiskClassificationClaimInput",
    "SubmitMergeRequestRiskClassification",
    "SubmitMergeRequestRiskClassificationInput",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [SubmitMergeRequestRiskClassification],
}
