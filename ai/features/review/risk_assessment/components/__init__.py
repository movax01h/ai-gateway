"""Feature-owned tools for the risk_assessment flow."""

from langchain.tools import BaseTool

from .submit_merge_request_risk_assessment import (
    RiskAssessmentAreaInput,
    SubmitMergeRequestRiskAssessment,
    SubmitMergeRequestRiskAssessmentInput,
)

__all__ = [
    "FEATURE_TOOLS",
    "RiskAssessmentAreaInput",
    "SubmitMergeRequestRiskAssessment",
    "SubmitMergeRequestRiskAssessmentInput",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [SubmitMergeRequestRiskAssessment],
}
