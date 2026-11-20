"""Feature-owned tools for the code review flows.

The standard_code_review flow uses these tools too. The tools registry serves every feature tool by name, so they live
in one feature.
"""

from langchain.tools import BaseTool

from .detect_review_context_truncation import DetectReviewContextTruncation
from .finalize_code_review_findings import FinalizeCodeReviewFindings
from .post_duo_code_review import PostDuoCodeReview
from .post_duo_code_review_findings import PostDuoCodeReviewFindings

__all__ = [
    "FEATURE_TOOLS",
    "DetectReviewContextTruncation",
    "FinalizeCodeReviewFindings",
    "PostDuoCodeReview",
    "PostDuoCodeReviewFindings",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_only_files": [FinalizeCodeReviewFindings, DetectReviewContextTruncation],
    "read_write_gitlab": [PostDuoCodeReview, PostDuoCodeReviewFindings],
}
