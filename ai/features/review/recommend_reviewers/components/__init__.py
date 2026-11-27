"""Feature-owned tools for the recommend_reviewers flow."""

from langchain.tools import BaseTool

from .add_merge_request_reviewers import (
    AddMergeRequestReviewers,
    AddMergeRequestReviewersInput,
)
from .post_suggested_reviewers import (
    PostSuggestedReviewers,
    PostSuggestedReviewersInput,
    SuggestedReviewerInput,
)

__all__ = [
    "FEATURE_TOOLS",
    "AddMergeRequestReviewers",
    "AddMergeRequestReviewersInput",
    "PostSuggestedReviewers",
    "PostSuggestedReviewersInput",
    "SuggestedReviewerInput",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [AddMergeRequestReviewers, PostSuggestedReviewers],
}
