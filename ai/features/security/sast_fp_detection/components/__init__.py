"""Feature-owned tools for the sast_fp_detection flow."""

from langchain.tools import BaseTool

from .post_sast_fp_analysis_to_gitlab import (
    PostSastFpAnalysisToGitlab,
    PostSastFpAnalysisToGitlabInput,
)

__all__ = [
    "FEATURE_TOOLS",
    "PostSastFpAnalysisToGitlab",
    "PostSastFpAnalysisToGitlabInput",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [PostSastFpAnalysisToGitlab],
}
