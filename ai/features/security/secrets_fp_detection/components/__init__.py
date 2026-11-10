"""Feature-owned tools for the secrets_fp_detection flow."""

from langchain.tools import BaseTool

from .post_secret_fp_analysis_to_gitlab import (
    PostSecretFpAnalysisToGitlab,
    PostSecretFpAnalysisToGitlabInput,
)

__all__ = [
    "FEATURE_TOOLS",
    "PostSecretFpAnalysisToGitlab",
    "PostSecretFpAnalysisToGitlabInput",
]

# Tools this feature owns, grouped by the agent privilege that gates them.
# The tools registry merges these into its privilege map at discovery time.
FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
    "read_write_gitlab": [PostSecretFpAnalysisToGitlab],
}
