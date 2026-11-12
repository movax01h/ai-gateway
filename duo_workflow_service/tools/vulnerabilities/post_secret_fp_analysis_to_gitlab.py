"""Deprecated import location. Moved into the secrets_fp_detection feature package.

Import from ``ai.features.security.secrets_fp_detection.components`` instead.
"""

import warnings

from ai.features.security.secrets_fp_detection.components.post_secret_fp_analysis_to_gitlab import (
    PostSecretFpAnalysisToGitlab,
    PostSecretFpAnalysisToGitlabInput,
)

warnings.warn(
    "duo_workflow_service.tools.vulnerabilities.post_secret_fp_analysis_to_gitlab has moved to "
    "ai.features.security.secrets_fp_detection.components.post_secret_fp_analysis_to_gitlab",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PostSecretFpAnalysisToGitlab", "PostSecretFpAnalysisToGitlabInput"]
