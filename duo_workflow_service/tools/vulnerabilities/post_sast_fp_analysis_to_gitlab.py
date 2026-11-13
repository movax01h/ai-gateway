"""Deprecated import location. Moved into the sast_fp_detection feature package.

Import from ``ai.features.security.sast_fp_detection.components`` instead.
"""

import warnings

from ai.features.security.sast_fp_detection.components.post_sast_fp_analysis_to_gitlab import (
    PostSastFpAnalysisToGitlab,
    PostSastFpAnalysisToGitlabInput,
)

warnings.warn(
    "duo_workflow_service.tools.vulnerabilities.post_sast_fp_analysis_to_gitlab has moved to "
    "ai.features.security.sast_fp_detection.components.post_sast_fp_analysis_to_gitlab",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PostSastFpAnalysisToGitlab", "PostSastFpAnalysisToGitlabInput"]
