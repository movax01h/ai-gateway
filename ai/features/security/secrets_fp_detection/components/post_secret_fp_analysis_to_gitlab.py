import json
from typing import Any, ClassVar, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tier_access_checker import (
    LICENSED_FEATURE_SECURITY_DASHBOARD,
)


class PostSecretFpAnalysisToGitlabInput(BaseModel):
    vulnerability_id: int = Field(
        description="The numeric ID of the vulnerability analyzed"
    )
    false_positive_likelihood: float = Field(
        description="Likelihood that this vulnerability is a false positive (0-100)",
        ge=0,
        le=100,
    )
    explanation: str = Field(
        description="Detailed explanation of the analysis, reasoning, and conclusion"
    )


class PostSecretFpAnalysisToGitlab(DuoBaseTool):
    tier_check_licensed_feature: ClassVar[str] = LICENSED_FEATURE_SECURITY_DASHBOARD
    name: str = "post_secret_fp_analysis_to_gitlab"
    description: str = """Post Secret Detection False Positive detection analysis results to GitLab via API.
    This tool posts the false positive analysis for a specific secret vulnerability using GitLab's API.
    For example:
    - Post FP analysis: post_secret_fp_analysis_to_gitlab(
        vulnerability_id=123,
        false_positive_likelihood=85,
        explanation="This appears to be a false positive because it's test data."
    )
"""
    args_schema: Type[BaseModel] = PostSecretFpAnalysisToGitlabInput

    async def _execute(
        self, vulnerability_id: int, false_positive_likelihood: float, explanation: str
    ) -> str:
        # EXPERIMENTAL: This tool is experimental and should be unified with the SAST FP post tool in the future
        data = {
            "confidence_score": false_positive_likelihood,
            "description": explanation,
            "origin": "ai_secret_detection_fp_detection",  # Additional property for secret detection
        }

        response = await self.gitlab_client.apost(
            path=f"/api/v4/vulnerabilities/{vulnerability_id}/flags/ai_detection",
            body=json.dumps(data),
        )

        body = self._process_http_response(
            "Post secret false positive analysis", response
        )

        return json.dumps(
            {
                "status": "success",
                "vulnerability_id": vulnerability_id,
                "false_positive_likelihood": false_positive_likelihood,
                "response": body,
            }
        )

    def format_display_message(
        self, args: PostSecretFpAnalysisToGitlabInput, _tool_response: Any = None
    ) -> str:
        return (
            f"Post secret false positive analysis for vulnerability {args.vulnerability_id} "
            f"(false_positive_likelihood: {args.false_positive_likelihood}%)"
        )
