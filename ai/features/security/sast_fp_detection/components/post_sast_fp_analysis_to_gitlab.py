import json
from typing import Any, ClassVar, Type

from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tier_access_checker import (
    LICENSED_FEATURE_SECURITY_DASHBOARD,
)


class PostSastFpAnalysisToGitlabInput(BaseModel):
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


class PostSastFpAnalysisToGitlab(DuoBaseTool):
    tier_check_licensed_feature: ClassVar[str] = LICENSED_FEATURE_SECURITY_DASHBOARD
    name: str = "post_sast_fp_analysis_to_gitlab"
    description: str = """Post SAST False Positive detection analysis results to GitLab via API.
    This tool posts the false positive analysis for a specific vulnerability using GitLab's API.
    For example:
    - Post FP analysis: post_sast_fp_analysis_to_gitlab(
        vulnerability_id=123,
        false_positive_likelihood=85,
        explanation="This appears to be a false positive because the input is not user-controlled."
    )
"""
    args_schema: Type[BaseModel] = PostSastFpAnalysisToGitlabInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self, vulnerability_id: int, false_positive_likelihood: float, explanation: str
    ) -> str:
        data = {
            "confidence_score": false_positive_likelihood,
            "description": explanation,
        }

        response = await self.gitlab_client.apost(
            path=f"/api/v4/vulnerabilities/{vulnerability_id}/flags/ai_detection",
            body=json.dumps(data),
        )

        body = self._process_http_response(
            "Post SAST false positive analysis", response
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
        self, args: PostSastFpAnalysisToGitlabInput, _tool_response: Any = None
    ) -> str:
        return (
            f"Post SAST false positive analysis for vulnerability {args.vulnerability_id} "
            f"(false_positive_likelihood: {args.false_positive_likelihood}%)"
        )
