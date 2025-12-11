import json
from typing import Any, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools import DuoBaseTool


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
            "detection_type": "secret_fp",  # Additional property for secret detection
        }

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/vulnerabilities/{vulnerability_id}/flags/ai_detection",
                body=json.dumps(data),
            )

            if not response.is_success():
                return json.dumps(
                    {
                        "error": f"Unexpected status code: {response.status_code} body: {response.body}"
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "vulnerability_id": vulnerability_id,
                    "false_positive_likelihood": false_positive_likelihood,
                    "response": response.body,
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to post secret false positive analysis for vulnerability {vulnerability_id}: {str(e)}"
                }
            )

    def format_display_message(
        self, args: PostSecretFpAnalysisToGitlabInput, _tool_response: Any = None
    ) -> str:
        return (
            f"Post secret false positive analysis for vulnerability {args.vulnerability_id} "
            f"(false_positive_likelihood: {args.false_positive_likelihood}%)"
        )
