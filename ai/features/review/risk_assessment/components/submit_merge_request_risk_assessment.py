import json
from typing import Any, List, Literal, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("workflow")

RESULTS_PATH = "/api/v4/ai/duo_workflows/tools/risk_assessment/results"


class RiskAssessmentAreaInput(BaseModel):
    name: str = Field(
        description="Area name, exactly as given in the area definitions the agent assessed."
    )
    level: Literal["low", "medium", "high", "critical"] = Field(
        description="The risk level of the change in this area."
    )
    reason: str = Field(
        description="The file or the change that decided the risk level."
    )


class SubmitMergeRequestRiskAssessmentInput(BaseModel):
    project_id: int = Field(
        description="Numeric ID of the project the merge request is in."
    )
    merge_request_iid: int = Field(
        description="IID of the merge request being assessed."
    )
    diff_sha: str = Field(
        description="The full 40-character SHA of the diff revision the assessment describes."
    )
    areas: List[RiskAssessmentAreaInput] = Field(
        description="The area assessments, unchanged from the assessment result."
    )


class SubmitMergeRequestRiskAssessment(DuoBaseTool):
    name: str = "submit_merge_request_risk_assessment"
    description: str = """Submit per-area risk levels for a merge request to GitLab.

    Records one risk level and reason for each area the change touches. GitLab applies each area's
    minimum risk level and decides the overall risk level. For example:
        submit_merge_request_risk_assessment(
            project_id=13,
            merge_request_iid=9,
            diff_sha="b83d6e391c22777fca1ed3012fce84f633d7fed0",
            areas=[{"name": "docs", "level": "low", "reason": "doc/user/index.md: wording only."}],
        )
    """
    args_schema: Type[BaseModel] = SubmitMergeRequestRiskAssessmentInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        diff_sha: str,
        areas: List[RiskAssessmentAreaInput],
    ) -> str:
        # Taken from the session rather than from the input, so GitLab links the assessment to the
        # run that produced them.
        if not self.workflow_id:
            raise ValueError(
                "submit_merge_request_risk_assessment requires a workflow_id in its metadata"
            )

        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "diff_sha": diff_sha,
            "workflow_id": int(self.workflow_id),
            # A DeterministicStepComponent passes the schema-validated answer as plain dicts.
            "areas": [
                RiskAssessmentAreaInput.model_validate(area).model_dump()
                for area in areas
            ],
        }

        try:
            response = await self.gitlab_client.apost(
                path=RESULTS_PATH,
                body=json.dumps(request_body),
            )
        except Exception as e:
            log.error(
                "submit_merge_request_risk_assessment: apost() raised",
                exception_class=type(e).__qualname__,
                project_id=project_id,
                merge_request_iid=merge_request_iid,
                areas_count=len(areas),
                exc_info=True,
            )
            raise

        self._process_http_response(
            "submit_merge_request_risk_assessment", response, log
        )

        return json.dumps(
            {
                "status": "success",
                "project_id": project_id,
                "merge_request_iid": merge_request_iid,
            }
        )

    def format_display_message(
        self,
        args: SubmitMergeRequestRiskAssessmentInput,
        tool_response: Any = None,
    ) -> str:
        base_msg = (
            f"Submit risk assessment for merge request !{args.merge_request_iid} "
            f"in project {args.project_id} ({len(args.areas)} area(s))"
        )

        # On failure the framework passes the exception text here; show it so it reaches the UI log.
        is_success_payload = False
        if tool_response:
            try:
                is_success_payload = (
                    json.loads(tool_response).get("status") == "success"
                )
            except (TypeError, ValueError, AttributeError):
                is_success_payload = False

        if tool_response and not is_success_payload:
            base_msg += f" -- {tool_response}"

        return base_msg
