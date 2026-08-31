import json
from typing import Any, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("workflow")

RESULTS_PATH = "/api/v4/ai/duo_workflows/tools/risk_classification/results"


class RiskClassificationClaimInput(BaseModel):
    name: str = Field(description="Claim name, exactly as given in required_claims.")
    value: str = Field(
        description="The categorical answer, e.g. true, false, or behavioral."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="A path:line reference into the diff supporting this answer, or null.",
    )


class SubmitMergeRequestRiskClassificationInput(BaseModel):
    project_id: int = Field(
        description="Numeric ID of the project the merge request is in."
    )
    merge_request_iid: int = Field(
        description="IID of the merge request being classified."
    )
    claims: List[RiskClassificationClaimInput] = Field(
        description="The answered claims, unchanged from the classification result."
    )
    summary: str = Field(
        description="Plain-language summary of the change and where its risk lies."
    )


class SubmitMergeRequestRiskClassification(DuoBaseTool):
    name: str = "submit_merge_request_risk_classification"
    description: str = """Submit categorical risk claims for a merge request to GitLab.

    Records the answered claims and summary produced by risk classification. This tool never
    accepts a score or a confidence value -- those are computed by GitLab from the claims. For
    example:
        submit_merge_request_risk_classification(
            project_id=13,
            merge_request_iid=9,
            claims=[{"name": "touches_auth", "value": "true", "evidence": "lib/auth.rb:44"}],
            summary="Adds a session token refresh path.",
        )
    """
    args_schema: Type[BaseModel] = SubmitMergeRequestRiskClassificationInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        claims: List[RiskClassificationClaimInput],
        summary: str,
    ) -> str:
        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "claims": [claim.model_dump() for claim in claims],
            "summary": summary,
        }

        try:
            response = await self.gitlab_client.apost(
                path=RESULTS_PATH,
                body=json.dumps(request_body),
            )
        except Exception as e:
            log.error(
                "submit_merge_request_risk_classification: apost() raised",
                exception_class=type(e).__qualname__,
                exception_repr=repr(e),
                project_id=project_id,
                merge_request_iid=merge_request_iid,
                claims_count=len(claims),
                exc_info=True,
            )
            raise

        # Raises ToolException on a non-2xx response. GitLab's REST API isn't
        # consistent about which key carries the error text (Grape param
        # validation failures use "error", explicit forbidden!/not_found!
        # calls use "message"), so this stringifies the whole body rather
        # than picking one key -- same convention every other REST-backed
        # tool in this codebase follows (see DuoBaseTool._process_http_response).
        self._process_http_response(
            "submit_merge_request_risk_classification", response, log
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
        args: SubmitMergeRequestRiskClassificationInput,
        tool_response: Any = None,
    ) -> str:
        base_msg = (
            f"Submit risk classification for merge request !{args.merge_request_iid} "
            f"in project {args.project_id} ({len(args.claims)} claim(s))"
        )

        # On failure the framework passes the raised exception's str() here (see
        # ToolNodeWithErrorCorrection._execute_tool); on success it passes this
        # tool's own JSON return value, which is redundant with base_msg above.
        # Parse it to check the actual status field rather than substring-matching
        # the text, so error text (permission failures, validation errors, etc.)
        # actually reaches the UI chat log instead of being silently dropped.
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
