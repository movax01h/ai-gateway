"""Deterministic publish step for the advanced code review flow.

Takes the reviewer's structured findings straight from its schema-validated final answer, applies the confidence gate,
orders and counts, composes the summary, and posts the findings as JSON. No finding passes through a model after the
reviewer writes it, so a line number cannot be recomputed and a finding cannot be reworded on the way to the reader.
The confidence gate is the one deliberate drop: logged and counted, never silent.
"""

import json
from typing import Any, Dict, List, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.code_review.findings import (
    SEVERITY_ORDER,
    build_review_payload,
    build_summary,
    render_previous_findings,
    select_findings,
    severity_counts,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

logger = structlog.stdlib.get_logger(__name__)

__all__ = [
    "SEVERITY_ORDER",
    "PostDuoCodeReviewFindings",
    "PostDuoCodeReviewFindingsInput",
    "build_summary",
    "render_previous_findings",
    "select_findings",
]


class PostDuoCodeReviewFindingsInput(BaseModel):
    """Input schema for posting Duo Code Review from the reviewer's structured findings."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The reviewer's `findings` array. Each item has file, new_line, "
            "target_code, severity, category, message, confidence, and optional "
            "old_line, end_line, suggestion, and custom_instruction_ref."
        ),
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "The reviewer's own narrative recap, carried into the published summary "
            "verbatim. Counts and the per-severity breakdown are computed from the "
            "findings regardless of what it claims."
        ),
    )
    previous_findings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "The reviewer's `previous_findings` array from a re-review. Each item "
            "has file, status (fixed, verified, partially_fixed, still_outstanding) "
            "and note. Rendered as a list only when something still needs attention."
        ),
    )
    min_confidence: int = Field(
        default=0,
        description=(
            "Publish only findings whose `confidence` is at least this value; 0 "
            "publishes everything. Reviewers score 0-10 and the value is not "
            "clamped, so a misconfigured threshold stays visible instead of being "
            "silently rewritten into a working one."
        ),
    )


class PostDuoCodeReviewFindings(DuoBaseTool):
    """Post a Duo Code Review from the reviewer's structured findings.

    Selects, orders, summarises and renders in code (no LLM) so that no finding can be dropped or reworded at publish
    time except by the counted confidence gate, then posts the review to the merge request.
    """

    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    name: str = "post_duo_code_review_findings"
    description: str = (
        "Post a Duo Code Review to a merge request from structured findings. "
        "Applies the confidence gate, orders by severity, composes the summary and "
        "renders the review comments deterministically."
    )
    args_schema: Type[BaseModel] = PostDuoCodeReviewFindingsInput

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        findings: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[str] = None,
        previous_findings: Optional[List[Dict[str, Any]]] = None,
        min_confidence: int = 0,
        **kwargs: Any,
    ) -> str:
        published, suppressed = select_findings(findings or [], min_confidence)
        summary_text = build_summary(published, summary)
        previous = render_previous_findings(previous_findings or [])
        logger.info(
            "Publishing code review findings",
            published=len(published),
            suppressed_below_threshold=suppressed,
            min_confidence=min_confidence,
            **severity_counts(published),
        )

        payload = build_review_payload(published, summary_text, previous)
        response = await self._post_review(project_id, merge_request_iid, payload)
        return self._format_response(
            response, merge_request_iid, len(published), suppressed
        )

    async def _post_review(
        self, project_id: int, merge_request_iid: int, payload: Dict[str, Any]
    ) -> dict:
        """Post review to GitLab API.

        The endpoint's `review_output` carries either the legacy review XML or, for this flow, the findings document
        as a JSON string; it picks the parser from the payload.
        """
        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "review_output": json.dumps(payload),
            "workflow_id": self.workflow_id,
        }
        response = await self.gitlab_client.apost(
            path="/api/v4/ai/duo_workflows/code_review/add_comments",
            body=json.dumps(request_body),
            parse_json=False,
        )

        try:
            return json.loads(response.body)
        except (TypeError, ValueError) as error:
            raise ToolException(
                f"Failed to post review: unreadable response from the review endpoint "
                f"(status {response.status_code}): {response.body!r}"
            ) from error

    def _format_response(
        self, response: dict, merge_request_iid: int, published: int, suppressed: int
    ) -> str:
        """Format API response as JSON string."""
        if response.get("message") == "Comments added successfully":
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Review posted to MR !{merge_request_iid}",
                    "published": published,
                    "suppressed_below_threshold": suppressed,
                }
            )
        raise ToolException(f"Failed to post review: {response}")

    def format_display_message(
        self, args: PostDuoCodeReviewFindingsInput, _tool_response: Any = None
    ) -> str:
        """Format a user-friendly display message."""
        return (
            f"Post Duo Code Review to merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )
