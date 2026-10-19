"""Deterministic terminal step for a code review that has nowhere to post.

Everything ``post_duo_code_review_findings`` does except the posting: the confidence gate, severity ordering,
per-severity counts, the summary and the custom-instruction attribution, all in code. The result is returned as a
mapping so that a flow's state carries it structured: a DeterministicStepComponent stores the return value under
``context:<step>.tool_responses``, which makes ``context:<step>.tool_responses.findings`` and ``.summary``
addressable to routers, later steps and, through the checkpoint, to the client. Until ``flow.outputs`` binds them to
stable names, a client reads that path directly.

The result stays a mapping up to the transport limit (``MAX_MESSAGE_SIZE``) and collapses to a truncated string past
it. The default tool budget protects LLM context, which this result never enters, and the ``ui_chat_log`` copy of a
mapping is never truncated anyway, so a smaller budget would only corrupt the state copy while the wire still carried
the whole payload.
"""

from typing import Any, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.code_review.findings import (
    SEVERITY_ORDER,
    build_summary,
    render_structured_finding,
    select_findings,
    severity_counts,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import TruncationConfig
from duo_workflow_service.workflows.type_definitions import MAX_MESSAGE_SIZE

logger = structlog.stdlib.get_logger(__name__)

__all__ = ["FinalizeCodeReviewFindings", "FinalizeCodeReviewFindingsInput"]


class FinalizeCodeReviewFindingsInput(BaseModel):
    """Input schema for finalizing the reviewer's structured findings without posting them."""

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
            "The reviewer's own narrative recap, carried into the summary verbatim. "
            "Counts and the per-severity breakdown are computed from the findings "
            "regardless of what it claims."
        ),
    )
    min_confidence: int = Field(
        default=0,
        description=(
            "Keep only findings whose `confidence` is at least this value; 0 keeps "
            "everything. Reviewers score 0-10 and the value is not clamped, so a "
            "misconfigured threshold stays visible instead of being silently "
            "rewritten into a working one."
        ),
    )


class FinalizeCodeReviewFindings(DuoBaseTool):
    """Gate, order, count and summarise the reviewer's findings, and return them as the flow's result.

    Nothing is posted and nothing is read: the tool touches neither the GitLab API nor the executor, so the only
    thing it changes is the flow state it returns into.
    """

    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL
    truncation_config: TruncationConfig = Field(
        default_factory=lambda: TruncationConfig(
            max_bytes=MAX_MESSAGE_SIZE, truncated_size=MAX_MESSAGE_SIZE // 2
        )
    )

    name: str = "finalize_code_review_findings"
    description: str = (
        "Finalize a code review from structured findings without posting it. "
        "Applies the confidence gate, orders by severity, computes per-severity "
        "counts, composes the summary and renders custom-instruction attribution "
        "deterministically, returning the result as structured data."
    )
    args_schema: Type[BaseModel] = FinalizeCodeReviewFindingsInput

    async def _execute(
        self,
        findings: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[str] = None,
        min_confidence: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        published, suppressed = select_findings(findings or [], min_confidence)
        counts = severity_counts(published)
        logger.info(
            "Finalizing code review findings",
            published=len(published),
            suppressed_below_threshold=suppressed,
            min_confidence=min_confidence,
            **counts,
        )

        return {
            "findings": [render_structured_finding(f) for f in published],
            "summary": build_summary(published, summary),
            "counts": counts,
            "published": len(published),
            "suppressed_below_threshold": suppressed,
            "min_confidence": min_confidence,
        }

    def format_display_message(
        self, args: FinalizeCodeReviewFindingsInput, tool_response: Any = None
    ) -> str:
        if not isinstance(tool_response, dict):
            return "Finalize the review findings"

        published = tool_response.get("published", 0)
        counts = tool_response.get("counts") or {}
        breakdown = ", ".join(
            f"{counts[severity]} {severity}"
            for severity in SEVERITY_ORDER
            if counts.get(severity)
        )
        message = f"Review finished: {published} finding{'s' if published != 1 else ''}"
        if breakdown:
            message += f" ({breakdown})"
        suppressed = tool_response.get("suppressed_below_threshold", 0)
        if suppressed:
            message += (
                f", {suppressed} suppressed below confidence "
                f"{tool_response.get('min_confidence', args.min_confidence)}"
            )
        return message
