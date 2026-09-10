"""Deterministic publish step for the advanced code review flow.

Takes the reviewer's structured findings straight from its schema-validated final answer, applies the confidence gate,
orders and counts, composes the summary, and posts the findings as JSON. No finding passes through a model after the
reviewer writes it, so a line number cannot be recomputed and a finding cannot be reworded on the way to the reader.
The confidence gate is the one deliberate drop: logged and counted, never silent.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
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

# Also the publish order: the most serious comment is posted first.
SEVERITY_ORDER: tuple[str, ...] = ("critical", "major", "minor")


def _severity_rank(severity: Any) -> int:
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return len(SEVERITY_ORDER)


def select_findings(
    findings: List[Dict[str, Any]], min_confidence: int = 0
) -> tuple[List[Dict[str, Any]], int]:
    """Choose and order the findings that will be published.

    Drops a finding below ``min_confidence`` (only when it carries a score, so a schema slip degrades to publishing)
    and logs the drop. The survivors are ordered by severity, reviewer order within a severity, so the same input
    always publishes in the same order.

    Returns:
        The publishable findings and the number suppressed by the confidence gate.
    """
    kept: List[Dict[str, Any]] = []
    suppressed = 0
    for finding in findings:
        confidence = finding.get("confidence")
        if (
            min_confidence > 0
            and confidence is not None
            and confidence < min_confidence
        ):
            suppressed += 1
            logger.info(
                "Suppressing finding below confidence threshold",
                file=finding.get("file"),
                new_line=finding.get("new_line"),
                severity=finding.get("severity"),
                category=finding.get("category"),
                confidence=confidence,
                threshold=min_confidence,
            )
            continue
        kept.append(finding)

    return sorted(kept, key=lambda f: _severity_rank(f.get("severity"))), suppressed


def build_summary(findings: List[Dict[str, Any]], narrative: Optional[str]) -> str:
    """Compose the reader-facing overview: the reviewer's narrative, then computed counts.

    The narrative is the reviewer's own judgment, carried verbatim. Every number and the per-severity breakdown are
    computed here from the findings that will actually be posted, so the overview cannot contradict the comments it
    sits above.
    """
    text = (narrative or "").strip()
    if not findings:
        return text or "No issues were raised in this review."

    sections: list[str] = [text] if text else []
    counts = Counter(f.get("severity") for f in findings)
    parts = [
        f"{counts[severity]} {severity}"
        for severity in SEVERITY_ORDER
        if counts[severity]
    ]
    total = len(findings)
    file_count = len({f.get("file") for f in findings})
    sections.append(
        f"{total} finding{'s' if total != 1 else ''} "
        f"({', '.join(parts)}) across "
        f"{file_count} file{'s' if file_count != 1 else ''}:"
    )

    for severity in SEVERITY_ORDER:
        members = [f for f in findings if f.get("severity") == severity]
        if members:
            sections.append(_severity_section(severity.capitalize(), members))
    others = [f for f in findings if f.get("severity") not in SEVERITY_ORDER]
    if others:
        sections.append(_severity_section("Other", others))

    return "\n\n".join(sections)


# Also the render order: what the author fixed first, what still needs them last.
PREVIOUS_FINDING_ORDER: tuple[str, ...] = (
    "fixed",
    "verified",
    "partially_fixed",
    "still_outstanding",
)
PREVIOUS_FINDING_LABELS = {
    "fixed": "Fixed",
    "verified": "Verified",
    "partially_fixed": "Partially fixed",
    "still_outstanding": "Still outstanding",
}
NEEDS_ATTENTION = {"partially_fixed", "still_outstanding"}


def _previous_finding_rank(status: Any) -> int:
    try:
        return PREVIOUS_FINDING_ORDER.index(status)
    except ValueError:
        return len(PREVIOUS_FINDING_ORDER)


def render_previous_findings(items: List[Dict[str, Any]]) -> Optional[str]:
    """Render the reviewer's reconciliation of earlier threads as a status-first bullet list.

    Returns ``None`` unless at least one thread still needs the author's attention: a list made only of fixed and
    verified items reads as if there were something left to do, so a re-review that closed everything is published as
    a clean review instead.
    """
    if not any(item.get("status") in NEEDS_ATTENTION for item in items):
        return None

    ordered = sorted(items, key=lambda item: _previous_finding_rank(item.get("status")))
    lines = []
    for item in ordered:
        status = str(item.get("status"))
        label = PREVIOUS_FINDING_LABELS.get(status, status)
        lines.append(f"- **{label}:** `{item.get('file')}`: {item.get('note')}")
    return "\n".join(lines)


def _severity_section(title: str, members: List[Dict[str, Any]]) -> str:
    lines = [f"**{title}**"]
    lines.extend(
        f"- {member.get('category')}: `{member.get('file')}:{member.get('new_line')}`"
        for member in members
    )
    return "\n".join(lines)


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
        counts = Counter(f.get("severity") for f in published)
        logger.info(
            "Publishing code review findings",
            published=len(published),
            suppressed_below_threshold=suppressed,
            min_confidence=min_confidence,
            critical=counts["critical"],
            major=counts["major"],
            minor=counts["minor"],
        )

        payload = self._build_payload(published, summary_text, previous)
        response = await self._post_review(project_id, merge_request_iid, payload)
        return self._format_response(
            response, merge_request_iid, len(published), suppressed
        )

    def _build_payload(
        self,
        findings: List[Dict[str, Any]],
        summary: str,
        previous_findings: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shape the selected findings into the JSON document the review endpoint parses.

        Only the fields the endpoint anchors, renders or counts with are sent. The message is rendered here so the
        severity header and custom-instruction attribution are decided in one place; code travels verbatim. Severity
        travels as a field so the endpoint can count posted comments by severity.

        A rendered previous-findings list joins the summary when there are comments to post. With none, it travels on
        its own so the endpoint can introduce it as the outcome of a re-review rather than as a clean first review.
        """
        payload: Dict[str, Any] = {
            "findings": [self._render_finding(f) for f in findings],
            "summary": summary,
        }
        if previous_findings and findings:
            payload["summary"] = (
                f"{summary}\n\n**Previous findings**\n{previous_findings}"
            )
        elif previous_findings:
            payload["previous_findings"] = previous_findings
        return payload

    def _render_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        rendered: Dict[str, Any] = {
            "file": finding.get("file", ""),
            "new_line": finding.get("new_line"),
            "message": self._render_message(finding),
            "target_code": finding.get("target_code", ""),
            "severity": finding.get("severity"),
            "confidence": finding.get("confidence"),
        }
        if finding.get("old_line"):
            rendered["old_line"] = finding["old_line"]
        # An empty suggestion renders as a one-click patch that deletes the line, and the
        # reviewer is told to omit the field rather than empty it, so treat it as a slip.
        if finding.get("suggestion"):
            rendered["suggestion"] = finding["suggestion"]
            end_line = finding.get("end_line")
            new_line = finding.get("new_line")
            if (
                isinstance(end_line, int)
                and isinstance(new_line, int)
                and end_line > new_line
            ):
                rendered["end_line"] = end_line
        return rendered

    def _render_message(self, finding: Dict[str, Any]) -> str:
        """Render one finding's comment body.

        The reviewer schema tells the model NOT to restate severity, category, or the custom instruction attribution
        in the message because they render here: severity/category into the header, and the attribution as an
        "According to..." prefix on the message itself so the reader knows upfront the comment enforces their own rule.
        """
        severity = finding.get("severity") or ""
        category = finding.get("category") or ""
        lines = []
        if severity or category:
            header = f"[{severity.capitalize()}] {category}" if severity else category
            lines.append(f"**{header.strip()}**")
            lines.append("")
        message = str(finding.get("message", ""))
        if finding.get("custom_instruction_ref"):
            # Wording is matched verbatim by the monolith's CUSTOM_INSTRUCTIONS_REGEXP,
            # which counts attributed comments, and by the current flow's own output.
            message = (
                "According to custom instructions in "
                f"'{finding['custom_instruction_ref']}': {message}"
            )
        lines.append(message)
        return "\n".join(lines)

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
