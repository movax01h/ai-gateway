"""Deterministic publish step for the advanced code review flow.

Takes the reviewer's structured findings straight from its schema-validated final answer, applies the confidence gate,
orders and counts, composes the summary, renders the review XML, and posts it. No finding passes through a model after
the reviewer writes it, so a line number cannot be recomputed and a finding cannot be reworded on the way to the reader.
The confidence gate is the one deliberate drop: logged and counted, never silent.
"""

import json
import re
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
    "select_findings",
]

# Also the render order: the most serious comment is posted first.
SEVERITY_ORDER: tuple[str, ...] = ("critical", "major", "minor")

# The review endpoint parses this XML with regexes and never unescapes entities, so
# XML-escaping would reach the reader verbatim. Neutralise only what breaks the parse:
# a quote or a closing bracket inside an attribute, or a structural tag inside free text.
_UNSAFE_IN_ATTRIBUTE = re.compile(r'[">\n\r]')
_STRUCTURAL_TAG = re.compile(
    r"</?(?:comment\b[^>\n]*|from|to|review|summary|comments_summary)>",
    re.IGNORECASE,
)


def _defuse(text: str) -> str:
    """Break structural tags in free text so they cannot be parsed as markup.

    The space keeps the text readable and leaves every other character byte-identical, unlike XML escaping, which the
    endpoint would render literally.
    """
    return _STRUCTURAL_TAG.sub(lambda match: match.group(0).replace("<", "< ", 1), text)


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
    and a finding whose file path cannot be rendered as an XML attribute. Both drops are logged. The survivors are
    ordered by severity, reviewer order within a severity, so the same input always publishes in the same order.

    Returns:
        The publishable findings and the number suppressed by the confidence gate.
    """
    kept: List[Dict[str, Any]] = []
    suppressed = 0
    for finding in findings:
        file = str(finding.get("file", ""))
        if _UNSAFE_IN_ATTRIBUTE.search(file):
            # A bad attribute would corrupt the whole review; drop the one finding.
            logger.warning(
                "Dropping finding whose file path cannot be rendered as an attribute",
                file=file,
            )
            continue
        confidence = finding.get("confidence")
        if (
            min_confidence > 0
            and confidence is not None
            and confidence < min_confidence
        ):
            suppressed += 1
            logger.info(
                "Suppressing finding below confidence threshold",
                file=file,
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
            "old_line, suggestion, and custom_instruction_ref."
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
        min_confidence: int = 0,
        **kwargs: Any,
    ) -> str:
        published, suppressed = select_findings(findings or [], min_confidence)
        summary_text = build_summary(published, summary)
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

        review_output = self._render_review_output(published, summary_text)
        response = await self._post_review(project_id, merge_request_iid, review_output)
        return self._format_response(
            response, merge_request_iid, len(published), suppressed
        )

    def _render_review_output(
        self, findings: List[Dict[str, Any]], summary: str
    ) -> str:
        """Render the selected findings into the review XML the endpoint expects.

        No message is reworded. A suggestion whose code carries a structural tag is withheld and logged, because
        defusing it would publish a one-click patch containing the inserted space.
        """
        summary = _defuse(summary)
        if not findings:
            return f"<summary>\n{summary}\n</summary>"

        comments = []
        for f in findings:
            old_line = f.get("old_line")
            old_attr = f' old_line="{old_line}"' if old_line else ""
            parts = [
                f'<comment file="{f.get("file", "")}"{old_attr} new_line="{f.get("new_line", "")}">',
                _defuse(self._render_message(f)),
            ]
            target_code = str(f.get("target_code") or "")
            suggestion = str(f.get("suggestion") or "")
            if suggestion and target_code:
                if _STRUCTURAL_TAG.search(target_code) or _STRUCTURAL_TAG.search(
                    suggestion
                ):
                    # The endpoint renders <to> as a one-click suggestion, so a defused
                    # tag here would commit the inserted space. Keep the message, drop
                    # the patch.
                    logger.warning(
                        "Withholding suggestion whose code contains a structural tag",
                        file=f.get("file", ""),
                    )
                else:
                    parts += [
                        "<from>",
                        target_code,
                        "</from>",
                        "<to>",
                        suggestion,
                        "</to>",
                    ]
            parts.append("</comment>")
            comments.append("\n".join(parts))

        return (
            "<review>\n"
            + "\n".join(comments)
            + "\n</review>\n"
            + f"<comments_summary>\n{summary}\n</comments_summary>"
        )

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
        self, project_id: int, merge_request_iid: int, review_output: str
    ) -> dict:
        """Post review to GitLab API."""
        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "review_output": review_output,
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
