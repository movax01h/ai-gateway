"""What happens to the reviewer's findings after it writes them, shared by every terminal step.

The deep code review flow ends in pure code: the confidence gate, severity ordering, per-severity counts, the
summary and the custom-instruction attribution are all decided here, never by a model. Posting to a merge request and
returning the result to a local session are two terminal steps over this one module, so the operating point set by
`min_confidence` is the same wherever the review runs.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.stdlib.get_logger(__name__)

__all__ = [
    "NEEDS_ATTENTION",
    "PREVIOUS_FINDING_LABELS",
    "PREVIOUS_FINDING_ORDER",
    "SEVERITY_ORDER",
    "anchor_fields",
    "attribute_message",
    "build_review_payload",
    "build_summary",
    "render_message",
    "render_posted_finding",
    "render_previous_findings",
    "render_structured_finding",
    "select_findings",
    "severity_counts",
]

# Also the publish order: the most serious comment is posted first.
SEVERITY_ORDER: tuple[str, ...] = ("critical", "major", "minor")

_ANCHOR_KEYS = ("old_line", "suggestion", "end_line")


def _severity_rank(severity: Any) -> int:
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return len(SEVERITY_ORDER)


def select_findings(
    findings: List[Dict[str, Any]], min_confidence: int = 0
) -> tuple[List[Dict[str, Any]], int]:
    """Choose and order the findings that will be published.

    Drops a finding below `min_confidence` (only when it carries a score, so a schema slip degrades to publishing)
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


def severity_counts(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count findings per known severity; unknown severities are not counted here."""
    counts = Counter(f.get("severity") for f in findings)
    return {severity: counts[severity] for severity in SEVERITY_ORDER}


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

    Returns `None` unless at least one thread still needs the author's attention: a list made only of fixed and
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


def attribute_message(finding: Dict[str, Any]) -> str:
    """Prefix the message with its custom-instruction attribution, when the finding enforces one.

    The reviewer schema tells the model NOT to write the attribution itself because it renders here, so the reader knows
    upfront the comment enforces their own rule and the comment is never attributed twice.
    """
    message = str(finding.get("message", ""))
    if finding.get("custom_instruction_ref"):
        # Wording is matched verbatim by the monolith's CUSTOM_INSTRUCTIONS_REGEXP,
        # which counts attributed comments, and by the current flow's own output.
        message = (
            "According to custom instructions in "
            f"'{finding['custom_instruction_ref']}': {message}"
        )
    return message


def anchor_fields(finding: Dict[str, Any]) -> Dict[str, Any]:
    """The optional anchor fields worth carrying, under the rules the review endpoint anchors with."""
    anchors: Dict[str, Any] = {}
    if finding.get("old_line"):
        anchors["old_line"] = finding["old_line"]
    # An empty suggestion renders as a one-click patch that deletes the line, and the
    # reviewer is told to omit the field rather than empty it, so treat it as a slip.
    if finding.get("suggestion"):
        anchors["suggestion"] = finding["suggestion"]
        end_line = finding.get("end_line")
        new_line = finding.get("new_line")
        if (
            isinstance(end_line, int)
            and isinstance(new_line, int)
            and end_line > new_line
        ):
            anchors["end_line"] = end_line
    return anchors


def render_message(finding: Dict[str, Any]) -> str:
    """Render one finding's merge request comment body: a severity header, then the attributed message."""
    severity = finding.get("severity") or ""
    category = finding.get("category") or ""
    lines = []
    if severity or category:
        header = f"[{severity.capitalize()}] {category}" if severity else category
        lines.append(f"**{header.strip()}**")
        lines.append("")
    lines.append(attribute_message(finding))
    return "\n".join(lines)


def render_posted_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one finding for the review endpoint: only what it anchors, renders or counts with."""
    return {
        "file": finding.get("file", ""),
        "new_line": finding.get("new_line"),
        "message": render_message(finding),
        "target_code": finding.get("target_code", ""),
        "severity": finding.get("severity"),
        "confidence": finding.get("confidence"),
        **anchor_fields(finding),
    }


def render_structured_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one finding for a client that renders it: every schema field, the message attributed.

    Severity and category stay structured fields rather than a rendered header, so a client groups and labels without
    parsing prose. The anchor rules are the posting path's, so the two payloads agree on where a finding points.
    """
    rendered = {k: v for k, v in finding.items() if k not in _ANCHOR_KEYS}
    rendered["message"] = attribute_message(finding)
    rendered.update(anchor_fields(finding))
    return rendered


def build_review_payload(
    findings: List[Dict[str, Any]],
    summary: str,
    previous_findings: Optional[str] = None,
) -> Dict[str, Any]:
    """Shape the selected findings into the JSON document the review endpoint parses.

    Only the fields the endpoint anchors, renders or counts with are sent. The message is rendered here so the severity
    header and custom-instruction attribution are decided in one place; code travels verbatim. Severity travels as a
    field so the endpoint can count posted comments by severity.

    A rendered previous-findings list joins the summary when there are comments to post. With none, it travels on its
    own so the endpoint can introduce it as the outcome of a re-review rather than as a clean first review.
    """
    payload: Dict[str, Any] = {
        "findings": [render_posted_finding(f) for f in findings],
        "summary": summary,
    }
    if previous_findings and findings:
        payload["summary"] = f"{summary}\n\n**Previous findings**\n{previous_findings}"
    elif previous_findings:
        payload["previous_findings"] = previous_findings
    return payload
