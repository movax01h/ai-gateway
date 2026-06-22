"""Tool for submitting a complete MR review with inline comments, verdict, and summary."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

logger = structlog.stdlib.get_logger(__name__)

LINE_MATCH_THRESHOLD = 3


@dataclass
class DiffLine:
    """A parsed line from a unified diff."""

    old_line: Optional[int]
    new_line: Optional[int]
    text: str
    line_type: str  # "added", "deleted", "context"


def parse_diff_lines(raw_diff: str) -> list[DiffLine]:
    """Parse a unified diff into structured DiffLine objects.

    Line counters start unset and are only established by a parseable ``@@`` hunk
    header. Lines under a header we cannot parse are skipped (and logged) rather
    than emitted with guessed positions, so a malformed hunk never yields a
    finding pinned to a bogus line number.
    """
    lines = []
    old_line: Optional[int] = None
    new_line: Optional[int] = None

    for line in raw_diff.split("\n"):
        if not line:
            continue

        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
            else:
                # Unparsable hunk header — stop attributing lines until the next
                # valid header rather than defaulting to misleading positions.
                old_line = None
                new_line = None
                logger.warning("Skipping unparsable diff hunk header", header=line[:80])
            continue

        if (
            line.startswith("+++")
            or line.startswith("---")
            or line.startswith("diff --git")
        ):
            continue

        if line.startswith("\\"):
            continue

        # No valid hunk header seen yet (or the last one was malformed): skip
        # rather than emit a line with a guessed position.
        if old_line is None or new_line is None:
            continue

        if line.startswith("+"):
            lines.append(
                DiffLine(
                    old_line=None,
                    new_line=new_line,
                    text=line[1:],
                    line_type="added",
                )
            )
            new_line += 1
        elif line.startswith("-"):
            lines.append(
                DiffLine(
                    old_line=old_line,
                    new_line=None,
                    text=line[1:],
                    line_type="deleted",
                )
            )
            old_line += 1
        else:
            text = line[1:] if line.startswith(" ") else line
            lines.append(
                DiffLine(
                    old_line=old_line,
                    new_line=new_line,
                    text=text,
                    line_type="context",
                )
            )
            old_line += 1
            new_line += 1

    return lines


def find_line_by_numbers(
    comment_old_line: Optional[int],
    comment_new_line: Optional[int],
    diff_lines: list[DiffLine],
) -> Optional[DiffLine]:
    """Find a diff line by matching old_line and new_line numbers.

    Mirrors ProcessCommentsService#find_line_by_line_numbers: first tries exact match on both, then falls back to
    new_line only.
    """
    # Exact match on both old_line and new_line
    for dl in diff_lines:
        if dl.old_line == comment_old_line and dl.new_line == comment_new_line:
            return dl

    # Fall back to new_line only
    if comment_new_line is not None:
        for dl in diff_lines:
            if dl.new_line == comment_new_line:
                return dl

    return None


def find_line_by_content(
    suggestion_lines: list[str],
    diff_lines: list[DiffLine],
) -> Optional[DiffLine]:
    """Find a diff line by matching consecutive content lines.

    Mirrors ProcessCommentsService#find_line_by_content: searches for a consecutive sequence of lines matching the
    suggestion context, skipping deleted lines (since suggestions can only apply to non-deleted lines).
    """
    if len(suggestion_lines) < LINE_MATCH_THRESHOLD:
        return None

    # Filter out deleted lines — suggestions can't target them
    actual_lines = [dl for dl in diff_lines if dl.line_type != "deleted"]

    for start_idx, start_line in enumerate(actual_lines):
        if start_idx + len(suggestion_lines) > len(actual_lines):
            break

        if start_line.text != suggestion_lines[0]:
            continue

        # Try to match the entire sequence
        if all(
            actual_lines[start_idx + i].text == suggestion_lines[i]
            for i in range(len(suggestion_lines))
        ):
            return start_line

    return None


def match_comment_to_diff_line(
    comment: "MrReviewComment",
    diff_lines: list[DiffLine],
) -> Optional[DiffLine]:
    """Match a comment to the correct diff line, with content-based fallback.

    Mirrors ProcessCommentsService#match_comment_to_diff_line.
    """
    line_match = find_line_by_numbers(comment.old_line, comment.new_line, diff_lines)

    suggestion_lines = comment.suggestion.splitlines() if comment.suggestion else []

    if len(suggestion_lines) >= LINE_MATCH_THRESHOLD:
        # If line_match already points to the right content, use it
        if line_match and line_match.text == suggestion_lines[0]:
            return line_match

        # Try content-based matching, fall back to line number match
        return find_line_by_content(suggestion_lines, diff_lines) or line_match

    return line_match


class MrReviewComment(BaseModel):
    """A single inline diff comment."""

    file: str = Field(description="File path as shown in the diff (new_path)")
    new_line: Optional[int] = Field(
        default=None,
        description="Line number in the new version of the file",
    )
    old_line: Optional[int] = Field(
        default=None,
        description="Line number in the old version of the file",
    )
    body: str = Field(description="Comment body in markdown")
    suggestion: Optional[str] = Field(
        default=None,
        description="Code suggestion to replace the commented line(s). "
        "Will be formatted as a GitLab suggestion block.",
    )


class SubmitMrReviewInput(BaseModel):
    """Submit a complete MR review with inline comments, verdict, and optional summary."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    comments: list[MrReviewComment] = Field(
        default=[],
        description="Inline diff comments to post on specific lines",
    )
    verdict: Optional[Literal["requested_changes", "reviewed"]] = Field(
        default=None,
        description="Review verdict: 'requested_changes' or 'reviewed'",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary note body (markdown) to post on the MR",
    )
    summary_internal: bool = Field(
        default=False,
        description="If true, the summary note is posted as an internal note",
    )
    fold_inline_into_summary_when_public: bool = Field(
        default=False,
        description=(
            "If true, when the project is public the inline comments are folded into the "
            "summary note (which is forced internal) instead of being posted as public "
            "inline diff comments. Use this when inline findings must not be disclosed "
            "publicly. When false (default), inline comments are posted as-is regardless "
            "of project visibility and no extra visibility lookup is performed."
        ),
    )
    inline_findings_title: Optional[str] = Field(
        default=None,
        description=(
            "Heading placed above the folded inline comments when "
            "fold_inline_into_summary_when_public folds them into the summary. "
            "Defaults to a neutral heading if not provided."
        ),
    )


class SubmitMrReview(DuoBaseTool):
    """Submit a complete MR review: inline diff comments, verdict, and optional summary note.

    Uses the Draft Notes API to create comments, then bulk publishes them as a grouped
    review. Optionally sets the reviewer state and posts a summary note in one atomic call.

    Includes line matching fallback: when the LLM provides inaccurate line numbers,
    the tool attempts to find the correct line by matching the suggestion content
    against the actual diff (same algorithm as ProcessCommentsService).
    """

    name: str = "submit_mr_review"
    description: str = (
        "Submit a complete MR review with inline diff comments, a review verdict, "
        "and an optional summary note.\n"
        "Comments are posted as inline diff comments on specific lines.\n"
        "The verdict sets the reviewer state: 'requested_changes' or 'reviewed'.\n"
        "The summary note can be internal (visible only to project members with Reporter+ role) "
        "by setting summary_internal=true.\n"
        "Set fold_inline_into_summary_when_public=true to fold inline comments into an internal "
        "summary on public projects when they must not be disclosed publicly.\n"
        "Example: submit_mr_review(project_id=123, merge_request_iid=45, "
        'comments=[{"file": "app.py", "new_line": 10, "body": "Issue here"}], '
        'verdict="requested_changes", summary="Found 1 issue", summary_internal=true)'
    )
    args_schema: Type[BaseModel] = SubmitMrReviewInput

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        comments: Optional[list[dict]] = None,
        verdict: Optional[str] = None,
        summary: Optional[str] = None,
        summary_internal: bool = False,
        fold_inline_into_summary_when_public: bool = False,
        inline_findings_title: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        logger.info(
            "submit_mr_review called",
            project_id=project_id,
            merge_request_iid=merge_request_iid,
            num_comments=len(comments) if comments else 0,
            verdict=verdict,
            has_summary=bool(summary),
        )

        parsed_comments = []
        for c in comments or []:
            try:
                parsed_comments.append(
                    MrReviewComment(**c) if isinstance(c, dict) else c
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse comment", comment=str(c)[:200], error=str(e)
                )
                raise

        # Optional R11-style guard: only when the caller opts in do we look up project
        # visibility and fold inline comments into an internal summary on public projects.
        # Generic callers leave this off and incur no extra API call.
        if fold_inline_into_summary_when_public and parsed_comments:
            is_public = await self._is_public_project(project_id)
            if is_public:
                logger.info(
                    "Public project — folding inline comments into internal summary",
                    num_comments=len(parsed_comments),
                )
                summary = self._build_public_project_summary(
                    parsed_comments, summary, inline_findings_title
                )
                summary_internal = True
                parsed_comments = []  # skip inline draft notes

        # Step 1 & 2: Create draft notes for inline comments (private projects only).
        # The loop is resilient: a comment that can't be resolved to a diff line or
        # whose draft note fails to post is logged, counted, and skipped — one bad
        # inline comment must not sink an otherwise-complete review.
        posted_count = 0
        failed_count = 0
        if parsed_comments:
            mr_data = await self._fetch_mr_data(project_id, merge_request_iid)
            diff_refs = mr_data.get("diff_refs")
            if not diff_refs:
                raise ToolException("Could not fetch diff_refs from merge request")

            diffs_by_file = await self._fetch_diffs_by_file(
                project_id, merge_request_iid
            )

            for i, comment in enumerate(parsed_comments):
                resolved_comment = self._resolve_comment_position(
                    comment, diffs_by_file
                )
                if not resolved_comment:
                    failed_count += 1
                    logger.warning(
                        "Skipping comment — could not resolve to a valid diff line",
                        index=i,
                        file=comment.file,
                        new_line=comment.new_line,
                    )
                    continue

                logger.info(
                    "Creating draft note",
                    index=i,
                    file=resolved_comment.file,
                    new_line=resolved_comment.new_line,
                    old_line=resolved_comment.old_line,
                )
                try:
                    await self._create_draft_note(
                        project_id, merge_request_iid, diff_refs, resolved_comment
                    )
                    posted_count += 1
                # Deliberately resilient: one comment that fails to post (e.g. a stale
                # diff line) is counted and skipped so it can't sink an otherwise
                # complete review. The failure is surfaced in the returned summary.
                except Exception as e:  # pylint: disable=exception-swallowing-in-tool
                    failed_count += 1
                    logger.warning(
                        "Skipping comment — failed to create draft note",
                        index=i,
                        file=resolved_comment.file,
                        new_line=resolved_comment.new_line,
                        error=str(e),
                    )

        # Step 3: Bulk publish + summary + verdict
        logger.info(
            "Calling bulk_publish",
            verdict=verdict,
            has_summary=bool(summary),
            num_comments=posted_count,
        )
        await self._bulk_publish(
            project_id,
            merge_request_iid,
            verdict,
            summary,
            summary_internal,
        )

        return self._build_response(
            merge_request_iid,
            posted_count,
            failed_count,
            verdict,
            summary,
            summary_internal,
        )

    async def _is_public_project(self, project_id: int) -> bool:
        """Check if the project is public.

        Fails secure: if visibility can't be determined (API error, timeout, etc.)
        we assume the project is public so findings stay in the internal-only note
        rather than being disclosed as public inline comments.
        """
        path = f"/api/v4/projects/{project_id}"
        try:
            response = await self.gitlab_client.aget(path, parse_json=False)
            body = self._process_http_response("fetch project", response, logger)
            project_data = json.loads(body)
            return project_data.get("visibility") == "public"
        except Exception as e:
            logger.error(
                "Could not determine project visibility — failing secure (assuming public)",
                project_id=project_id,
                error=str(e),
            )
            return True

    def _build_public_project_summary(
        self,
        comments: list[MrReviewComment],
        existing_summary: Optional[str],
        title: Optional[str] = None,
    ) -> str:
        """Build a combined summary with inline findings for public projects.

        Since DiffNote comments cannot be internal/confidential, on public projects all
        inline comments are folded into the internal summary note instead. ``title`` is
        the caller-supplied heading placed above the folded comments; it defaults to a
        neutral heading so the tool carries no domain-specific wording.
        """
        heading = title or (
            "**Inline findings** (folded into this internal note because the project is public)"
        )
        findings_section = f"{heading}\n\n"
        for i, c in enumerate(comments, 1):
            line_ref = ""
            if c.new_line:
                line_ref = f":`{c.new_line}`"
            findings_section += f"**{i}.** `{c.file}`{line_ref}\n\n{self._fix_newlines(c.body)}\n\n---\n\n"

        if existing_summary:
            return (
                self._fix_newlines(existing_summary) + "\n\n---\n\n" + findings_section
            )
        return findings_section

    async def _fetch_mr_data(self, project_id: int, merge_request_iid: int) -> dict:
        """Fetch MR metadata including diff_refs."""
        path = f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}"
        response = await self.gitlab_client.aget(path, parse_json=False)
        body = self._process_http_response("fetch merge request", response, logger)
        return json.loads(body)

    async def _fetch_diffs_by_file(
        self, project_id: int, merge_request_iid: int
    ) -> dict[str, list[DiffLine]]:
        """Fetch MR diffs and parse them into DiffLine objects keyed by file path."""
        path = f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/diffs"
        response = await self.gitlab_client.aget(path, parse_json=False)
        body = self._process_http_response(
            "fetch merge request diffs", response, logger
        )
        diffs_data = json.loads(body)

        result: dict[str, list[DiffLine]] = {}
        for diff in diffs_data:
            file_path = diff.get("new_path") or diff.get("old_path")
            raw_diff = diff.get("diff", "")
            if file_path and raw_diff:
                result[file_path] = parse_diff_lines(raw_diff)

        return result

    def _resolve_comment_position(
        self,
        comment: MrReviewComment,
        diffs_by_file: dict[str, list[DiffLine]],
    ) -> Optional[MrReviewComment]:
        """Resolve a comment to the correct diff line, correcting line numbers if needed."""
        diff_lines = diffs_by_file.get(comment.file)
        if not diff_lines:
            logger.warning("File not found in diffs", file=comment.file)
            return None

        matched_line = match_comment_to_diff_line(comment, diff_lines)
        if not matched_line:
            return None

        if (
            matched_line.new_line == comment.new_line
            and matched_line.old_line == comment.old_line
        ):
            return comment

        logger.info(
            "Line matching corrected position",
            file=comment.file,
            original_new_line=comment.new_line,
            resolved_new_line=matched_line.new_line,
            original_old_line=comment.old_line,
            resolved_old_line=matched_line.old_line,
        )

        return MrReviewComment(
            file=comment.file,
            new_line=matched_line.new_line,
            old_line=matched_line.old_line,
            body=comment.body,
            suggestion=comment.suggestion,
        )

    async def _create_draft_note(
        self,
        project_id: int,
        merge_request_iid: int,
        diff_refs: dict,
        comment: MrReviewComment,
    ) -> None:
        """Create a single draft note for an inline comment."""
        body = self._fix_newlines(comment.body)
        if comment.suggestion:
            body += f"\n\n```suggestion:-0+0\n{comment.suggestion}\n```"

        position: dict[str, Any] = {
            "position_type": "text",
            "base_sha": diff_refs["base_sha"],
            "start_sha": diff_refs["start_sha"],
            "head_sha": diff_refs["head_sha"],
            "new_path": comment.file,
            "old_path": comment.file,
        }
        if comment.new_line is not None:
            position["new_line"] = comment.new_line
        if comment.old_line is not None:
            position["old_line"] = comment.old_line

        payload: dict[str, Any] = {
            "note": body,
            "position": position,
        }

        path = f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/draft_notes"
        response = await self.gitlab_client.apost(
            path=path,
            body=json.dumps(payload),
            parse_json=False,
        )
        self._process_http_response("create draft note", response, logger)

    @staticmethod
    def _fix_newlines(text: str) -> str:
        """Fix literal \\n sequences that LLMs sometimes produce instead of actual newlines."""
        return text.replace("\\n", "\n")

    async def _bulk_publish(
        self,
        project_id: int,
        merge_request_iid: int,
        verdict: Optional[str],
        summary: Optional[str],
        summary_internal: bool,
    ) -> None:
        """Bulk publish all draft notes, optionally setting verdict and posting summary.

        The summary is posted as-is; the caller is responsible for any footer or
        disclaimer it wants to include in ``summary``.
        """
        payload: dict[str, Any] = {}
        if verdict:
            payload["reviewer_state"] = verdict
        if summary:
            payload["note"] = self._fix_newlines(summary)
            payload["internal"] = summary_internal

        path = f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/draft_notes/bulk_publish"
        response = await self.gitlab_client.apost(
            path=path,
            body=json.dumps(payload),
            parse_json=False,
        )
        self._process_http_response("bulk publish review", response, logger)

    def _build_response(
        self,
        merge_request_iid: int,
        comment_count: int,
        failed_count: int,
        verdict: Optional[str],
        summary: Optional[str],
        summary_internal: bool,
    ) -> str:
        parts = []
        if comment_count:
            parts.append(f"{comment_count} inline comment(s) posted")
        if failed_count:
            parts.append(f"{failed_count} inline comment(s) could not be posted")
        if verdict:
            parts.append(f"verdict: {verdict}")
        if summary:
            kind = "internal" if summary_internal else "public"
            parts.append(f"{kind} summary note posted")

        detail = ", ".join(parts) if parts else "no actions taken"
        return json.dumps(
            {
                "status": "success",
                "message": f"Review submitted on MR !{merge_request_iid}: {detail}",
            }
        )

    def format_display_message(
        self, args: SubmitMrReviewInput, _tool_response: Any = None
    ) -> str:
        return (
            f"Submit review to merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )
