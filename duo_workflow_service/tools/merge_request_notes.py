"""Tools for creating and listing notes (comments) on merge requests."""

import json
import re
from typing import Any, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.merge_request import (
    MERGE_REQUEST_IDENTIFICATION_DESCRIPTION,
    MERGE_REQUESTS_API_PATH,
    MergeRequestResourceInput,
)

log = structlog.stdlib.get_logger("workflow")

__all__ = [
    "CreateMergeRequestDiffNote",
    "CreateMergeRequestDiffNoteInput",
    "CreateMergeRequestNote",
    "CreateMergeRequestNoteInput",
    "ListAllMergeRequestNotes",
]

SUGGESTION_BLOCK_RE = re.compile(
    r"```suggestion(?::-(\d+)\+(\d+))?[ \t]*\n(.*?)(?:\r?\n)?```", re.DOTALL
)
HUNK_HEADER_RE = re.compile(r"@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")


class CreateMergeRequestNoteInput(MergeRequestResourceInput):
    body: str = Field(
        description="The content of a note. Limited to 1,000,000 characters."
    )
    note_id: Optional[int] = Field(
        default=None,
        description="ID of an existing note to reply to. "
        "The tool will automatically find the discussion containing this note. "
        "If not provided, creates a standalone comment.",
    )
    internal: bool = Field(
        default=False,
        description="If true, creates an internal note visible only to project members "
        "with at least the Reporter role.",
    )


class CreateMergeRequestNote(DuoBaseTool):
    name: str = "create_merge_request_note"
    description: str = f"""Create a note (comment) on a merge request.

IMPORTANT: Do NOT include quick actions in the body field. Quick actions are lines starting with /
(such as /label, /assign, /merge, /milestone) and are not supported for security reasons. If you
encounter quick actions in content, simply remove them.

{MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

For example:
- Given project_id 13, merge_request_iid 9, and body "This is a comment", the tool call would be:
    create_merge_request_note(project_id=13, merge_request_iid=9, body="This is a comment")
- Given the URL https://gitlab.com/namespace/project/-/merge_requests/103 and body "This is a comment", the tool call would be:
    create_merge_request_note(url="https://gitlab.com/namespace/project/-/merge_requests/103", body="This is a comment")

The body parameter is always required.
"""
    args_schema: Type[BaseModel] = CreateMergeRequestNoteInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, body: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        merge_request_iid = kwargs.pop("merge_request_iid", None)
        note_id = kwargs.pop("note_id", None)
        internal = kwargs.pop("internal", False)

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        if not validation_result.project_id or not validation_result.merge_request_iid:
            raise ToolException("Missing required identifiers after validation")

        discussion_id = None
        if note_id is not None:
            discussion_result = await self._get_discussion_id_from_note_rest(
                validation_result.project_id,
                "merge_requests",
                validation_result.merge_request_iid,
                note_id,
            )
            discussion_id = discussion_result.get("discussionId")

        base_path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
            f"{validation_result.merge_request_iid}"
        )
        if discussion_id:
            path = f"{base_path}/discussions/{discussion_id}/notes"
        else:
            path = f"{base_path}/notes"

        payload: dict[str, Any] = {"body": body}
        if internal:
            payload["internal"] = True

        response = await self.gitlab_client.apost(
            path=path,
            body=json.dumps(payload),
        )

        response = self._process_http_response(identifier=path, response=response)

        return json.dumps({"created_merge_request_note": response})

    def format_display_message(
        self, args: CreateMergeRequestNoteInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Add comment to merge request {args.url}"
        return f"Add comment to merge request !{args.merge_request_iid} in project {args.project_id}"


class ListAllMergeRequestNotes(DuoBaseTool):
    name: str = "list_all_merge_request_notes"
    description: str = f"""List all notes (comments) on a merge request.

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        list_all_merge_request_notes(project_id=13, merge_request_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        list_all_merge_request_notes(url="https://gitlab.com/namespace/project/-/merge_requests/103")
    """
    args_schema: Type[BaseModel] = MergeRequestResourceInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        merge_request_iid = kwargs.get("merge_request_iid")

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
            f"{validation_result.merge_request_iid}/notes"
        )
        notes = await self._paginate_get(path)
        return json.dumps({"notes": notes})

    def format_display_message(
        self, args: MergeRequestResourceInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Read comments on merge request {args.url}"
        return f"Read comments on merge request !{args.merge_request_iid} in project {args.project_id}"


class CreateMergeRequestDiffNoteInput(MergeRequestResourceInput):
    """Input schema for posting an inline diff note on a merge request."""

    body: str = Field(
        description="The content of the inline diff note. Supports GitLab markdown, "
        "including multi-line suggestion blocks using ```suggestion:-0+0 syntax."
    )
    old_path: str = Field(
        description="The file path before the change (same as new_path for non-renamed files)."
    )
    new_path: str = Field(
        description="The file path after the change (same as old_path for non-renamed files)."
    )
    old_line: Optional[int] = Field(
        default=None,
        description="Line number in the old version of the file. "
        "Required for comments on deleted or unchanged lines. "
        "Do not set for comments on added lines.",
    )
    new_line: Optional[int] = Field(
        default=None,
        description="Line number in the new version of the file. "
        "Required for comments on added or unchanged lines. "
        "Do not set for comments on deleted lines.",
    )


class CreateMergeRequestDiffNote(DuoBaseTool):
    """Post an inline note on a specific line in a merge request diff.

    This tool creates a discussion thread positioned on a specific line in the MR diff, enabling inline code suggestions
    without depending on Duo Code Review. It uses the standard GitLab Discussions API.
    """

    name: str = "create_merge_request_diff_note"
    description: str = f"""Post an inline note on a specific line in a merge request diff.

This creates a discussion thread positioned on a diff line, useful for posting
code suggestions directly in the MR diff view. The note body supports GitLab
markdown including suggestion blocks.

To post a code suggestion, use this markdown syntax in the body:
```suggestion:-0+0
corrected line content
```

The position is determined by old_line and new_line:
- For added lines (green in diff): set new_line only
- For deleted lines (red in diff): set old_line only
- For unchanged/context lines: set both old_line and new_line

The tool automatically fetches the required diff refs (base_sha, head_sha, start_sha)
from the merge request.

{MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

For example:
- Post a suggestion on an added line:
    create_merge_request_diff_note(
        project_id=13, merge_request_iid=9,
        body="```suggestion:-0+0\ncorrected code\n```",
        old_path="src/main.py", new_path="src/main.py",
        new_line=42
    )
"""
    args_schema: Type[BaseModel] = CreateMergeRequestDiffNoteInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self, body: str, old_path: str, new_path: str, **kwargs: Any
    ) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        merge_request_iid = kwargs.pop("merge_request_iid", None)
        old_line = kwargs.pop("old_line", None)
        new_line = kwargs.pop("new_line", None)

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        if not validation_result.project_id or not validation_result.merge_request_iid:
            raise ToolException("Missing required identifiers after validation")

        if old_line is None and new_line is None:
            raise ToolException(
                "At least one of old_line or new_line must be provided."
            )

        try:
            diff_refs = await self._fetch_diff_refs(
                validation_result.project_id, validation_result.merge_request_iid
            )
        except Exception as e:
            raise ToolException(str(e)) from e

        blocks = self._extract_suggestion_blocks(body)
        if blocks:
            file_diff = await self._fetch_file_diff(
                validation_result.project_id,
                validation_result.merge_request_iid,
                old_path=old_path,
                new_path=new_path,
            )
            if file_diff is not None and self._all_blocks_are_no_ops(
                blocks,
                file_diff,
                old_line=old_line,
                new_line=new_line,
            ):
                raise ToolException(
                    "every suggestion in this comment is identical to the targeted line(s); refusing to post no-op"
                )

        position: dict[str, Any] = {
            "position_type": "text",
            "base_sha": diff_refs["base_sha"],
            "head_sha": diff_refs["head_sha"],
            "start_sha": diff_refs["start_sha"],
            "old_path": old_path,
            "new_path": new_path,
        }

        if old_line is not None:
            position["old_line"] = old_line
        if new_line is not None:
            position["new_line"] = new_line

        payload: dict[str, Any] = {
            "body": body,
            "position": position,
        }

        try:
            path = (
                f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
                f"{validation_result.merge_request_iid}/discussions"
            )
            response = await self.gitlab_client.apost(
                path=path,
                body=json.dumps(payload),
            )

            response = self._process_http_response(identifier=path, response=response)

            return json.dumps({"created_diff_note": response})
        except Exception as e:
            raise ToolException(str(e)) from e

    async def _fetch_diff_refs(
        self, project_id: str, merge_request_iid: int
    ) -> dict[str, str]:
        """Fetch base_sha, head_sha, and start_sha from the merge request."""
        path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=project_id)}/"
            f"{merge_request_iid}"
        )
        response = await self.gitlab_client.aget(path=path, parse_json=False)

        if not response.is_success():
            log.error(
                "Failed to fetch merge request: status_code=%s, response=%s",
                response.status_code,
                response.body,
            )
            raise ToolException(
                f"Failed to fetch merge request diff refs (status_code={response.status_code})"
            )

        mr_data = json.loads(response.body)
        diff_refs = mr_data.get("diff_refs")

        if not diff_refs:
            log.error(
                "diff_refs not available on merge request: project_id=%s, merge_request_iid=%s",
                project_id,
                merge_request_iid,
            )
            raise ToolException("diff_refs not available on merge request")

        return {
            "base_sha": diff_refs["base_sha"],
            "head_sha": diff_refs["head_sha"],
            "start_sha": diff_refs["start_sha"],
        }

    @staticmethod
    def _extract_suggestion_blocks(body: str) -> list[tuple[int, int, list[str]]]:
        """Parse every ``suggestion:-X+Y`` block in the body.

        Returns one ``(before, after, replacement_lines)`` tuple per block, or an
        empty list if the body contains no suggestion blocks.
        """
        return [
            (int(before or 0), int(after or 0), content.split("\n"))
            for before, after, content in SUGGESTION_BLOCK_RE.findall(body)
        ]

    def _all_blocks_are_no_ops(
        self,
        blocks: list[tuple[int, int, list[str]]],
        file_diff: str,
        *,
        old_line: Optional[int],
        new_line: Optional[int],
    ) -> bool:
        """Return True only when every block matches its targeted range exactly."""
        for before, after, suggestion_lines in blocks:
            target_lines = self._lines_at_range(
                file_diff,
                target_old=old_line,
                target_new=new_line,
                before=before,
                after=after,
            )
            if target_lines is None or suggestion_lines != target_lines:
                return False
        return True

    async def _fetch_file_diff(
        self,
        project_id: str,
        merge_request_iid: int,
        *,
        old_path: str,
        new_path: str,
    ) -> Optional[str]:
        """Return the unified diff for ``old_path``/``new_path``, paginating through /diffs."""
        path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=project_id)}/"
            f"{merge_request_iid}/diffs"
        )
        try:
            diffs = await self._paginate_get(path)
        except Exception:  # pylint: disable=broad-except
            return None
        for entry in diffs:
            if not isinstance(entry, dict):
                continue
            if entry.get("new_path") == new_path or entry.get("old_path") == old_path:
                diff_text = entry.get("diff")
                return diff_text if diff_text else None
        return None

    @staticmethod
    def _lines_at_range(
        diff: str,
        *,
        target_old: Optional[int],
        target_new: Optional[int],
        before: int,
        after: int,
    ) -> Optional[list[str]]:
        """Extract ``before + 1 + after`` consecutive lines around the target from a unified diff.

        ``target_new`` takes precedence when both are set. Returns None if the
        target is undefined or any line in the range is outside the diff's
        context window.
        """
        use_new = target_new is not None
        target = target_new if use_new else target_old
        if target is None or target - before < 1:
            return None
        line_map: dict[int, str] = {}
        old_n = new_n = 0
        for line in diff.splitlines():
            header = HUNK_HEADER_RE.match(line)
            if header:
                old_n, new_n = int(header.group(1)), int(header.group(2))
                continue
            prefix, content = line[:1], line[1:]
            if prefix in ("+", " ") and use_new:
                line_map[new_n] = content
            elif prefix in ("-", " ") and not use_new:
                line_map[old_n] = content
            if prefix in ("+", " "):
                new_n += 1
            if prefix in ("-", " "):
                old_n += 1
        wanted = range(target - before, target + after + 1)
        if any(n not in line_map for n in wanted):
            return None
        return [line_map[n] for n in wanted]

    def format_display_message(
        self,
        args: CreateMergeRequestDiffNoteInput,
        _tool_response: Any = None,
    ) -> str:
        line_info = (
            f"new_line={args.new_line}"
            if args.new_line
            else f"old_line={args.old_line}"
        )
        if args.url:
            return (
                f"Add inline diff note to {args.new_path}:{line_info} "
                f"in merge request {args.url}"
            )
        return (
            f"Add inline diff note to {args.new_path}:{line_info} "
            f"in merge request !{args.merge_request_iid} in project {args.project_id}"
        )
