"""Tools for managing MR discussion threads: list, reply, and resolve."""

import json
from typing import Any, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

logger = structlog.stdlib.get_logger(__name__)

NOTE_BODY_LIMIT = 2000


def _truncate_note_body(body: str) -> str:
    """Cap a note body, making any truncation explicit so the LLM doesn't treat a cut-off note as the complete text."""
    if len(body) <= NOTE_BODY_LIMIT:
        return body

    dropped = len(body) - NOTE_BODY_LIMIT
    return (
        body[:NOTE_BODY_LIMIT]
        + f"\n\n...<TRUNCATED: {dropped} CHARACTERS DROPPED DUE TO SIZE LIMIT>"
    )


class ListMrDiscussionsInput(BaseModel):
    """List discussions on a merge request."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    only_from_author: Optional[str] = Field(
        default=None,
        description="Filter to discussions started by this username (e.g., 'duo-security-reviewer')",
    )
    only_resolvable: bool = Field(
        default=False,
        description="If true, only return resolvable discussions (inline diff comments)",
    )


class ListMrDiscussions(DuoBaseTool):
    """List discussion threads on a merge request.

    Returns discussions with their notes, resolution state, and positions. Useful for checking prior review comments
    before posting new ones.
    """

    name: str = "list_mr_discussions"
    description: str = (
        "List discussion threads on a merge request. "
        "Returns each discussion's ID, resolution state, position (file and line), "
        "and the text of all notes in the thread. "
        "Use only_from_author to filter to a specific reviewer's threads. "
        "Use only_resolvable=true to get only inline diff comment threads."
    )
    args_schema: Type[BaseModel] = ListMrDiscussionsInput

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        only_from_author: Optional[str] = None,
        only_resolvable: bool = False,
        **kwargs: Any,
    ) -> str:
        path = f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/discussions"
        # Fetch every page: GitLab paginates discussions (default 20 per page), so a
        # single request would hide the reviewer's prior threads on large MRs and lead
        # to duplicate findings being re-posted. See gitlab-org/gitlab#603791.
        discussions = await self._paginate_get(path)

        result = []
        for d in discussions:
            notes = d.get("notes", [])
            if not notes:
                continue

            first_note = notes[0]

            if only_resolvable and not first_note.get("resolvable"):
                continue

            if (
                only_from_author
                and first_note.get("author", {}).get("username") != only_from_author
            ):
                continue

            position = first_note.get("position")
            discussion_info: dict[str, Any] = {
                "discussion_id": d["id"],
                "resolved": first_note.get("resolved", False),
                "author": first_note.get("author", {}).get("username"),
            }

            if position:
                discussion_info["file"] = position.get("new_path")
                discussion_info["new_line"] = position.get("new_line")
                discussion_info["old_line"] = position.get("old_line")

            discussion_info["notes"] = [
                {
                    "id": n["id"],
                    "author": n.get("author", {}).get("username"),
                    "body": _truncate_note_body(n["body"]),
                }
                for n in notes
            ]

            result.append(discussion_info)

        return json.dumps(result)

    def format_display_message(
        self, args: ListMrDiscussionsInput, _tool_response: Any = None
    ) -> str:
        return (
            f"List discussions on merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )


class ReplyToDiscussionInput(BaseModel):
    """Reply to an existing discussion thread."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    discussion_id: str = Field(description="The discussion ID to reply to")
    body: str = Field(description="The reply text in markdown")


class ReplyToDiscussion(DuoBaseTool):
    """Reply to an existing discussion thread on a merge request.

    Use this to respond to prior review findings — e.g., to note that a finding was addressed, or to explain why a fix
    attempt is insufficient.
    """

    name: str = "reply_to_discussion"
    description: str = (
        "Reply to an existing discussion thread on a merge request. "
        "Provide the discussion_id and a markdown body. "
        "Use this to respond to prior review findings."
    )
    args_schema: Type[BaseModel] = ReplyToDiscussionInput

    @staticmethod
    def _fix_newlines(text: str) -> str:
        """Fix literal \\n sequences that LLMs sometimes produce instead of actual newlines."""
        return text.replace("\\n", "\n")

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        discussion_id: str,
        body: str,
        **kwargs: Any,
    ) -> str:
        path = (
            f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}"
            f"/discussions/{discussion_id}/notes"
        )
        payload = {"body": self._fix_newlines(body)}
        response = await self.gitlab_client.apost(
            path=path,
            body=json.dumps(payload),
            parse_json=False,
        )
        result = self._process_http_response("reply to discussion", response, logger)
        note = json.loads(result)

        return json.dumps(
            {
                "status": "success",
                "note_id": note.get("id"),
                "message": f"Replied to discussion {discussion_id[:12]}",
            }
        )

    def format_display_message(
        self, args: ReplyToDiscussionInput, _tool_response: Any = None
    ) -> str:
        return f"Reply to discussion {args.discussion_id[:12]} on MR !{args.merge_request_iid}"


class SetDiscussionResolvedInput(BaseModel):
    """Set the resolved status of a discussion thread."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    discussion_id: str = Field(description="The discussion ID to resolve/unresolve")
    resolved: bool = Field(
        default=True,
        description="True to resolve the discussion, False to unresolve it",
    )


class SetDiscussionResolved(DuoBaseTool):
    """Set the resolved status of a discussion thread on a merge request.

    Use this after replying to a finding that has been fully addressed to mark the thread as resolved.
    """

    name: str = "set_discussion_resolved"
    description: str = (
        "Set the resolved status of a discussion thread on a merge request. "
        "Use resolved=true to mark a finding as addressed, "
        "resolved=false to reopen it."
    )
    args_schema: Type[BaseModel] = SetDiscussionResolvedInput

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        discussion_id: str,
        resolved: bool = True,
        **kwargs: Any,
    ) -> str:
        path = (
            f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}"
            f"/discussions/{discussion_id}"
        )
        payload = {"resolved": resolved}
        response = await self.gitlab_client.aput(
            path=path,
            body=json.dumps(payload),
        )
        self._process_http_response("set discussion resolved", response, logger)

        action = "resolved" if resolved else "unresolved"
        return json.dumps(
            {
                "status": "success",
                "message": f"Discussion {discussion_id[:12]} {action}",
            }
        )

    def format_display_message(
        self, args: SetDiscussionResolvedInput, _tool_response: Any = None
    ) -> str:
        action = "Resolve" if args.resolved else "Unresolve"
        return f"{action} discussion {args.discussion_id[:12]} on MR !{args.merge_request_iid}"
