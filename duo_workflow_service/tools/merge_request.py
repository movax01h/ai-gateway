import json
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

DESCRIPTION_CHARACTER_LIMIT = 1_048_576


class CreateMergeRequestInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    source_branch: str = Field(description="The source branch name")
    target_branch: str = Field(description="The target branch name")
    title: str = Field(description="Title of the merge request")
    description: Optional[str] = Field(
        default=None,
        description=f"Description of the merge request. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters.",
    )
    assignee_ids: Optional[list[int]] = Field(
        default=None, description="The ID of the users to assign the merge request to"
    )
    reviewer_ids: Optional[list[int]] = Field(
        default=None, description="The ID of the users to request a review from"
    )
    remove_source_branch: Optional[bool] = Field(
        default=None,
        description="Flag indicating if a merge request should remove the source branch when merging",
    )
    squash: Optional[bool] = Field(
        default=None,
        description="Flag indicating if the merge request should squash commits when merging",
    )


class CreateMergeRequest(DuoBaseTool):
    name: str = "create_merge_request"
    description: str = """Create a new merge request in the specified project"""
    args_schema: Type[BaseModel] = CreateMergeRequestInput

    async def _arun(
        self,
        project_id: int,
        source_branch: str,
        target_branch: str,
        title: str,
        **kwargs: Any,
    ) -> str:
        data: dict[str, Any] = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title,
        }

        optional_params = [
            "description",
            "assignee_ids",
            "reviewer_ids",
            "remove_source_branch",
            "squash",
        ]
        data.update({k: kwargs[k] for k in optional_params if k in kwargs})

        response = await self.gitlab_client.apost(
            path=f"/api/v4/projects/{project_id}/merge_requests",
            body=json.dumps(data),
        )

        return json.dumps({"status": "success", "data": data, "response": response})

    def format_display_message(self, args: CreateMergeRequestInput) -> str:
        return f"Create merge request from '{args.source_branch}' to '{args.target_branch}' in project {args.project_id}"


class GetMergeRequestInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_iid: int = Field(description="Id of the merge request")


class GetMergeRequest(DuoBaseTool):
    name: str = "get_merge_request"
    description: str = """Fetch details about the merge request"""
    args_schema: Type[BaseModel] = GetMergeRequestInput  # type: ignore

    async def _arun(self, project_id: int, merge_request_iid: int) -> str:
        return await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}",
            parse_json=False,
        )

    def format_display_message(self, args: GetMergeRequestInput) -> str:
        return (
            f"Read merge request !{args.merge_request_iid} in project {args.project_id}"
        )


class ListMergeRequestDiffsInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_iid: int = Field(description="Id of the merge request")


class ListMergeRequestDiffs(DuoBaseTool):
    name: str = "list_merge_request_diffs"
    description: str = """Fetch the diffs of the files changed in a merge request"""
    args_schema: Type[BaseModel] = ListMergeRequestDiffsInput  # type: ignore

    async def _arun(self, project_id: int, merge_request_iid: int) -> str:
        return await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/diffs",
            parse_json=False,
        )

    def format_display_message(self, args: ListMergeRequestDiffsInput) -> str:
        return f"View changes in merge request !{args.merge_request_iid} in project {args.project_id}"


# The merge_request_diff_head_sha parameter is required for the /merge quick action.
# We exclude it here as an added precautionary layer to prevent Duo Workflow from merging code without human approval.
class CreateMergeRequestNoteInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_iid: int = Field(description="Id of the merge request")
    body: str = Field(
        description="The content of a note. Limited to 1,000,000 characters."
    )


class CreateMergeRequestNote(DuoBaseTool):
    name: str = "create_merge_request_note"
    description: str = """Create a note (comment) on a merge request. You are NOT allowed to ever use a GitLab quick action in a merge request note.
                    Quick actions are text-based shortcuts for common GitLab actions. They are commands that are on their own line and
                    start with a backslash. Examples include /merge, /approve, /close, etc."""
    args_schema: Type[BaseModel] = CreateMergeRequestNoteInput  # type: ignore

    def _contains_quick_action(self, body: str) -> bool:
        quick_action_pattern = r"(?m)^/[a-zA-Z]+"
        return bool(re.search(quick_action_pattern, body))

    async def _arun(self, project_id: int, merge_request_iid: int, body: str) -> str:
        if self._contains_quick_action(body):
            return json.dumps(
                {
                    "status": "error",
                    "message": """Notes containing GitLab quick actions are not allowed. Quick actions are text-based shortcuts for common GitLab actions.
                                  They are commands that are on their own line and start with a backslash. Examples include /merge, /approve, /close, etc.""",
                }
            )

        response = await self.gitlab_client.apost(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/notes",
            body=json.dumps(
                {
                    "body": body,
                },
            ),
        )
        return json.dumps({"status": "success", "body": body, "response": response})

    def format_display_message(self, args: CreateMergeRequestNoteInput) -> str:
        return f"Add comment to merge request !{args.merge_request_iid} in project {args.project_id}"


class ListAllMergeRequestNotesInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_iid: int = Field(description="Id of the merge request")


class ListAllMergeRequestNotes(DuoBaseTool):
    name: str = "list_all_merge_request_notes"
    description: str = """List all notes (comments) on a merge request.

    DO NOT use this tool to get issue notes/comments (use list_issue_notes).
    """
    args_schema: Type[BaseModel] = ListAllMergeRequestNotesInput  # type: ignore

    async def _arun(self, project_id: int, merge_request_iid: int) -> str:
        return await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}/notes",
            parse_json=False,
        )

    def format_display_message(self, args: ListAllMergeRequestNotesInput) -> str:
        return f"Read comments on merge request !{args.merge_request_iid} in project {args.project_id}"


class UpdateMergeRequestInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_iid: int = Field(description="Id of the merge request")
    allow_collaboration: Optional[bool] = Field(
        default=None,
        description="Allow commits from members who can merge to the target branch.",
    )
    assignee_ids: Optional[list[int]] = Field(
        default=None,
        description="The ID of the users to assign the merge request to. Set to 0 or provide an empty value to unassign all assignees.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the merge request. Limited to 1,048,576 characters.",
    )
    discussion_locked: Optional[bool] = Field(
        default=None,
        description="Flag indicating if the merge request’s discussion is locked. Only project members can add, edit or resolve notes to locked discussions.",
    )
    milestone_id: Optional[int] = Field(
        default=None,
        description="The global ID of a milestone to assign the merge request to. Set to 0 or provide an empty value to unassign a milestone.",
    )
    remove_source_branch: Optional[bool] = Field(
        default=None,
        description="Flag indicating if a merge request should remove the source branch when merging.",
    )
    reviewer_ids: Optional[list[int]] = Field(
        default=None,
        description="The ID of the users to request a review from. Set to an empty value to unassign all reviewers.",
    )
    squash: Optional[bool] = Field(
        default=None,
        description="Flag indicating if the merge request should squash commits when merging.",
    )
    state_event: Optional[str] = Field(
        default=None,
        description="The state event of the merge request. Set to close to close the merge request.",
    )
    target_branch: Optional[str] = Field(
        default=None, description="The target branch of the merge request."
    )
    title: Optional[str] = Field(
        default=None, description="The title of the merge request."
    )


class UpdateMergeRequest(DuoBaseTool):
    name: str = "update_merge_request"
    description: str = f"""Updates an existing merge request. You can change the target branch, title, or even close the MR.
    Max character limit of {DESCRIPTION_CHARACTER_LIMIT} characters."""
    args_schema: Type[BaseModel] = UpdateMergeRequestInput

    async def _arun(
        self, project_id: int, merge_request_iid: int, **kwargs: Any
    ) -> str:
        data = {k: v for k, v in kwargs.items() if v is not None}

        return await self.gitlab_client.aput(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_iid}",
            body=json.dumps(data),
        )

    def format_display_message(self, args: UpdateMergeRequestInput) -> str:
        return f"Update merge request !{args.merge_request_iid} in project {args.project_id}"
