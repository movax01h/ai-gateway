import json
from typing import Any, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import (
    DESCRIPTION_CHARACTER_LIMIT,
    DuoBaseTool,
)
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

log = structlog.stdlib.get_logger("workflow")

# API endpoint for merge requests
MERGE_REQUESTS_API_PATH = "/api/v4/projects/{project_id}/merge_requests"

# editorconfig-checker-disable
PROJECT_IDENTIFICATION_DESCRIPTION = """To identify the project you must provide either:
- project_id parameter, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project
  - https://gitlab.com/namespace/project/-/merge_requests
  - https://gitlab.com/group/subgroup/project
  - https://gitlab.com/group/subgroup/project/-/merge_requests
"""

MERGE_REQUEST_IDENTIFICATION_DESCRIPTION = """To identify a merge request you must provide either:
- project_id and merge_request_iid, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project/-/merge_requests/42
  - https://gitlab.com/group/subgroup/project/-/merge_requests/42
"""
# editorconfig-checker-enable


class MergeRequestResourceInput(ProjectResourceInput):
    merge_request_iid: Optional[int] = Field(
        default=None,
        description="The internal ID of the project merge request. Required if URL is not provided.",
    )


class CreateMergeRequestInput(ProjectResourceInput):
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
    labels: Optional[str] = Field(
        default=None,
        description="Comma-separated label names for the merge request. For example: 'bug,feature,high-priority'",
    )


class CreateMergeRequest(DuoBaseTool):
    name: str = "create_merge_request"
    description: str = f"""Create a new merge request in the specified project.

    IMPORTANT: Do NOT include quick actions in the description field. Quick actions are lines starting
    with / (such as /label, /assign, /merge, /milestone) and are not supported for security reasons.
    If you encounter quick actions in a merge request template, remove them and use the dedicated tool
    parameters instead (e.g., use the labels parameter for labels, the assignee parameter for assignments).

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, source_branch "feature", target_branch "main", and title "New feature", the tool call would be:
        create_merge_request(project_id=13, source_branch="feature", target_branch="main", title="New feature")
    - Given the URL https://gitlab.com/namespace/project, source_branch "feature", target_branch "main", and title "New feature", the tool call would be:
        create_merge_request(url="https://gitlab.com/namespace/project", source_branch="feature", target_branch="main", title="New feature")
    """
    args_schema: Type[BaseModel] = CreateMergeRequestInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self,
        source_branch: str,
        target_branch: str,
        title: str,
        **kwargs: Any,
    ) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})
        data = {
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
            "labels",
        ]
        data.update(
            {
                k: kwargs[k]
                for k in optional_params
                if k in kwargs and kwargs[k] is not None
            }
        )

        try:
            path = MERGE_REQUESTS_API_PATH.format(project_id=project_id)
            response = await self.gitlab_client.apost(
                path=path,
                body=json.dumps(data),
            )

            response = self._process_http_response(
                identifier=path,
                response=response,
            )

            return json.dumps({"created_merge_request": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: CreateMergeRequestInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Create merge request from '{args.source_branch}' to '{args.target_branch}' in {args.url}"
        return (
            f"Create merge request from '{args.source_branch}' to '{args.target_branch}' "
            f"in project {args.project_id}"
        )


class GetMergeRequest(DuoBaseTool):
    name: str = "get_merge_request"
    description: str = f"""Fetch details about the merge request.

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        get_merge_request(project_id=13, merge_request_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        get_merge_request(url="https://gitlab.com/namespace/project/-/merge_requests/103")
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
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            path = (
                f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
                f"{validation_result.merge_request_iid}"
            )
            response = await self.gitlab_client.aget(
                path=path,
                parse_json=False,
            )

            if not response.is_success():
                log.error(
                    "Failed to fetch merge request: status_code=%s, response=%s",
                    response.status_code,
                    response.body,
                )

            return json.dumps({"merge_request": response.body})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: MergeRequestResourceInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Read merge request {args.url}"
        return (
            f"Read merge request !{args.merge_request_iid} in project {args.project_id}"
        )


class ListMergeRequestDiffs(DuoBaseTool):
    name: str = "list_merge_request_diffs"
    description: str = f"""Fetch the diffs of the files changed in a merge request.

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        list_merge_request_diffs(project_id=13, merge_request_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        list_merge_request_diffs(url="https://gitlab.com/namespace/project/-/merge_requests/103")
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
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            path = (
                f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
                f"{validation_result.merge_request_iid}/diffs"
            )
            response = await self.gitlab_client.aget(
                path=path,
                parse_json=False,
            )

            if not response.is_success():
                log.error(
                    "Failed to fetch merge request diffs: status_code=%s, response=%s",
                    response.status_code,
                    response.body,
                )

            # Parse the response and apply diff exclusion policy
            diff_data = json.loads(response.body)
            diff_policy = DiffExclusionPolicy(self.project)
            filtered_diff, excluded_files = diff_policy.filter_allowed_diffs(diff_data)

            result: dict[str, Any] = {"diffs": filtered_diff}

            if len(excluded_files) > 0:
                result["excluded_files"] = excluded_files
                result["excluded_reason"] = (
                    DiffExclusionPolicy.format_llm_exclusion_message(excluded_files)
                )

            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: MergeRequestResourceInput, tool_response: Any = None
    ) -> str:
        if args.url:
            msg = f"View changes in merge request {args.url}"
        else:
            msg = f"View changes in merge request !{args.merge_request_iid} in project {args.project_id}"

        if tool_response:
            excluded_files = json.loads(tool_response.content).get("excluded_files")
            return msg + DiffExclusionPolicy.format_user_exclusion_message(
                excluded_files
            )

        return msg


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

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        if not validation_result.project_id or not validation_result.merge_request_iid:
            return json.dumps(
                {"error": "Missing required identifiers after validation"}
            )

        discussion_id = None
        if note_id is not None:
            discussion_result = await self._get_discussion_id_from_note_rest(
                validation_result.project_id,
                "merge_requests",
                validation_result.merge_request_iid,
                note_id,
            )
            if "error" in discussion_result:
                return json.dumps(discussion_result)
            discussion_id = discussion_result.get("discussionId")

        base_path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
            f"{validation_result.merge_request_iid}"
        )
        if discussion_id:
            path = f"{base_path}/discussions/{discussion_id}/notes"
        else:
            path = f"{base_path}/notes"

        payload = {"body": body}

        try:
            response = await self.gitlab_client.apost(
                path=path,
                body=json.dumps(payload),
            )

            response = self._process_http_response(identifier=path, response=response)

            return json.dumps({"created_merge_request_note": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

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
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            path = (
                f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
                f"{validation_result.merge_request_iid}/notes"
            )
            response = await self.gitlab_client.aget(
                path=path,
                parse_json=False,
            )

            if not response.is_success():
                log.error(
                    "Failed to fetch merge request notes: status_code=%s, response=%s",
                    response.status_code,
                    response.body,
                )

            return json.dumps({"notes": response.body})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: MergeRequestResourceInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Read comments on merge request {args.url}"
        return f"Read comments on merge request !{args.merge_request_iid} in project {args.project_id}"


class UpdateMergeRequestInput(MergeRequestResourceInput):
    allow_collaboration: Optional[bool] = Field(
        default=None,
        description="Allow commits from members who can merge to the target branch.",
    )
    assignee_ids: Optional[list[int]] = Field(
        default=None,
        description="The ID of the users to assign the merge request to. Set to 0 or provide an empty value to "
        "unassign all assignees.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the merge request. Limited to 1,048,576 characters.",
    )
    discussion_locked: Optional[bool] = Field(
        default=None,
        description="Flag indicating if the merge request’s discussion is locked. Only project members can add, edit "
        "or resolve notes to locked discussions.",
    )
    milestone_id: Optional[int] = Field(
        default=None,
        description="The global ID of a milestone to assign the merge request to. Set to 0 or provide an empty value "
        "to unassign a milestone.",
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
    labels: Optional[str] = Field(
        default=None,
        description="Comma-separated label names for the merge request. For example: 'bug,feature,high-priority'",
    )


class ListMergeRequestInput(ProjectResourceInput):
    author_username: Optional[str] = Field(
        default=None,
        description="Returns merge requests created by the given username. Mutually exclusive with author_id.",
    )
    author_id: Optional[int] = Field(
        default=None,
        description="Returns merge requests created by the given user ID. Mutually exclusive with author_username.",
    )
    assignee_username: Optional[str] = Field(
        default=None,
        description="Returns merge requests assigned to the given username. Mutually exclusive with assignee_id.",
    )
    assignee_id: Optional[int] = Field(
        default=None,
        description="Returns merge requests assigned to the given user ID. Mutually exclusive with assignee_username.",
    )
    reviewer_username: Optional[str] = Field(
        default=None,
        description="Returns merge requests with the given username as reviewer. Mutually exclusive with reviewer_id.",
    )
    reviewer_id: Optional[int] = Field(
        default=None,
        description="Returns merge requests with the given user ID as reviewer. "
        "Mutually exclusive with reviewer_username.",
    )
    state: Optional[str] = Field(
        default=None,
        description="Filter by state: 'opened', 'closed', 'locked', 'merged', or 'all'.",
    )
    milestone: Optional[str] = Field(
        default=None,
        description="Returns merge requests for a specific milestone. 'None' returns merge requests with no milestone.",
    )
    labels: Optional[str] = Field(
        default=None,
        description="Comma-separated list of label names. Returns merge requests matching all labels.",
    )
    search: Optional[str] = Field(
        default=None,
        description="Search merge requests against their title and description.",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Filter by scope: 'created_by_me', 'assigned_to_me', or 'all'.",
    )


class ListMergeRequest(DuoBaseTool):
    name: str = "gitlab_merge_request_search"
    description: str = f"""List merge requests in a GitLab project.
    This tool supports filtering by author, assignee, reviewer, state, milestone, labels, and more.
    This tool also supports searching for merge requests against their title and description.
    Use this tool when you need to filter or search for merge requests by author or other specific criteria.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - List merge requests by author username:
        gitlab_merge_request_search(project_id=13, author_username="janedoe1337")
    - List merge requests assigned to a specific user:
        gitlab_merge_request_search(project_id=13, assignee_username="janedoe1337")
    - List all open merge requests:
        gitlab_merge_request_search(project_id=13, state="opened")
    - List merge requests with specific labels:
        gitlab_merge_request_search(project_id=13, labels="bug,urgent")
    - Given the URL https://gitlab.com/namespace/project and author filter:
        gitlab_merge_request_search(url="https://gitlab.com/namespace/project", author_username="janedoe1337")
    - Search merge requests against their title and description
        gitlab_merge_request_search(project_id=13, search="bug fix")
    """
    args_schema: Type[BaseModel] = ListMergeRequestInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        # Build query parameters
        params = {}
        optional_params = [
            "author_username",
            "author_id",
            "assignee_username",
            "assignee_id",
            "reviewer_username",
            "reviewer_id",
            "state",
            "milestone",
            "labels",
            "search",
            "scope",
            "updated_after",
        ]

        for param in optional_params:
            if param in kwargs and kwargs.get(param) is not None:
                params[param] = kwargs[param]

        try:
            path = MERGE_REQUESTS_API_PATH.format(project_id=project_id)
            response = await self.gitlab_client.aget(
                path=path,
                params=params,
                parse_json=False,
            )

            if not response.is_success():
                log.error(
                    "Failed to list merge requests: status_code=%s, response=%s",
                    response.status_code,
                    response.body,
                )

            return json.dumps({"merge_requests": response.body})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: ListMergeRequestInput, _tool_response: Any = None
    ) -> str:
        filters = []
        if args.author_username:
            filters.append(f"author: {args.author_username}")
        if args.author_id:
            filters.append(f"author ID: {args.author_id}")
        if args.assignee_username:
            filters.append(f"assignee: {args.assignee_username}")
        if args.assignee_id:
            filters.append(f"assignee ID: {args.assignee_id}")
        if args.reviewer_username:
            filters.append(f"reviewer: {args.reviewer_username}")
        if args.reviewer_id:
            filters.append(f"reviewer ID: {args.reviewer_id}")
        if args.state:
            filters.append(f"state: {args.state}")
        if args.milestone:
            filters.append(f"milestone: {args.milestone}")
        if args.labels:
            filters.append(f"labels: {args.labels}")
        if args.search:
            filters.append(f"search: {args.search}")

        filter_text = f"with filters: {', '.join(filters)}" if filters else ""

        if args.url:
            return f"List merge requests in {args.url} {filter_text}"
        return f"List merge requests in project {args.project_id} {filter_text}"


class UpdateMergeRequest(DuoBaseTool):
    name: str = "update_merge_request"
    # pylint: disable=line-too-long
    description: str = f"""Updates an existing merge request. You can change the target branch, title, or even close the MR.
Max character limit of {DESCRIPTION_CHARACTER_LIMIT} characters.

IMPORTANT: Do NOT include quick actions in the description field. Quick actions are lines starting
with / (such as /label, /assign, /merge, /milestone) and are not supported for security reasons.
If you encounter quick actions in a merge request template, remove them and use the dedicated tool
parameters instead (e.g., use the labels parameter for labels, the assignee parameter for assignments).

{MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

For example:
- Given project_id 13, merge_request_iid 9, and title "Updated title", the tool call would be:
    update_merge_request(project_id=13, merge_request_iid=9, title="Updated title")
- Given the URL https://gitlab.com/namespace/project/-/merge_requests/103 and title "Updated title", the tool call would be:
    update_merge_request(url="https://gitlab.com/namespace/project/-/merge_requests/103", title="Updated title")
    """
    args_schema: Type[BaseModel] = UpdateMergeRequestInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        merge_request_iid = kwargs.pop("merge_request_iid", None)

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        data = {k: v for k, v in kwargs.items() if v is not None}

        try:
            path = (
                f"{MERGE_REQUESTS_API_PATH.format(project_id=validation_result.project_id)}/"
                f"{validation_result.merge_request_iid}"
            )
            response = await self.gitlab_client.aput(
                path=path,
                body=json.dumps(data),
            )

            response = self._process_http_response(
                identifier=path,
                response=response,
            )

            return json.dumps({"updated_merge_request": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: UpdateMergeRequestInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Update merge request {args.url}"
        return f"Update merge request !{args.merge_request_iid} in project {args.project_id}"
