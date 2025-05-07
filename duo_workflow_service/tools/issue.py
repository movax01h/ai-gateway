import json
from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

DESCRIPTION_CHARACTER_LIMIT = 1_048_576

PROJECT_IDENTIFICATION_DESCRIPTION = """To identify the project you must provide either:
- project_id parameter, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project
  - https://gitlab.com/namespace/project/-/issues
  - https://gitlab.com/group/subgroup/project
  - https://gitlab.com/group/subgroup/project/-/issues
"""

ISSUE_IDENTIFICATION_DESCRIPTION = """To identify an issue you must provide either:
- project_id and issue_iid, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project/-/issues/42
  - https://gitlab.com/group/subgroup/project/-/issues/42
"""


class IssueResourceInput(ProjectResourceInput):
    issue_iid: Optional[int] = Field(
        default=None,
        description="The internal ID of the project issue. Required if URL is not provided.",
    )


class IssueBaseTool(DuoBaseTool):
    def _validate_issue_url(
        self, url: Optional[str], project_id: Optional[Any], issue_iid: Optional[int]
    ) -> Tuple[Optional[str], Optional[int], List[str]]:
        """Validate issue URL and extract project_id and issue_iid.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user
            issue_iid: The issue IID provided by the user

        Returns:
            Tuple containing:
                - The validated project_id (or None if validation failed)
                - The validated issue_iid (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is not")
            if not issue_iid:
                errors.append("'issue_iid' must be provided when 'url' is not")
            return project_id, issue_iid, errors

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id, url_issue_iid = GitLabUrlParser.parse_issue_url(
                url, self.gitlab_host
            )

            # If both URL and IDs are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )
            if issue_iid is not None and issue_iid != url_issue_iid:
                errors.append(
                    f"Issue ID mismatch: provided '{issue_iid}' but URL contains '{url_issue_iid}'"
                )

            # Use the IDs from the URL
            return url_project_id, url_issue_iid, errors
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return project_id, issue_iid, errors


class CreateIssueInput(ProjectResourceInput):
    title: str = Field(description="Title of the issue")
    description: Optional[str] = Field(
        default=None,
        description=f"The description of an issue. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters.",
    )
    labels: Optional[str] = Field(
        default=None,
        description="""Comma-separated label names to assign to the new issue.
                                  If a label does not already exist, this creates a new project label and assigns it to the issue.""",
    )
    assignee_ids: Optional[list[int]] = Field(
        default=None, description="The IDs of the users to assign the issue to"
    )
    confidential: Optional[bool] = Field(
        default=None,
        description="Set to true to create a confidential issue. Default is false.",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="The due date. Date time string in the format YYYY-MM-DD, for example 2016-03-11.",
    )
    issue_type: Optional[str] = Field(
        default=None,
        description="The type of issue. One of issue, incident, test_case or task. Default is issue.",
    )


class CreateIssue(IssueBaseTool):
    name: str = "create_issue"
    description: str = f"""Create a new issue in a GitLab project.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and the title "Fix bug in login form", the tool call would be:
        create_issue(project_id=13, title="Fix bug in login form")
    - Given the URL https://gitlab.com/namespace/project and the title "Fix bug in login form", the tool call would be:
        create_issue(url="https://gitlab.com/namespace/project", title="Fix bug in login form")
    """
    args_schema: Type[BaseModel] = CreateIssueInput  # type: ignore

    async def _arun(self, title: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        data = {"title": title, **{k: v for k, v in kwargs.items() if v is not None}}

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/projects/{project_id}/issues",
                body=json.dumps(data),
            )
            return json.dumps({"created_issue": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: CreateIssueInput) -> str:
        if args.url:
            return f"Create issue '{args.title}' in {args.url}"
        return f"Create issue '{args.title}' in project {args.project_id}"


class ListIssuesInput(ProjectResourceInput):
    assignee_id: Optional[int] = Field(
        default=None,
        description="""Return issues assigned to the given user ID. It can't be used together with assignee_username.
                              None returns unassigned issues. Any returns issues with an assignee.""",
    )
    assignee_username: Optional[List[str]] = Field(
        default=None,
        description="""Return issues assigned to the given username. This works like assignee_id but can't be used together with it. In GitLab CE,
                        assignee_username can only have one value. If there's more than one, an error will be returned.""",
    )
    author_id: Optional[int] = Field(
        default=None,
        description="""Return issues created by the given user id.
                                     It can't be used together with author_username.
                                     Combine with scope=all or scope=assigned_to_me.""",
    )
    author_username: Optional[str] = Field(
        default=None,
        description="""Return issues created by the given username.
                                           Similar to author_id and it can't be used together with author_id.
                                           Combine with scope=all or scope=assigned_to_me.""",
    )
    confidential: Optional[bool] = Field(
        default=None, description="Filter confidential or public issues"
    )
    created_after: Optional[str] = Field(
        default=None,
        description="Return issues created on or after the given time. Expected in ISO 8601 date and time format (YYYY-MM-DDTHH:MM:SSZ)",
    )
    created_before: Optional[str] = Field(
        default=None,
        description="Return issues created on or before the given time. Expected in ISO 8601 date and time format (YYYY-MM-DDTHH:MM:SSZ)",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="""Return issues that have no due date, are overdue, or whose due date is this week, this month, or between two weeks ago and next month.
        Accepts: 0 (no due date), any, today, tomorrow, overdue, week, month, next_month_and_previous_two_weeks.""",
    )
    health_status: Optional[str] = Field(
        default=None,
        description="""Return issues with the specified health_status. None returns issues with no
            health status assigned, and Any returns issues with a health status assigned.""",
    )
    issue_type: Optional[str] = Field(
        default=None,
        description="Filter to a given type of issue. One of issue, incident, test_case or task.",
    )
    labels: Optional[str] = Field(
        default=None,
        description="""Comma-separated list of label names, issues must have all labels to be returned.
                                  None lists all issues with no labels. Any lists all issues with at least one label. Predefined names are case-insensitive.""",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Return issues for the given scope: created_by_me, assigned_to_me or all. Defaults to created_by_me.",
    )
    search: Optional[str] = Field(
        default=None, description="Search issues against their title and description"
    )
    sort: Optional[str] = Field(
        default=None,
        description="Return issues sorted in asc or desc order. Default is desc.",
    )
    state: Optional[str] = Field(
        default=None,
        description="Return all issues or just those that are opened or closed",
    )


class ListIssues(IssueBaseTool):
    name: str = "list_issues"
    description: str = f"""List issues in a GitLab project.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, the tool call would be:
        list_issues(project_id=13)
    - Given the URL https://gitlab.com/namespace/project, the tool call would be:
        list_issues(url="https://gitlab.com/namespace/project")
    """
    args_schema: Type[BaseModel] = ListIssuesInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        params = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/issues",
                params=params,
                parse_json=False,
            )
            return json.dumps({"issues": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListIssuesInput) -> str:
        if args.url:
            return f"List issues in {args.url}"
        return f"List issues in project {args.project_id}"


class GetIssue(IssueBaseTool):
    name: str = "get_issue"
    description: str = f"""Get a single issue in a GitLab project.

    {ISSUE_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and issue_iid 9, the tool call would be:
        get_issue(project_id=13, issue_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/issues/103, the tool call would be:
        get_issue(url=https://gitlab.com/namespace/project/-/issues/103)
    """
    args_schema: Type[BaseModel] = IssueResourceInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        issue_iid = kwargs.get("issue_iid")

        project_id, issue_iid, errors = self._validate_issue_url(
            url, project_id, issue_iid
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/issues/{issue_iid}",
                parse_json=False,
            )
            return json.dumps({"issue": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: IssueResourceInput) -> str:
        if args.url:
            return f"Read issue {args.url}"
        return f"Read issue #{args.issue_iid} in project {args.project_id}"


class UpdateIssueInput(IssueResourceInput):
    title: Optional[str] = Field(default=None, description="Title of the issue")
    description: Optional[str] = Field(
        default=None,
        description=f"Description of the issue. Max character limit of {DESCRIPTION_CHARACTER_LIMIT} characters.",
    )
    labels: Optional[str] = Field(
        default=None, description="Comma-separated list of label names"
    )
    assignee_ids: Optional[list[int]] = Field(
        default=None,
        description="The ID of the users to assign the issue to. Set to `0` or provide an empty value to unassign all assignees.",
    )
    confidential: Optional[bool] = Field(
        default=None, description="Set to true to make the issue confidential"
    )
    due_date: Optional[str] = Field(
        default=None, description="Date string in the format YYYY-MM-DD"
    )
    state_event: Optional[str] = Field(
        default=None,
        description="The state event of an issue. To close the issue, use 'close', and to reopen it, use 'reopen'.",
    )
    discussion_locked: Optional[bool] = Field(
        default=None,
        description="Flag indicating if the issue's discussion is locked. If the discussion is locked only project members can add or edit comments.",
    )


class UpdateIssue(IssueBaseTool):
    name: str = "update_issue"
    description: str = f"""Update an existing issue in a GitLab project.

    {ISSUE_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, issue_iid 9, and title "Updated title", the tool call would be:
        update_issue(project_id=13, issue_iid=9, title="Updated title")
    - Given the URL https://gitlab.com/namespace/project/-/issues/103 and title "Updated title", the tool call would be:
        update_issue(url="https://gitlab.com/namespace/project/-/issues/103", title="Updated title")
    """
    args_schema: Type[BaseModel] = UpdateIssueInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        issue_iid = kwargs.pop("issue_iid", None)

        project_id, issue_iid, errors = self._validate_issue_url(
            url, project_id, issue_iid
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        data = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aput(
                path=f"/api/v4/projects/{project_id}/issues/{issue_iid}",
                body=json.dumps(data),
            )
            return json.dumps({"updated_issue": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: UpdateIssueInput) -> str:
        if args.url:
            return f"Update issue {args.url}"
        return f"Update issue #{args.issue_iid} in project {args.project_id}"


class CreateIssueNoteInput(IssueResourceInput):
    body: str = Field(
        description="The content of the note. Limited to 1,000,000 characters."
    )


class CreateIssueNote(IssueBaseTool):
    name: str = "create_issue_note"
    description: str = f"""Create a new note (comment) on a GitLab issue.

    {ISSUE_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, issue_iid 9, and body "This is a comment", the tool call would be:
        create_issue_note(project_id=13, issue_iid=9, body="This is a comment")
    - Given the URL https://gitlab.com/namespace/project/-/issues/103 and body "This is a comment", the tool call would be:
        create_issue_note(url="https://gitlab.com/namespace/project/-/issues/103", body="This is a comment")

    The body parameter is always required.
    """
    args_schema: Type[BaseModel] = CreateIssueNoteInput  # type: ignore

    async def _arun(self, body: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        issue_iid = kwargs.pop("issue_iid", None)

        project_id, issue_iid, errors = self._validate_issue_url(
            url, project_id, issue_iid
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
                body=json.dumps(
                    {
                        "body": body,
                    }
                ),
            )
            return json.dumps({"status": "success", "body": body, "response": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: CreateIssueNoteInput) -> str:
        if args.url:
            return f"Add comment to issue {args.url}"
        return f"Add comment to issue #{args.issue_iid} in project {args.project_id}"


class ListIssueNotesInput(IssueResourceInput):
    sort: Optional[str] = Field(
        default=None,
        description="Return issue notes sorted in asc or desc order. Default is desc",
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Return issue notes ordered by created_at or updated_at fields. Default is created_at",
    )


class ListIssueNotes(IssueBaseTool):
    name: str = "list_issue_notes"
    description: str = f"""Get a list of all notes (comments) for a specific issue.

    {ISSUE_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and issue_iid 9, the tool call would be:
        list_issue_notes(project_id=13, issue_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/issues/103, the tool call would be:
        list_issue_notes(url="https://gitlab.com/namespace/project/-/issues/103")
    """
    args_schema: Type[BaseModel] = ListIssueNotesInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        issue_iid = kwargs.pop("issue_iid", None)

        project_id, issue_iid, errors = self._validate_issue_url(
            url, project_id, issue_iid
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        params = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
                params=params,
                parse_json=False,
            )
            return json.dumps({"notes": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListIssueNotesInput) -> str:
        if args.url:
            return f"Read comments on issue {args.url}"
        return f"Read comments on issue #{args.issue_iid} in project {args.project_id}"


class GetIssueNoteInput(IssueResourceInput):
    note_id: int = Field(description="The ID of the note")


class GetIssueNote(IssueBaseTool):
    name: str = "get_issue_note"
    description: str = f"""Get a single note (comment) from a specific issue.

    {ISSUE_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, issue_iid 9, and note_id 5, the tool call would be:
        get_issue_note(project_id=13, issue_iid=9, note_id=5)
    - Given the URL https://gitlab.com/namespace/project/-/issues/103 and note_id 42, the tool call would be:
        get_issue_note(url="https://gitlab.com/namespace/project/-/issues/103", note_id=42)

    The note_id parameter is always required.
    """
    args_schema: Type[BaseModel] = GetIssueNoteInput  # type: ignore

    async def _arun(self, note_id: int, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        issue_iid = kwargs.pop("issue_iid", None)

        project_id, issue_iid, errors = self._validate_issue_url(
            url, project_id, issue_iid
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/issues/{issue_iid}/notes/{note_id}",
                parse_json=False,
            )
            return json.dumps({"note": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: GetIssueNoteInput) -> str:
        if args.url:
            return f"Read comment #{args.note_id} on issue {args.url}"
        return f"Read comment #{args.note_id} on issue #{args.issue_iid} in project {args.project_id}"
