import json
from enum import Enum
from typing import Annotated, Any, List, Optional, Type, Union

from pydantic import BaseModel, Field, StringConstraints

from duo_workflow_service.security.quick_actions import validate_no_quick_actions
from duo_workflow_service.tools.duo_base_tool import DESCRIPTION_CHARACTER_LIMIT
from duo_workflow_service.tools.work_items.base_tool import (
    ResolvedWorkItem,
    WorkItemBaseTool,
)
from duo_workflow_service.tools.work_items.queries.work_items import (
    CREATE_NOTE_MUTATION,
)

# Supported work item types in GitLab
GROUP_ONLY_TYPES = {"Epic", "Objective", "Key Result"}
ALL_TYPES = {"Issue", "Task", *GROUP_ONLY_TYPES}

PARENT_IDENTIFICATION_DESCRIPTION = """To identify the parent (group or project) you must provide either:
- group_id parameter, or
- project_id parameter, or
- A GitLab URL like:
    - https://gitlab.com/namespace/group
    - https://gitlab.com/groups/namespace/group
    - https://gitlab.com/namespace/project
    - https://gitlab.com/namespace/group/project
"""


WORK_ITEM_IDENTIFICATION_DESCRIPTION = """To identify a work item you must provide either:
- group_id/project_id and work_item_iid
    - group_id can be either a numeric ID (e.g., 42) or a path string (e.g., 'my-group' or 'namespace/subgroup')
    - project_id can be either a numeric ID (e.g., 13) or a path string (e.g., 'namespace/project')
    - work_item_iid is always a numeric value (e.g., 7)
- or a GitLab URL like:
    - https://gitlab.com/groups/namespace/group/-/work_items/42
    - https://gitlab.com/namespace/project/-/work_items/42
"""

WORK_ITEM_QUICK_ACTION_NOTE = """You are NOT allowed to ever use a GitLab quick action in work item description
or work item note body. Quick actions are text-based shortcuts for common GitLab actions. They are commands that are
on their own line and start with a backslash. Examples include /merge, /approve, /close, etc."""

DateString = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class HealthStatus(str, Enum):
    ON_TRACK = "onTrack"
    NEEDS_ATTENTION = "needsAttention"
    AT_RISK = "atRisk"


class ParentResourceInput(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="GitLab URL for the resource. If provided, other ID fields are not required.",
    )
    group_id: Optional[Union[int, str]] = Field(
        default=None,
        description="The ID or URL-encoded path of the group. Required if URL and project_id are not provided.",
    )
    project_id: Optional[Union[int, str]] = Field(
        default=None,
        description="The ID or URL-encoded path of the project. Required if URL and group_id are not provided.",
    )


class ListWorkItemsInput(ParentResourceInput):
    state: Optional[str] = Field(
        default=None,
        description="Filter by work item state (e.g., 'opened', 'closed', 'all'). If not set, all states are included.",
    )
    search: Optional[str] = Field(
        default=None, description="Search for work items by title or description."
    )
    author_username: Optional[str] = Field(
        default=None, description="Filter by username of the author."
    )
    created_after: Optional[str] = Field(
        default=None,
        description="Include only work items created on or after this date (ISO 8601 format).",
    )
    created_before: Optional[str] = Field(
        default=None,
        description="Include only work items created on or before this date (ISO 8601 format).",
    )
    updated_after: Optional[str] = Field(
        default=None,
        description="Include only work items updated on or after this date (ISO 8601 format).",
    )
    updated_before: Optional[str] = Field(
        default=None,
        description="Include only work items updated on or before this date (ISO 8601 format).",
    )
    due_after: Optional[str] = Field(
        default=None,
        description="Include only work items due on or after this date (ISO 8601 format).",
    )
    due_before: Optional[str] = Field(
        default=None,
        description="Include only work items due on or before this date (ISO 8601 format).",
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort results by field and direction (e.g., 'CREATED_DESC', 'UPDATED_ASC').",
    )
    first: Optional[int] = Field(
        default=20,
        description="Number of work items to return per page (max 100).",
        le=100,
        ge=1,
    )
    after: Optional[str] = Field(
        default=None,
        description="Cursor for pagination. Use endCursor from a previous response.",
    )
    types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Filter by work item types. Must be one of: "
            + ", ".join(sorted(type.upper().replace(" ", "_") for type in ALL_TYPES))
        ),
    )


class ListWorkItems(WorkItemBaseTool):
    name: str = "list_work_items"
    description: str = f"""List work items in a GitLab project or group.
    By default, only returns the first 20 work items. Use 'after' parameter with the
    endCursor from previous responses to fetch subsequent pages.

    {PARENT_IDENTIFICATION_DESCRIPTION}

    This tool only supports the following types: ({', '.join(sorted(ALL_TYPES))})

    For example:
    - Given group_id 'namespace/group', the tool call would be:
        list_work_items(group_id='namespace/group')
    - Given project_id 'namespace/project', the tool call would be:
        list_work_items(project_id='namespace/project')
    - Given the URL https://gitlab.com/groups/namespace/group, the tool call would be:
        list_work_items(url="https://gitlab.com/groups/namespace/group")
    - Given the URL https://gitlab.com/namespace/project, the tool call would be:
        list_work_items(url="https://gitlab.com/namespace/project")
    """
    args_schema: Type[BaseModel] = ListWorkItemsInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)
        types = kwargs.pop("types", None)

        resolved = await self._validate_parent_url(url, group_id, project_id)
        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        query, root_key = self._LIST_WORK_ITEMS_QUERIES[resolved.type]

        query_variables = {
            "fullPath": resolved.full_path,
            "first": kwargs.get("first"),
            "after": kwargs.get("after"),
        }

        # Handle optional filters
        for key in [
            "state",
            "search",
            "authorUsername",
            "createdAfter",
            "createdBefore",
            "updatedAfter",
            "updatedBefore",
            "dueAfter",
            "dueBefore",
            "sort",
        ]:
            arg_key = key[0].lower() + key[1:]  # match Pydantic input
            value = kwargs.get(arg_key)
            if value is not None:
                query_variables[key] = value

        warnings = []

        if types:
            normalized_input = [type.upper().replace(" ", "_") for type in types]
            valid_types = [
                type
                for type in normalized_input
                if type in {type.upper().replace(" ", "_") for type in ALL_TYPES}
            ]
            invalid_types = [
                type
                for type in normalized_input
                if type not in {type.upper().replace(" ", "_") for type in ALL_TYPES}
            ]

            if valid_types:
                query_variables["types"] = valid_types

            if invalid_types:
                warnings.append(
                    f"Some types were invalid and skipped: {', '.join(invalid_types)}"
                )

        try:
            response = await self.gitlab_client.graphql(query, query_variables)

            if root_key not in response:
                return json.dumps({"error": f"No {root_key} found in response"})

            work_items_data = response[root_key].get("workItems", {})
            result = {
                "work_items": work_items_data.get("nodes", []),
                "page_info": work_items_data.get("pageInfo", {}),
            }
            if warnings:
                result["warnings"] = warnings
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: ListWorkItemsInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"List work items in {args.url}"
        if args.group_id:
            return f"List work items in group {args.group_id}"

        return f"List work items in project {args.project_id}"


class WorkItemResourceInput(ParentResourceInput):
    work_item_iid: Optional[int] = Field(
        default=None,
        description="The internal ID of the work item. Required if URL is not provided.",
    )


class GetWorkItem(WorkItemBaseTool):
    name: str = "get_work_item"
    description: str = f"""Get a single work item in a GitLab group or project.

    {WORK_ITEM_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and work_item_iid 42, the tool call would be:
        get_work_item(group_id='namespace/group', work_item_iid=42)
    - Given project_id 'namespace/project' and work_item_iid 42, the tool call would be:
        get_work_item(project_id='namespace/project', work_item_iid=42)
    - Given the URL https://gitlab.com/groups/namespace/group/-/work_items/42, the tool call would be:
        get_work_item(url="https://gitlab.com/groups/namespace/group/-/work_items/42")
    - Given the URL https://gitlab.com/namespace/project/-/work_items/42, the tool call would be:
        get_work_item(url="https://gitlab.com/namespace/project/-/work_items/42")
    """
    args_schema: Type[BaseModel] = WorkItemResourceInput

    async def _execute(self, **kwargs: Any) -> str:
        resolved = await self._validate_work_item_url(
            url=kwargs.get("url"),
            group_id=kwargs.get("group_id"),
            project_id=kwargs.get("project_id"),
            work_item_iid=kwargs.get("work_item_iid"),
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        try:
            if (work_item := await self._get_work_item_data(resolved)) is None:
                return json.dumps({"error": "Work item not found"})

            if work_item.get("error"):
                return json.dumps(work_item)

            return json.dumps({"work_item": work_item})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: WorkItemResourceInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Read work item {args.url}"
        if args.group_id:
            return f"Read work item #{args.work_item_iid} in group {args.group_id}"

        return f"Read work item #{args.work_item_iid} in project {args.project_id}"


class GetWorkItemNotesInput(WorkItemResourceInput):
    sort: Optional[str] = Field(
        default=None,
        description="Return work item notes sorted in asc or desc order. Default is desc.",
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Return work item notes ordered by created_at or updated_at fields. Default is created_at",
    )


class GetWorkItemNotes(WorkItemBaseTool):
    name: str = "get_work_item_notes"
    description: str = f"""Get all comments (notes) for a specific work item.

    {WORK_ITEM_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and work_item_iid 42, the tool call would be:
        get_work_item_notes(group_id='namespace/group', work_item_iid=42)
    - Given project_id 'namespace/project' and work_item_iid 42, the tool call would be:
        get_work_item_notes(project_id='namespace/project', work_item_iid=42)
    - Given the URL https://gitlab.com/groups/namespace/group/-/work_items/42, the tool call would be:
        get_work_item_notes(url="https://gitlab.com/groups/namespace/group/-/work_items/42")
    - Given the URL https://gitlab.com/namespace/project/-/work_items/42, the tool call would be:
        get_work_item_notes(url="https://gitlab.com/namespace/project/-/work_items/42")
    """
    args_schema: Type[BaseModel] = GetWorkItemNotesInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)
        work_item_iid = kwargs.pop("work_item_iid", None)

        resolved = await self._validate_work_item_url(
            url, group_id, project_id, work_item_iid
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        query, root_key = self._GET_WORK_ITEM_NOTES_QUERIES[resolved.parent.type]

        query_variables = {
            "fullPath": resolved.parent.full_path,
            "workItemIid": str(resolved.work_item_iid),
        }

        try:
            response = await self.gitlab_client.graphql(query, query_variables)
            nodes = response.get(root_key, {}).get("workItems", {}).get("nodes", [])

            if not nodes:
                return json.dumps({"error": "No work item found."})

            widgets = nodes[0].get("widgets", [])
            for widget in widgets:
                if "notes" in widget:
                    notes = widget.get("notes", {}).get("nodes", [])
                    return json.dumps({"notes": notes}, indent=2)

            return json.dumps({"notes": []})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: GetWorkItemNotesInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Read comments on work item {args.url}"
        if args.group_id:
            return f"Read comments on work item #{args.work_item_iid} in group {args.group_id}"

        return f"Read comments on work item #{args.work_item_iid} in project {args.project_id}"


class CreateWorkItemInput(ParentResourceInput):
    title: str = Field(description="Title of the work item.")
    type_name: str = Field(
        description="Work item type. One of: 'Issue', 'Epic', 'Task', 'Objective', 'Key Result'."
    )
    description: Optional[str] = Field(
        default=None,
        description=f"The description of the work item. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters.",
    )
    assignee_ids: Optional[List[int]] = Field(
        default=None, description="IDs of users to assign"
    )
    label_ids: Optional[List[str]] = Field(
        default=None, description="IDs of labels to assign"
    )
    confidential: Optional[bool] = Field(
        default=None, description="Set to true to create a confidential work item."
    )
    start_date: Optional[str] = Field(
        default=None, description="Start date in YYYY-MM-DD format."
    )
    due_date: Optional[str] = Field(
        default=None, description="Due date in YYYY-MM-DD format."
    )
    is_fixed: Optional[bool] = Field(
        default=None, description="Whether the start and due dates are fixed."
    )
    health_status: Optional[str] = Field(
        default=None,
        description="Health status: 'onTrack', 'needsAttention', 'atRisk'.",
    )
    state: Optional[str] = Field(
        default=None, description="Work item state. Use 'opened' or 'closed'."
    )


class CreateWorkItem(WorkItemBaseTool):
    name: str = "create_work_item"
    description: str = f"""Create a new work item in a GitLab group or project.

    {WORK_ITEM_QUICK_ACTION_NOTE}

    {PARENT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and title "Implement feature X", the tool call would be:
        create_work_item(group_id='namespace/group', title="Implement feature X", type_name="issue")
    """
    args_schema: Type[BaseModel] = CreateWorkItemInput

    async def _execute(self, type_name: str, **kwargs: Any) -> str:
        kwargs["type_name"] = type_name
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)

        resolved = await self._validate_parent_url(url, group_id, project_id)
        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        return await self._create_work_item(resolved, type_name, kwargs)

    def format_display_message(
        self, args: CreateWorkItemInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Create work item '{args.title}' in {args.url}"
        if args.group_id:
            return f"Create work item '{args.title}' in group {args.group_id}"
        return f"Create work item '{args.title}' in project {args.project_id}"


class UpdateWorkItemInput(WorkItemResourceInput):
    title: Optional[str] = Field(default=None, description="Title of the work item")
    description: Optional[str] = Field(
        default=None, description="Description of the work item."
    )
    assignee_ids: Optional[List[int]] = Field(
        default=None, description="IDs of users to assign."
    )
    confidential: Optional[bool] = Field(
        default=None, description="Set to true to make the work item confidential."
    )
    start_date: Optional[DateString] = Field(
        default=None,
        description="The start date. Date time string in the format YYYY-MM-DD.",
    )
    due_date: Optional[DateString] = Field(
        default=None,
        description="The due date. Date time string in the format YYYY-MM-DD.",
    )
    is_fixed: Optional[bool] = Field(
        default=None, description="Whether the start and due dates are fixed."
    )
    health_status: Optional[HealthStatus] = Field(
        default=None,
        description="Health status of the work item. Values: 'onTrack', 'needsAttention', 'atRisk'.",
    )
    state: Optional[str] = Field(
        default=None,
        description="The state of the work item. Use 'opened' or 'closed'.",
    )
    add_label_ids: Optional[List[str]] = Field(
        default=None,
        description="Label global IDs or numeric IDs to add to the work item.",
    )
    remove_label_ids: Optional[List[str]] = Field(
        default=None,
        description="Label global IDs or numeric IDs to remove from the work item.",
    )


class UpdateWorkItem(WorkItemBaseTool):
    name: str = "update_work_item"
    description: str = f"""Update an existing work item in a GitLab group or project.

    {WORK_ITEM_QUICK_ACTION_NOTE}

    {WORK_ITEM_IDENTIFICATION_DESCRIPTION}

    For example:
    - update_work_item(group_id='parent/child', work_item_iid=42, title="Updated title")
    - update_work_item(project_id='namespace/project', work_item_iid=42, title="Updated title")
    - update_work_item(url="https://gitlab.com/groups/namespace/group/-/work_items/42", title="Updated title")
    - update_work_item(url="https://gitlab.com/namespace/project/-/work_items/42", title="Updated title")
    """
    args_schema: Type[BaseModel] = UpdateWorkItemInput

    async def _execute(self, **kwargs: Any) -> str:
        resolved = await self._resolve_work_item_data(
            url=kwargs.get("url"),
            group_id=kwargs.get("group_id"),
            project_id=kwargs.get("project_id"),
            work_item_iid=kwargs.get("work_item_iid"),
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        return await self._update_work_item(resolved, kwargs)

    def format_display_message(
        self, args: UpdateWorkItemInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Update work item in {args.url}"
        if args.group_id:
            return f"Update work item #{args.work_item_iid} in group {args.group_id}"
        return f"Update work item #{args.work_item_iid} in project {args.project_id}"


class CreateWorkItemNoteInput(WorkItemResourceInput):
    body: str = Field(
        description=f"The content of the note. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters.",
        max_length=1_048_576,
    )
    internal: Optional[bool] = Field(
        default=None, description="Internal flag for a note. Default is false."
    )
    discussion_id: Optional[str] = Field(
        default=None, description="Global ID of the discussion the note is in reply to."
    )


class CreateWorkItemNote(WorkItemBaseTool):
    name: str = "create_work_item_note"
    description: str = f"""Create a new note (comment) on a GitLab work item.

    {WORK_ITEM_QUICK_ACTION_NOTE}

    {WORK_ITEM_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group', work_item_iid 42, and body "This is a comment", the tool call would be:
        create_work_item_note(group_id='namespace/group', work_item_iid=42, body="This is a comment")
    - Given project_id 'namespace/project', work_item_iid 42, and body "This is a comment", the tool call would be:
        create_work_item_note(project_id='namespace/project', work_item_iid=42, body="This is a comment")
    - Given the URL https://gitlab.com/groups/namespace/group/-/work_items/42 and body "This is a comment", the tool call would be:
        create_work_item_note(url="https://gitlab.com/groups/namespace/group/-/work_items/42", body="This is a comment")
    - Given the URL https://gitlab.com/namespace/project/-/work_items/42 and body "This is a comment", the tool call would be:
        create_work_item_note(url="https://gitlab.com/namespace/project/-/work_items/42", body="This is a comment")

    The body parameter is always required.
    """
    args_schema: Type[BaseModel] = CreateWorkItemNoteInput

    async def _execute(self, body: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)
        work_item_iid = kwargs.pop("work_item_iid", None)
        internal = kwargs.pop("internal", None)
        discussion_id = kwargs.pop("discussion_id", None)

        if err := validate_no_quick_actions(body, field="body"):
            return json.dumps({"error": err})

        resolved = await self._validate_work_item_url(
            url, group_id, project_id, work_item_iid
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        try:
            if "error" in (result := await self._get_work_item_id(resolved)):
                return json.dumps(result)

            note_input = {"noteableId": result["id"], "body": body}

            if internal is not None:
                note_input["internal"] = internal

            if discussion_id is not None:
                note_input["discussionId"] = discussion_id

            note_response = await self.gitlab_client.graphql(
                CREATE_NOTE_MUTATION, {"input": note_input}
            )

            return self._process_note_response(note_response)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_work_item_id(self, resolved: ResolvedWorkItem) -> dict:
        """Get work item ID from resolved work item."""
        try:
            work_item = await self._get_work_item_data(resolved)
            if isinstance(work_item, dict) and "error" in work_item:
                return work_item

            if not work_item:
                return {"error": "Work item not found"}
            if not work_item.get("id"):
                return {"error": "Work item exists but has no ID field"}

            return {"id": work_item["id"]}

        except Exception as e:
            return {"error": f"Failed to get work item ID: {str(e)}"}

    def _process_note_response(self, note_response: dict) -> str:
        """Process the GraphQL response from creating a note."""
        # Top-level GraphQL errors (e.g., auth, syntax, variables)
        if top_errors := note_response.get("errors"):
            return json.dumps({"error": top_errors})

        create_note = note_response.get("createNote", {})
        created_note = create_note.get("note", {})
        note_errors = create_note.get("errors", [])

        # Application-level errors (mutation ran but failed validation)
        if note_errors or not created_note.get("id"):
            return json.dumps(
                {
                    "error": "Failed to create note",
                    "details": {
                        "graphql_errors": top_errors,
                        "note_errors": note_errors,
                    },
                }
            )

        return json.dumps(
            {
                "status": "success",
                "message": "Note created successfully.",
                "note": created_note,
            }
        )

    def format_display_message(
        self, args: CreateWorkItemNoteInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Add comment to work item {args.url}"
        if args.group_id:
            return f"Add comment to work item #{args.work_item_iid} in group {args.group_id}"
        return f"Add comment to work item #{args.work_item_iid} in project {args.project_id}"
