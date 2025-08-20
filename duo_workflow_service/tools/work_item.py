import json
import urllib
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Type, Union

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.queries.work_items import (
    CREATE_WORK_ITEM_MUTATION,
    GET_GROUP_WORK_ITEM_NOTES_QUERY,
    GET_GROUP_WORK_ITEM_QUERY,
    GET_PROJECT_WORK_ITEM_NOTES_QUERY,
    GET_PROJECT_WORK_ITEM_QUERY,
    GET_WORK_ITEM_TYPE_BY_NAME_QUERY,
    LIST_GROUP_WORK_ITEMS_QUERY,
    LIST_PROJECT_WORK_ITEMS_QUERY,
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


class ResolvedParent(NamedTuple):
    type: Literal["group", "project"]
    full_path: str


class ResolvedWorkItem(NamedTuple):
    parent: ResolvedParent
    work_item_iid: int


class WorkItemBaseTool(DuoBaseTool):
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_WORK_ITEM

    async def _validate_parent_url(
        self,
        url: Optional[str],
        group_id: Optional[Union[int, str]],
        project_id: Optional[Union[int, str]],
    ) -> Union[ResolvedParent, str]:
        """Resolve parent information (group or project) from URL or IDs."""
        if url:
            return self._parse_parent_work_item_url(url)
        if group_id:
            return await self._resolve_parent_path(
                parent_type="group", identifier=group_id
            )
        if project_id:
            return await self._resolve_parent_path(
                parent_type="project", identifier=project_id
            )

        return "Must provide either URL, group_id, or project_id"

    async def _validate_work_item_url(
        self,
        url: Optional[str],
        group_id: Optional[Union[int, str]],
        project_id: Optional[Union[int, str]],
        work_item_iid: Optional[int],
    ) -> Union[ResolvedWorkItem, str]:
        """Resolve work item information from URL or IDs."""
        if not work_item_iid and not url:
            return "Must provide work_item_iid if no URL is given"

        if url:
            return self._parse_work_item_url(url)

        parent = await self._validate_parent_url(
            url=None, group_id=group_id, project_id=project_id
        )
        if isinstance(parent, str):
            return parent

        if not work_item_iid:
            return "Must provide work_item_iid if no URL is given"

        return ResolvedWorkItem(parent=parent, work_item_iid=work_item_iid)

    async def _resolve_parent_path(
        self,
        parent_type: Literal["group", "project"],
        identifier: Union[int, str],
    ) -> Union[ResolvedParent, str]:
        identifier_str = str(identifier)

        if identifier_str.isdigit():
            try:
                endpoint = "projects" if parent_type == "project" else "groups"
                data = await self.gitlab_client.aget(
                    f"/api/v4/{endpoint}/{identifier_str}"
                )
                full_path = data.get(
                    "path_with_namespace" if parent_type == "project" else "full_path"
                )
                if not full_path:
                    return f"Could not resolve {parent_type} full path from ID '{identifier_str}'"
            except Exception as e:
                return f"Failed to resolve {parent_type} from ID '{identifier_str}': {str(e)}"
        else:
            full_path = identifier_str

        return ResolvedParent(
            type=parent_type,
            full_path=self._decode_path(full_path),
        )

    @staticmethod
    def _decode_path(path: str) -> str:
        """Make sure the path is safe for GraphQL (i.e., decoded slashes)."""

        return urllib.parse.unquote(path)

    def _parse_parent_work_item_url(self, url: str) -> Union[ResolvedParent, str]:
        """Parse parent work item (by group or project) from URL."""
        try:
            parent_type = GitLabUrlParser.detect_parent_type(url)

            parser_map = {
                "group": GitLabUrlParser.parse_group_url,
                "project": GitLabUrlParser.parse_project_url,
            }

            parsed_url = parser_map.get(parent_type)
            if not parsed_url:
                return f"Unknown parent type: {parent_type}"

            path = parsed_url(url, self.gitlab_host)
            return ResolvedParent(type=parent_type, full_path=self._decode_path(path))
        except GitLabUrlParseError as e:
            return f"Failed to parse parent work item URL: {e}"

    def _parse_work_item_url(self, url: str) -> Union[ResolvedWorkItem, str]:
        """Parse work item from URL."""
        if "/-/work_items/" not in url:
            return "URL is not a work item URL"

        try:
            work_item = GitLabUrlParser.parse_work_item_url(url, self.gitlab_host)

            return ResolvedWorkItem(
                parent=ResolvedParent(
                    type=work_item.parent_type,
                    full_path=self._decode_path(work_item.full_path),
                ),
                work_item_iid=work_item.work_item_iid,
            )
        except GitLabUrlParseError as e:
            return f"Failed to parse work item URL: {e}"

    async def _resolve_work_item_type_id(
        self, full_path: str, type_name: str
    ) -> Union[str, dict]:
        """Returns type ID or error dict."""
        response = await self.gitlab_client.graphql(
            GET_WORK_ITEM_TYPE_BY_NAME_QUERY, {"fullPath": full_path}
        )

        if "errors" in response:
            return {"error": response["errors"]}

        types = response.get("namespace", {}).get("workItemTypes", {}).get("nodes", [])
        match = next((t for t in types if t["name"] == type_name), None)

        if not match:
            available = [t["name"] for t in types]
            return {
                "error": f"Work item type '{type_name}' not found.",
                "available_types": available,
            }

        return match["id"]

    async def _create_work_item_with_type_id(
        self,
        namespace_path: str,
        type_id: str,
        input_kwargs: Dict[str, Any],
    ) -> str:
        variables = {
            "input": {
                "namespacePath": namespace_path,
                "workItemTypeId": type_id,
            }
        }
        variables["input"].update(self._build_work_item_input_fields(input_kwargs))

        response = await self.gitlab_client.graphql(
            CREATE_WORK_ITEM_MUTATION, variables
        )

        if "errors" in response:
            return json.dumps({"error": response["errors"]})

        created = response.get("workItemCreate", {}).get("workItem", {})
        errors = response.get("workItemCreate", {}).get("errors", [])

        if errors or not created.get("id"):
            return json.dumps(
                {
                    "error": "Failed to create work item.",
                    "details": {
                        "graphql_errors": response.get("errors"),
                        "work_item_errors": errors,
                    },
                }
            )

        return json.dumps(
            {
                "message": f"Work item '{created.get('title')}' created successfully.",
                "work_item": created,
            }
        )

    @staticmethod
    def _build_work_item_input_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        input_data = {}
        type_name = kwargs.get("type_name")

        if type_name in ["Issue", "Epic"]:
            start_and_due = {}

            for key in ["start_date", "due_date", "is_fixed"]:
                value = kwargs.get(key)
                if value is not None:
                    graphql_key = "".join(
                        part.capitalize() if i > 0 else part
                        for i, part in enumerate(key.split("_"))
                    )
                    start_and_due[graphql_key] = value

            if start_and_due:
                input_data["startAndDueDateWidget"] = start_and_due

        if kwargs.get("title") is not None:
            input_data["title"] = kwargs["title"]

        if kwargs.get("description") is not None:
            input_data["descriptionWidget"] = {"description": kwargs["description"]}

        if kwargs.get("health_status") is not None and type_name in ["Issue", "Epic"]:
            input_data["healthStatusWidget"] = {"healthStatus": kwargs["health_status"]}

        if kwargs.get("confidential") is not None:
            input_data["confidential"] = kwargs["confidential"]

        if kwargs.get("assignee_ids") is not None:
            input_data["assigneesWidget"] = {
                "assigneeIds": [
                    (
                        assignee
                        if isinstance(assignee, str) and assignee.startswith("gid://")
                        else f"gid://gitlab/User/{assignee}"
                    )
                    for assignee in kwargs["assignee_ids"]
                ]
            }

        if kwargs.get("label_ids") is not None:
            input_data["labelsWidget"] = {
                "labelIds": [
                    (
                        label
                        if isinstance(label, str) and label.startswith("gid://")
                        else f"gid://gitlab/Label/{label}"
                    )
                    for label in kwargs["label_ids"]
                ]
            }

        return input_data


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

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)
        types = kwargs.pop("types", None)

        resolved = await self._validate_parent_url(url, group_id, project_id)
        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        query = (
            LIST_GROUP_WORK_ITEMS_QUERY
            if resolved.type == "group"
            else LIST_PROJECT_WORK_ITEMS_QUERY
        )

        variables = {
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
                variables[key] = value

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
                variables["types"] = valid_types

            if invalid_types:
                warnings.append(
                    f"Some types were invalid and skipped: {', '.join(invalid_types)}"
                )

        try:
            response = await self.gitlab_client.graphql(query, variables)
            root_key = "namespace" if resolved.type == "group" else "project"

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

    async def _arun(self, **kwargs: Any) -> str:
        resolved = await self._validate_work_item_url(
            url=kwargs.get("url"),
            group_id=kwargs.get("group_id"),
            project_id=kwargs.get("project_id"),
            work_item_iid=kwargs.get("work_item_iid"),
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        # Select the appropriate query based on parent type
        query = (
            GET_GROUP_WORK_ITEM_QUERY
            if resolved.parent.type == "group"
            else GET_PROJECT_WORK_ITEM_QUERY
        )

        variables = {
            "fullPath": resolved.parent.full_path,
            "iid": str(resolved.work_item_iid),
        }

        try:
            response = await self.gitlab_client.graphql(query, variables)
            root_key = "namespace" if resolved.parent.type == "group" else "project"

            if root_key not in response:
                return json.dumps({"error": f"No {root_key} found in response"})

            work_items = (
                response.get(root_key, {}).get("workItems", {}).get("nodes", [])
            )
            work_item = work_items[0] if work_items else None

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

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)
        work_item_iid = kwargs.pop("work_item_iid", None)

        resolved = await self._validate_work_item_url(
            url, group_id, project_id, work_item_iid
        )

        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        query = (
            GET_GROUP_WORK_ITEM_NOTES_QUERY
            if resolved.parent.type == "group"
            else GET_PROJECT_WORK_ITEM_NOTES_QUERY
        )

        variables = {
            "fullPath": resolved.parent.full_path,
            "workItemIid": str(resolved.work_item_iid),
        }

        try:
            response = await self.gitlab_client.graphql(query, variables)
            root_key = "namespace" if resolved.parent.type == "group" else "project"
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
    title: str = Field(description="Title of the work item")
    type_name: str = Field(
        description="Work item type. One of: 'Issue', 'Epic', 'Task', 'Objective', 'Key Result'."
    )
    description: Optional[str] = Field(
        default=None, description="The description of the work item."
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

    {PARENT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and title "Implement feature X", the tool call would be:
        create_work_item(group_id='namespace/group', title="Implement feature X", type_name="issue")
    """
    args_schema: Type[BaseModel] = CreateWorkItemInput

    async def _arun(self, type_name: str, **kwargs: Any) -> str:
        kwargs["type_name"] = type_name
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        project_id = kwargs.pop("project_id", None)

        resolved = await self._validate_parent_url(url, group_id, project_id)
        if isinstance(resolved, str):
            return json.dumps({"error": resolved})

        if type_name not in ALL_TYPES:
            supported_types = ", ".join(sorted(ALL_TYPES))
            return json.dumps(
                {
                    "error": f"Unknown work item type: '{type_name}'. "
                    f"Supported types are: {supported_types}."
                }
            )

        if resolved.type == "project" and type_name in GROUP_ONLY_TYPES:
            return json.dumps(
                {
                    "error": f"Work item type '{type_name}' cannot be created in a project – only in groups."
                }
            )

        try:
            type_id_or_error = await self._resolve_work_item_type_id(
                resolved.full_path, type_name
            )
            if isinstance(type_id_or_error, dict):
                return json.dumps(type_id_or_error)

            created = await self._create_work_item_with_type_id(
                namespace_path=resolved.full_path,
                type_id=type_id_or_error,
                input_kwargs=kwargs,
            )
            return created

        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: CreateWorkItemInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Create work item '{args.title}' in {args.url}"
        if args.group_id:
            return f"Create work item '{args.title}' in group {args.group_id}"
        return f"Create work item '{args.title}' in project {args.project_id}"
