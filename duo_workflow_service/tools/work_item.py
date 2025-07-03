import json
import urllib
from typing import Any, Literal, NamedTuple, Optional, Type, Union

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.queries.work_items import (
    GET_GROUP_WORK_ITEM_QUERY,
    GET_PROJECT_WORK_ITEM_QUERY,
)

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

    def _validate_parent_url(
        self,
        url: Optional[str],
        group_id: Optional[Union[int, str]],
        project_id: Optional[Union[int, str]],
    ) -> Union[ResolvedParent, str]:
        """Resolve parent information (group or project) from URL or IDs."""
        if url:
            return self._parse_parent_work_item_url(url)
        if group_id:
            return self._create_parent_work_item(
                parent_type="group", identifier=group_id
            )

        if project_id:
            return self._create_parent_work_item(
                parent_type="project", identifier=project_id
            )

        return "Must provide either URL, group_id, or project_id"

    def _validate_work_item_url(
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

        parent = self._validate_parent_url(
            url=None, group_id=group_id, project_id=project_id
        )
        if isinstance(parent, str):
            return parent

        if not work_item_iid:
            return "Must provide work_item_iid if no URL is given"

        return ResolvedWorkItem(parent=parent, work_item_iid=work_item_iid)

    def _decode_path(self, path: str) -> str:
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
            return self._create_parent_work_item(parent_type, path)
        except GitLabUrlParseError as e:
            return f"Failed to parse parent work item URL: {e}"

    def _create_parent_work_item(
        self, parent_type: Literal["group", "project"], identifier: Union[int, str]
    ):
        """Create a ResolvedParent with decoded path."""
        return ResolvedParent(
            type=parent_type, full_path=self._decode_path(str(identifier))
        )

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
        resolved = self._validate_work_item_url(
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

    def format_display_message(self, args: WorkItemResourceInput) -> str:
        if args.url:
            return f"Read work item {args.url}"
        if args.group_id:
            return f"Read work item #{args.work_item_iid} in group {args.group_id}"

        return f"Read work item #{args.work_item_iid} in project {args.project_id}"
