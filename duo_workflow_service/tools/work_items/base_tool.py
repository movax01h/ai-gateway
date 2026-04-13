import json
import re
import urllib
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    override,
)

import structlog
from langchain_core.tools import ToolException
from pydantic import StringConstraints

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.work_items.queries.work_items import (
    CREATE_WORK_ITEM_MUTATION,
    GET_GROUP_WORK_ITEM_NOTES_QUERY,
    GET_GROUP_WORK_ITEM_QUERY,
    GET_PROJECT_WORK_ITEM_NOTES_QUERY,
    GET_PROJECT_WORK_ITEM_QUERY,
    GET_WORK_ITEM_TYPE_BY_NAME_QUERY,
    LIST_GROUP_WORK_ITEMS_QUERY,
    LIST_PROJECT_WORK_ITEMS_QUERY,
    UPDATE_WORK_ITEM_MUTATION,
)
from duo_workflow_service.tools.work_items.version_compatibility import (
    get_query_variables_for_version,
)

log = structlog.stdlib.get_logger(__name__)

# Supported work item types in GitLab
GROUP_ONLY_TYPES = {"Epic", "Objective", "Key Result"}
ALL_TYPES = {"Issue", "Task", *GROUP_ONLY_TYPES}
STATE_EVENT_MAPPING = {
    "closed": "CLOSE",
    "opened": "REOPEN",
}


class ResolvedParent(NamedTuple):
    type: Literal["group", "project"]
    full_path: str


class ResolvedWorkItem(NamedTuple):
    parent: ResolvedParent
    full_path: Optional[str] = None
    work_item_iid: Optional[int] = None
    id: Optional[str] = None
    full_data: Optional[dict] = None


class HealthStatus(str, Enum):
    ON_TRACK = "onTrack"
    NEEDS_ATTENTION = "needsAttention"
    AT_RISK = "atRisk"


DateString = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")]


# Mapping from work item type to licensed feature enum value for tier checks.
# Values must match GitlabSubscriptions::LicensedFeatureEnum GraphQL enum.
_TYPE_TO_FEATURE = {
    "EPIC": "EPICS",
    "OBJECTIVE": "OKRS",
    "KEY_RESULT": "OKRS",
}


class WorkItemBaseTool(DuoBaseTool):
    _GET_WORK_ITEM_QUERIES = {
        "group": (GET_GROUP_WORK_ITEM_QUERY, "namespace"),
        "project": (GET_PROJECT_WORK_ITEM_QUERY, "project"),
    }

    _GET_WORK_ITEM_NOTES_QUERIES = {
        "group": (GET_GROUP_WORK_ITEM_NOTES_QUERY, "namespace"),
        "project": (GET_PROJECT_WORK_ITEM_NOTES_QUERY, "project"),
    }

    _LIST_WORK_ITEMS_QUERIES = {
        "group": (LIST_GROUP_WORK_ITEMS_QUERY, "namespace"),
        "project": (LIST_PROJECT_WORK_ITEMS_QUERY, "project"),
    }

    async def _validate_parent_url(
        self,
        url: Optional[str],
        group_id: Optional[Union[int, str]],
        project_id: Optional[Union[int, str]],
    ) -> ResolvedParent:
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

        raise ToolException("Must provide either URL, group_id, or project_id")

    async def _validate_work_item_url(
        self,
        url: Optional[str],
        group_id: Optional[Union[int, str]],
        project_id: Optional[Union[int, str]],
        work_item_iid: Optional[int],
    ) -> ResolvedWorkItem:
        """Resolve work item information from URL or IDs."""
        if not work_item_iid and not url:
            raise ToolException("Must provide work_item_iid if no URL is given")

        if url:
            return self._parse_work_item_url(url)

        parent = await self._validate_parent_url(
            url=None, group_id=group_id, project_id=project_id
        )

        return ResolvedWorkItem(parent=parent, work_item_iid=work_item_iid)

    async def _resolve_parent_path(
        self,
        parent_type: Literal["group", "project"],
        identifier: Union[int, str],
    ) -> ResolvedParent:
        full_path = await self._resolve_identifier_to_path(str(identifier), parent_type)
        return ResolvedParent(type=parent_type, full_path=full_path)

    @staticmethod
    def _decode_path(path: str) -> str:
        """Make sure the path is safe for GraphQL (i.e., decoded slashes)."""

        return urllib.parse.unquote(path)

    def _parse_parent_work_item_url(self, url: str) -> ResolvedParent:
        """Parse parent work item (by group or project) from URL."""
        try:
            parent_type = GitLabUrlParser.detect_parent_type(url)

            parser_map = {
                "group": GitLabUrlParser.parse_group_url,
                "project": GitLabUrlParser.parse_project_url,
            }

            parsed_url = parser_map.get(parent_type)
            if not parsed_url:
                raise ToolException(f"Unknown parent type: {parent_type}")

            path = parsed_url(url, self.gitlab_host)
            return ResolvedParent(type=parent_type, full_path=self._decode_path(path))
        except GitLabUrlParseError as e:
            raise ToolException(f"Failed to parse parent work item URL: {e}")

    def _parse_work_item_url(self, url: str) -> ResolvedWorkItem:
        """Parse work item from URL."""
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
            raise ToolException(f"Failed to parse work item URL: {e}")

    async def _resolve_work_item_type_id(self, full_path: str, type_name: str) -> str:
        """Returns type ID or raises ToolException."""
        response = await self.gitlab_client.graphql(
            GET_WORK_ITEM_TYPE_BY_NAME_QUERY, {"fullPath": full_path}
        )

        if "errors" in response:
            raise ToolException(f"GraphQL errors: {json.dumps(response['errors'])}")

        types = response.get("namespace", {}).get("workItemTypes", {}).get("nodes", [])
        match = next((t for t in types if t["name"] == type_name), None)

        if not match:
            available = [t["name"] for t in types]
            raise ToolException(
                f"Work item type '{type_name}' not found. Available types: {', '.join(available)}"
            )

        return match["id"]

    @staticmethod
    def _build_work_item_input_fields(
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
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

        warnings: List[str] = []

        assignees_widget = WorkItemBaseTool._build_assignees_widget(kwargs, warnings)

        if assignees_widget:
            input_data["assigneesWidget"] = assignees_widget

        labels_widget = WorkItemBaseTool._build_labels_widget(kwargs, warnings)

        if labels_widget:
            input_data["labelsWidget"] = labels_widget

        hierarchy_widget = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        if hierarchy_widget:
            input_data["hierarchyWidget"] = hierarchy_widget

        return input_data, warnings

    @staticmethod
    def _build_assignees_widget(
        kwargs: Dict[str, Any], warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Build assignees widget configuration for work item operations.

        Args:
            kwargs: Input parameters that may contain assignee_ids
            warnings: List to collect validation warnings

        Returns:
            Dictionary with assignees widget configuration or None if not provided
        """
        if kwargs.get("assignee_ids") is None:
            return None

        valid_ids, invalid_ids = WorkItemBaseTool._normalize_gids(
            kwargs["assignee_ids"], "User"
        )

        if invalid_ids:
            warnings.append(
                f"Some assignee_ids were invalid and skipped: {invalid_ids}"
            )

        if valid_ids:
            return {"assigneeIds": valid_ids}

        return None

    @staticmethod
    def _build_labels_widget(
        kwargs: Dict[str, Any], warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        widget = {}

        # For work item creation, use labelIds
        if kwargs.get("label_ids"):
            valid_labels, invalid_labels = WorkItemBaseTool._normalize_gids(
                kwargs["label_ids"], "Label"
            )
            if valid_labels:
                widget["labelIds"] = valid_labels
            if invalid_labels:
                warnings.append(
                    f"Some label_ids were invalid and skipped: {invalid_labels}"
                )

        # For work item updates, use addLabelIds and removeLabelIds
        if kwargs.get("add_label_ids"):
            valid_add, invalid_add = WorkItemBaseTool._normalize_gids(
                kwargs["add_label_ids"], "Label"
            )
            if valid_add:
                widget["addLabelIds"] = valid_add
            if invalid_add:
                warnings.append(
                    f"Some add_label_ids were invalid and skipped: {invalid_add}"
                )

        if kwargs.get("remove_label_ids"):
            valid_remove, invalid_remove = WorkItemBaseTool._normalize_gids(
                kwargs["remove_label_ids"], "Label"
            )
            if valid_remove:
                widget["removeLabelIds"] = valid_remove
            if invalid_remove:
                warnings.append(
                    f"Some remove_label_ids were invalid and skipped: {invalid_remove}"
                )

        return widget

    @staticmethod
    def _build_hierarchy_widget(
        kwargs: Dict[str, Any], warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Build hierarchy widget configuration for work item operations.

        Args:
            kwargs: Input parameters that may contain hierarchy_widget
            warnings: List to collect validation warnings

        Returns:
            Dictionary with hierarchy widget configuration or None if not provided
        """
        hierarchy_widget = kwargs.get("hierarchy_widget")

        if not hierarchy_widget:
            return None

        if not isinstance(hierarchy_widget, dict):
            warnings.append("hierarchy_widget must be a dictionary")
            return None

        parent_id = hierarchy_widget.get("parent_id")
        if not parent_id:
            warnings.append("hierarchy_widget must contain 'parent_id' key")
            return None

        # Validate and normalize the parent_id to proper GitLab GID format
        if not re.match(r"^gid://gitlab/WorkItem/\d+$", parent_id):
            warnings.append(
                f"Invalid parent_id format: {parent_id}. Expected GitLab GID."
            )
            return None

        return {"parentId": parent_id}  # Note: GraphQL expects camelCase

    @staticmethod
    def _normalize_gids(ids: list[Any], gid_type: str) -> tuple[list[str], list[Any]]:
        """Return (valid GIDs, invalid entries) for given user or label IDs."""
        valid = []
        invalid = []

        prefix = f"gid://gitlab/{gid_type}/"

        for value in ids:
            if not value:
                continue
            if isinstance(value, str) and value.startswith("gid://"):
                valid.append(value)
            elif isinstance(value, (int, str)) and str(value).isdigit():
                valid.append(f"{prefix}{value}")
            else:
                invalid.append(value)

        return valid, invalid

    async def _resolve_work_item_data(
        self,
        *,
        url: Optional[str],
        group_id: Optional[str],
        project_id: Optional[str],
        work_item_iid: Optional[int],
    ) -> ResolvedWorkItem:
        resolved = await self._validate_work_item_url(
            url=url,
            group_id=group_id,
            project_id=project_id,
            work_item_iid=work_item_iid,
        )

        return await self._fetch_work_item_data(resolved)

    async def _fetch_work_item_data(
        self, resolved: ResolvedWorkItem
    ) -> ResolvedWorkItem:
        query = (
            GET_GROUP_WORK_ITEM_QUERY
            if resolved.parent.type == "group"
            else GET_PROJECT_WORK_ITEM_QUERY
        )

        variables = {
            "fullPath": resolved.parent.full_path,
            "iid": str(resolved.work_item_iid),
            **get_query_variables_for_version(
                "includeHierarchyWidget", "includeDevelopmentWidget"
            ),
        }

        response = await self.gitlab_client.graphql(query, variables)
        if not isinstance(response, dict):
            raise ToolException("GraphQL query returned no response or invalid format")

        root_key = "namespace" if resolved.parent.type == "group" else "project"

        if root_key not in response:
            raise ToolException(f"No {root_key} found in response")

        work_items = response.get(root_key, {}).get("workItems", {}).get("nodes", [])
        work_item = work_items[0] if work_items else None

        if not work_item:
            raise ToolException(f"Work item {resolved.work_item_iid} not found")

        work_item_id = work_item.get("id")
        if not work_item_id:
            raise ToolException("Could not find work item ID")

        return ResolvedWorkItem(
            id=work_item_id,
            full_data=work_item,
            parent=resolved.parent,
            work_item_iid=resolved.work_item_iid,
        )

    async def _create_work_item(self, resolved, type_name: str, kwargs: dict) -> str:
        if type_name not in ALL_TYPES:
            supported_types = ", ".join(sorted(ALL_TYPES))
            raise ToolException(
                f"Unknown work item type: '{type_name}'. "
                f"Supported types are: {supported_types}."
            )

        if resolved.type == "project" and type_name in GROUP_ONLY_TYPES:
            raise ToolException(
                f"Work item type '{type_name}' cannot be created in a project – only in groups."
            )

        return await self._execute_create_work_item(
            namespace_path=resolved.full_path,
            input_kwargs=kwargs,
            type_name=type_name,
        )

    async def _execute_create_work_item(
        self,
        namespace_path: str,
        input_kwargs: Dict[str, Any],
        type_name: str,
    ) -> str:
        type_id = await self._resolve_work_item_type_id(namespace_path, type_name)

        input_fields, warnings = self._build_work_item_input_fields(input_kwargs)
        variables = {
            "input": {
                "namespacePath": namespace_path,
                "workItemTypeId": type_id,
                **input_fields,
            }
        }

        response = await self.gitlab_client.graphql(
            CREATE_WORK_ITEM_MUTATION, variables
        )

        if "errors" in response:
            raise ToolException(f"GraphQL errors: {json.dumps(response['errors'])}")

        created = response.get("workItemCreate", {}).get("workItem", {})
        errors = response.get("workItemCreate", {}).get("errors", [])

        if errors or not created.get("id"):
            raise ToolException(
                f"Failed to create work item. GraphQL errors: {response.get('errors')}, "
                f"Work item errors: {errors}"
            )

        result = {
            "message": f"Work item '{created.get('title')}' created successfully.",
            "work_item": created,
        }
        if warnings:
            result["warnings"] = warnings
        return json.dumps(result)

    async def _update_work_item(self, resolved, kwargs: dict) -> str:
        work_item_id = resolved.id

        if not kwargs.get("type_name"):
            kwargs["type_name"] = (
                (resolved.full_data or {}).get("workItemType", {}).get("name", "")
            )

        input_fields, warnings = self._build_work_item_input_fields(kwargs)

        state = kwargs.get("state")
        if state in STATE_EVENT_MAPPING:
            input_fields["stateEvent"] = STATE_EVENT_MAPPING[state]

        variables = {
            "input": {
                "id": work_item_id,
                **input_fields,
            }
        }

        response = await self.gitlab_client.graphql(
            UPDATE_WORK_ITEM_MUTATION, variables
        )

        if "errors" in response:
            raise ToolException(f"GraphQL errors: {json.dumps(response['errors'])}")

        updated = response.get("data", {}).get("workItemUpdate", {}).get("workItem", {})
        result = {"updated_work_item": updated}
        if warnings:
            result["warnings"] = warnings
        return json.dumps(result)

    async def _get_work_item_data(
        self, resolved: ResolvedWorkItem
    ) -> Union[dict, None]:
        """Get work item data from resolved work item info.

        Returns work item dict or None if not found.
        """
        query, root_key = self._GET_WORK_ITEM_QUERIES[resolved.parent.type]

        query_variables = {
            "fullPath": resolved.parent.full_path,
            "iid": str(resolved.work_item_iid),
            **get_query_variables_for_version(
                "includeHierarchyWidget", "includeDevelopmentWidget"
            ),
        }

        response = await self.gitlab_client.graphql(query, query_variables)

        if not response.get(root_key):
            raise ToolException(f"No {root_key} found in response")

        work_items = response.get(root_key, {}).get("workItems", {}).get("nodes", [])

        return work_items[0] if work_items else None

    @override
    def _get_required_feature_for_tier_check(
        self, kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Return the feature name if all requested types map to a single licensed feature."""
        types = kwargs.get("types")
        if not types:
            type_name = kwargs.get("type_name")
            types = [type_name] if type_name else None
        if not types:
            return None

        normalized = {t.upper().replace(" ", "_") for t in types}
        features = {_TYPE_TO_FEATURE[t] for t in normalized if t in _TYPE_TO_FEATURE}

        if len(features) == 1:
            return features.pop()

        return None
