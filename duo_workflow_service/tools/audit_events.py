import json
import logging
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

logger = logging.getLogger(__name__)


class BaseAuditEventsInput(BaseModel):
    """Base model for audit events input parameters."""

    created_after: Optional[str] = Field(
        default=None,
        description="Return audit events created on or after the given time. Format: ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)",
    )
    created_before: Optional[str] = Field(
        default=None,
        description="Return audit events created on or before the given time. Format: ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)",
    )
    per_page: Optional[int] = Field(
        default=20,
        description="Number of results per page (default: 20, max: 100).",
        ge=1,
        le=100,
    )
    page: Optional[int] = Field(
        default=1,
        description="Page number to fetch (default: 1).",
        ge=1,
    )
    fetch_all_pages: Optional[bool] = Field(
        default=False,
        description="Whether to fetch all pages of results (default: False). Use with caution for large datasets.",
    )


class ListInstanceAuditEventsInput(BaseAuditEventsInput):
    entity_type: Optional[str] = Field(
        default=None,
        description="Return audit events for the given entity type. Valid values are: User, Group, Project, or "
        "Gitlab::Audit::InstanceScope.",
    )
    entity_id: Optional[int] = Field(
        default=None,
        description="Return audit events for the given entity ID. Requires entity_type attribute to be present.",
    )


class ListGroupAuditEventsInput(BaseAuditEventsInput):
    group_id: Optional[int] = Field(
        default=None,
        description="The ID of the group.",
    )
    group_path: Optional[str] = Field(
        default=None,
        description="The URL-encoded path of the group.",
    )


class ListProjectAuditEventsInput(ProjectResourceInput, BaseAuditEventsInput):
    pass


class BaseAuditEventsTool(DuoBaseTool):
    """Base class for audit events tools with shared pagination logic."""

    async def _fetch_paginated_audit_events(
        self,
        api_path: str,
        params: Dict[str, Any],
        fetch_all_pages: bool,
        per_page: int,
        initial_page: int,
    ) -> Tuple[list, Dict[str, Any]]:
        """Fetch audit events with pagination support.

        Returns:
            Tuple of (audit_events_list, pagination_info)
        """
        all_audit_events = []
        current_page = initial_page
        total_pages = None
        params["per_page"] = per_page

        while True:
            params["page"] = current_page
            response = await self.gitlab_client.aget(
                path=api_path,
                params=params,
                parse_json=True,
                use_http_response=True,
            )

            if not response.is_success():
                logger.error(
                    "API error - Status: %s, Body: %s",
                    response.status_code,
                    response.body,
                )

            if isinstance(response.body, dict) and (
                "message" in response.body or "error" in response.body
            ):
                error_msg = response.body.get(
                    "message", response.body.get("error", "Unknown error")
                )
                return [], {"error": error_msg}

            audit_events = response.body
            all_audit_events.extend(audit_events)

            # Get total pages from headers if available
            if (
                total_pages is None
                and hasattr(self.gitlab_client, "last_response")
                and self.gitlab_client.last_response
            ):
                try:
                    total_pages = int(
                        self.gitlab_client.last_response.headers.get("X-Total-Pages", 0)
                    )
                except (ValueError, TypeError):
                    total_pages = None

            # Break if we're not fetching all pages or if we've reached the last page
            if (
                not fetch_all_pages
                or len(audit_events) < per_page
                or (total_pages and current_page >= total_pages)
            ):
                break

            current_page += 1

        pagination_info = {
            "total_items": len(all_audit_events),
            "total_pages": total_pages,
            "current_page": current_page,
            "per_page": per_page,
        }

        return all_audit_events, pagination_info

    def _format_response(self, audit_events: list, pagination: Dict[str, Any]) -> str:
        """Format the response in a consistent way."""
        return json.dumps(
            {
                "audit_events": audit_events,
                "pagination": pagination,
            }
        )

    def _format_error(self, error: str) -> str:
        """Format error responses consistently."""
        return json.dumps({"error": error})

    async def _execute_audit_query(self, api_path: str, **kwargs: Any) -> str:
        """Common execution logic for audit event queries.

        This method handles the common pattern of:
        1. Extracting pagination parameters
        2. Building request parameters
        3. Calling the paginated fetch method
        4. Handling exceptions
        5. Formatting the response
        """
        fetch_all_pages = kwargs.pop("fetch_all_pages", False)
        per_page = kwargs.pop("per_page", 20)
        page = kwargs.pop("page", 1)

        params = {k: v for k, v in kwargs.items() if v is not None}

        audit_events, pagination = await self._fetch_paginated_audit_events(
            api_path=api_path,
            params=params,
            fetch_all_pages=fetch_all_pages,
            per_page=per_page,
            initial_page=page,
        )

        if "error" in pagination:
            return self._format_error(pagination["error"])

        return self._format_response(audit_events, pagination)


class ListInstanceAuditEvents(BaseAuditEventsTool):
    name: str = "list_instance_audit_events"
    description: str = """List instance-level audit events in GitLab.

    **Access Requirements**: Only instance administrators can access instance audit events.

    **Confidentiality**: Audit events contain sensitive security and compliance information.
    Do not share these events outside of this chat conversation.

    Examples:
    - List all instance audit events:
        list_instance_audit_events()
    - List audit events for a specific entity:
        list_instance_audit_events(entity_type="Project", entity_id=6)
    - List audit events created after a certain date:
        list_instance_audit_events(created_after="2023-01-01T00:00:00Z")
    """
    args_schema: Type[BaseModel] = ListInstanceAuditEventsInput  # type: ignore

    async def _execute(self, **kwargs: Any) -> str:
        # Validate entity_id requires entity_type
        if kwargs.get("entity_id") and not kwargs.get("entity_type"):
            return self._format_error("entity_id requires entity_type to be specified")

        return await self._execute_audit_query(
            api_path="/api/v4/audit_events", **kwargs
        )

    def format_display_message(
        self, args: ListInstanceAuditEventsInput, _tool_response: Any = None
    ) -> str:
        if args.entity_type and args.entity_id:
            return f"List instance audit events for {args.entity_type} {args.entity_id}"
        return "List instance audit events"


class ListGroupAuditEvents(BaseAuditEventsTool):
    name: str = "list_group_audit_events"
    description: str = """List audit events for a GitLab group.

    **Access Requirements**: Only group owners can access group audit events.

    **Confidentiality**: Audit events contain sensitive security and compliance information.
    Do not share these events outside of this chat conversation.

    To identify the group you must provide either:
    - group_id parameter, or
    - group_path parameter (the URL-encoded path of the group)

    Examples:
    - List audit events for group with ID 60:
        list_group_audit_events(group_id=60)
    - List audit events for group by path:
        list_group_audit_events(group_path="gitlab-org/gitlab")
    - List recent audit events:
        list_group_audit_events(group_id=60, created_after="2023-01-01T00:00:00Z")
    """
    args_schema: Type[BaseModel] = ListGroupAuditEventsInput  # type: ignore

    async def _execute(self, **kwargs: Any) -> str:
        group_id = kwargs.get("group_id")
        group_path = kwargs.get("group_path")

        # Validate group identification
        if not group_id and not group_path:
            return self._format_error("Either group_id or group_path must be provided")

        # Use group_id if provided, otherwise use group_path
        group_identifier = group_id if group_id else group_path

        kwargs.pop("group_id", None)
        kwargs.pop("group_path", None)

        return await self._execute_audit_query(
            api_path=f"/api/v4/groups/{group_identifier}/audit_events", **kwargs
        )

    def format_display_message(
        self, args: ListGroupAuditEventsInput, _tool_response: Any = None
    ) -> str:
        if args.group_path:
            return f"List audit events for group {args.group_path}"
        return f"List audit events for group {args.group_id}"


class ListProjectAuditEvents(BaseAuditEventsTool):
    name: str = "list_project_audit_events"
    description: str = """List audit events for a GitLab project.

    **Access Requirements**: Only project owners can access project audit events.

    **Confidentiality**: Audit events contain sensitive security and compliance information.
    Do not share these events outside of this chat conversation.

    To identify the project you must provide either:
    - project_id parameter, or
    - A GitLab URL like:
        - https://gitlab.com/namespace/project
        - https://gitlab.com/group/subgroup/project

    Examples:
    - List audit events for project with ID 7:
        list_project_audit_events(project_id=7)
    - List audit events for project by URL:
        list_project_audit_events(url="https://gitlab.com/gitlab-org/gitlab")
    - List recent audit events:
        list_project_audit_events(project_id=7, created_after="2023-01-01T00:00:00Z")
    """
    args_schema: Type[BaseModel] = ListProjectAuditEventsInput  # type: ignore

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return self._format_error("; ".join(errors))

        kwargs.pop("url", None)
        kwargs.pop("project_id", None)

        return await self._execute_audit_query(
            api_path=f"/api/v4/projects/{project_id}/audit_events", **kwargs
        )

    def format_display_message(
        self, args: ListProjectAuditEventsInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"List audit events for {args.url}"
        return f"List audit events for project {args.project_id}"
