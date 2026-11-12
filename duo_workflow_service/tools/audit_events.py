import json
from typing import Any, ClassVar, Dict, Optional, Tuple, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

logger = structlog.stdlib.get_logger(__name__)

# Upper bound on pages fetched by a single tool call when fetch_all_pages is true.
MAX_AUDIT_EVENT_PAGES = 5


def _read_pagination_headers(response: Any) -> Dict[str, str]:
    """Return the response headers with lowercased keys.

    Headers may be a dict or a protobuf map; a missing or None headers object is treated as empty.
    """
    headers = getattr(response, "headers", None)
    if not headers:
        return {}

    try:
        items = headers.items()
    except AttributeError:
        return {}

    return {
        str(key).lower(): "" if value is None else str(value) for key, value in items
    }


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


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
        description=(
            "Whether to fetch all pages of results (default: False). Use with caution for large datasets. "
            f"At most {MAX_AUDIT_EVENT_PAGES} pages are fetched per call; when the response pagination has "
            "truncated=true and has_more=true, narrow the time range or continue from the next page."
        ),
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

    tier_check_licensed_feature: ClassVar[str] = "AUDIT_EVENTS"

    async def _fetch_paginated_audit_events(
        self,
        api_path: str,
        params: Dict[str, Any],
        fetch_all_pages: bool,
        per_page: int,
        initial_page: int,
    ) -> Tuple[list, Dict[str, Any]]:
        """Fetch audit events with pagination support.

        When fetch_all_pages is true, at most MAX_AUDIT_EVENT_PAGES pages are fetched. The loop also stops on a short
        page or when the x-next-page header is present and empty. x-total-pages is reported but never used to stop,
        because GitLab omits it for queries with more than 10,000 records.

        Returns:
            Tuple of (audit_events_list, pagination_info)
        """
        all_audit_events: list = []
        current_page = initial_page
        last_fetched_page = initial_page
        pages_fetched = 0
        total_pages: Optional[int] = None
        truncated = False
        has_more = False
        error: Optional[str] = None
        params["per_page"] = per_page

        while True:
            params["page"] = current_page
            response = await self.gitlab_client.aget(
                path=api_path,
                params=params,
                parse_json=True,
            )

            # Read headers from the raw response: _process_http_response returns only the body.
            headers = _read_pagination_headers(response)

            try:
                # A later-page failure is logged once, with the page number, by the warning below, so only
                # the first page lets _process_http_response log it.
                audit_events = self._process_http_response(
                    identifier="audit events",
                    response=response,
                    logger=logger if pages_fetched == 0 else None,
                )
            except ToolException as exc:
                if pages_fetched == 0:
                    raise
                # The HTTP client retries transient failures such as 429, so a later-page failure only
                # reaches here after those retries are exhausted. Keep the events collected so far.
                logger.warning(
                    "Audit events pagination stopped early",
                    page=current_page,
                    error=str(exc),
                )
                truncated = True
                has_more = True
                error = str(exc)
                break

            pages_fetched += 1
            last_fetched_page = current_page
            all_audit_events.extend(audit_events)

            if "x-total-pages" in headers:
                total_pages = _parse_int(headers["x-total-pages"])

            # x-next-page absent means unknown, so rely on the short-page check; present and empty means
            # this is the last page. Never stop on a missing x-total-pages: GitLab omits it above 10,000 records.
            next_page = headers.get("x-next-page")
            is_last_page = next_page is not None and not next_page.strip()
            has_more = len(audit_events) >= per_page and not is_last_page

            if not fetch_all_pages or not has_more:
                break

            if pages_fetched >= MAX_AUDIT_EVENT_PAGES:
                truncated = True
                break

            current_page += 1

        pagination_info: Dict[str, Any] = {
            "total_items": len(all_audit_events),
            "total_pages": total_pages,
            "current_page": last_fetched_page,
            "per_page": per_page,
            "truncated": truncated,
            "has_more": has_more,
        }
        if error is not None:
            pagination_info["error"] = error

        return all_audit_events, pagination_info

    def _format_response(self, audit_events: list, pagination: Dict[str, Any]) -> str:
        """Format the response in a consistent way."""
        return json.dumps(
            {
                "audit_events": audit_events,
                "pagination": pagination,
            }
        )

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
    args_schema: Type[BaseModel] = ListInstanceAuditEventsInput

    async def _execute(self, **kwargs: Any) -> str:
        # Validate entity_id requires entity_type
        if kwargs.get("entity_id") and not kwargs.get("entity_type"):
            raise ToolException("entity_id requires entity_type to be specified")

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
    args_schema: Type[BaseModel] = ListGroupAuditEventsInput

    async def _execute(self, **kwargs: Any) -> str:
        group_id = kwargs.get("group_id")
        group_path = kwargs.get("group_path")

        # Validate group identification
        if not group_id and not group_path:
            raise ToolException("Either group_id or group_path must be provided")

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
    - A GitLab SaaS URL like:
        - https://gitlab.com/namespace/project
        - https://gitlab.com/group/subgroup/project
    - A self-managed GitLab URL like:
        - https://gitlab.example.com/namespace/project
        - https://gitlab.example.com/group/subgroup/project

    Examples:
    - List audit events for project with ID 7:
        list_project_audit_events(project_id=7)
    - List audit events for project by URL:
        list_project_audit_events(url="https://gitlab.com/gitlab-org/gitlab")
    - List recent audit events:
        list_project_audit_events(project_id=7, created_after="2023-01-01T00:00:00Z")
    """
    args_schema: Type[BaseModel] = ListProjectAuditEventsInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            raise ToolException("; ".join(errors))

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
