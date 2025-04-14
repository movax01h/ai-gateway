import json
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

DESCRIPTION_CHARACTER_LIMIT = 1_048_576


class WriteEpicInput(BaseModel):
    group_id: Union[int, str] = Field(
        description="The ID or URL-encoded path of the group"
    )
    title: str = Field(description="The title of the epic")
    description: Optional[str] = Field(
        description=f"The description of the epic. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters."
    )
    labels: Optional[str] = Field(description="The comma-separated list of labels")
    confidential: Optional[bool] = Field(
        description="Whether the epic should be confidential"
    )
    start_date_is_fixed: Optional[bool] = Field(
        description="Whether start date should be sourced from `start_date_fixed` or from milestones"
    )
    start_date_fixed: Optional[str] = Field(
        description="The fixed start date of an epic"
    )
    due_date_is_fixed: Optional[bool] = Field(
        description="Whether due date should be sourced from `due_date_fixed` or from milestones"
    )
    due_date_fixed: Optional[str] = Field(description="The fixed due date of an epic")
    parent_id: Optional[int] = Field(description="The ID of a parent epic")


class CreateEpic(DuoBaseTool):
    name: str = "create_epic"
    description: str = "Create a new epic in a GitLab group."
    args_schema: Type[BaseModel] = WriteEpicInput

    async def _arun(self, group_id: Union[int, str], title: str, **kwargs: Any) -> str:
        data = {"title": title, **{k: v for k, v in kwargs.items() if v is not None}}

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/groups/{group_id}/epics",
                body=json.dumps(data),
            )
            return json.dumps({"created_epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: WriteEpicInput) -> str:
        return f"Create epic '{args.title}' in group {args.group_id}"


class ListEpicsInput(BaseModel):
    group_id: Union[int, str] = Field(
        description="The ID or URL-encoded path of the group"
    )
    author_id: Optional[int] = Field(
        description="Return epics created by the given user id"
    )
    author_username: Optional[str] = Field(
        description="Return epics created by the user with the given username"
    )
    labels: Optional[str] = Field(
        description="Return epics matching a comma-separated list of label names. Label names from the epic group or a parent group can be used"
    )
    with_labels_details: Optional[bool] = Field(
        description="""If true, response returns more details for each label in labels field: :name, :color, :description,
        :description_html, :text_color. Default is false"""
    )
    order_by: Optional[str] = Field(
        description="Return epics ordered by created_at, updated_at, or title fields. Default is created_at"
    )
    sort: Optional[str] = Field(
        description="Return epics sorted in asc or desc order. Default is desc"
    )
    search: Optional[str] = Field(
        description="Search epics against their title and description"
    )
    state: Optional[str] = Field(
        description="Search epics against their state, possible filters: opened, closed, and all, default: all"
    )
    created_after: Optional[str] = Field(
        description="Return epics created on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)"
    )
    created_before: Optional[str] = Field(
        description="Return epics created on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)"
    )
    updated_before: Optional[str] = Field(
        description="Return epics updated on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)"
    )
    updated_after: Optional[str] = Field(
        description="Return epics updated on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)"
    )
    include_ancestor_groups: Optional[bool] = Field(
        description="Include epics from the requested group's ancestors. Default is false"
    )
    include_descendant_groups: Optional[bool] = Field(
        description="Include epics from the requested group's descendants. Default is true"
    )
    my_reaction_emoji: Optional[str] = Field(
        description="""Return epics reacted by the authenticated user by the given emoji. None returns epics not given a reaction.
        Any returns epics given at least one reaction"""
    )
    negate: Optional[dict] = Field(
        description="Return epics that do not match the parameters supplied. Accepts: author_id, author_username and labels."
    )


class ListEpics(DuoBaseTool):
    name: str = "list_epics"
    description: str = "Get all epics of the requested group and its subgroups."
    args_schema: Type[BaseModel] = ListEpicsInput

    async def _arun(self, group_id: Union[int, str], **kwargs: Any) -> str:
        negate = kwargs.pop("negate", None)
        params = {k: v for k, v in kwargs.items() if v is not None}

        if negate:
            params |= {"not": negate}

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{group_id}/epics",
                params=params,
                parse_json=False,
            )
            return json.dumps({"epics": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListEpicsInput) -> str:
        return f"List epics in group {args.group_id}"


class GetEpicInput(BaseModel):
    group_id: Union[int, str] = Field(
        description="The ID or URL-encoded path of the group"
    )
    epic_iid: int = Field(description="The internal ID of the epic")


class GetEpic(DuoBaseTool):
    name: str = "get_epic"
    description: str = "Get a single epic in a GitLab group"
    args_schema: Type[BaseModel] = GetEpicInput

    async def _arun(self, group_id: Union[int, str], epic_iid: int) -> str:
        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{group_id}/epics/{epic_iid}",
                parse_json=False,
            )
            return json.dumps({"epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: GetEpicInput) -> str:
        return f"Read epic #{args.epic_iid} in group {args.group_id}"


class UpdateEpicInput(WriteEpicInput):
    epic_iid: int = Field(description="The internal ID of the epic")
    add_labels: Optional[str] = Field(
        description="Comma-separated label names to add to an epic"
    )
    remove_labels: Optional[str] = Field(
        description="Comma-separated label names to remove from an epic"
    )
    state_event: Optional[str] = Field(
        description="State event for an epic. Set `close` to close the epic and `reopen` to reopen it"
    )


class UpdateEpic(DuoBaseTool):
    name: str = "update_epic"
    description: str = "Update an existing epic in a GitLab group."
    args_schema: Type[BaseModel] = UpdateEpicInput  # type: ignore

    async def _arun(
        self, group_id: Union[int, str], epic_iid: int, **kwargs: Any
    ) -> str:
        data = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aput(
                path=f"/api/v4/groups/{group_id}/epics/{epic_iid}",
                body=json.dumps(data),
            )
            return json.dumps({"updated_epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: UpdateEpicInput) -> str:
        return f"Update epic #{args.epic_iid}"
