import json
from typing import Any, List, NamedTuple, Optional, Type, Union

from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

DESCRIPTION_CHARACTER_LIMIT = 1_048_576

GROUP_IDENTIFICATION_DESCRIPTION = """To identify the group you must provide either:
- group_id parameter, or
- A GitLab URL like:
  - https://gitlab.com/namespace/group
  - https://gitlab.com/groups/namespace/group
"""

EPIC_IDENTIFICATION_DESCRIPTION = """To identify an epic you must provide either:
- group_id and epic_iid, or
- A GitLab URL like:
  - https://gitlab.com/groups/namespace/group/-/epics/42
"""


class GroupURLValidationResult(NamedTuple):
    group_id: Optional[str]
    errors: List[str]


class EpicURLValidationResult(NamedTuple):
    group_id: Optional[str]
    epic_iid: Optional[int]
    errors: List[str]


class EpicIdsResult(NamedTuple):
    group_id: Optional[str]
    epic_iid: Optional[int]
    epic_id: Optional[int]
    errors: List[str]


class EpicBaseTool(DuoBaseTool):
    def _validate_group_url(
        self, url: Optional[str], group_id: Optional[int | str]
    ) -> GroupURLValidationResult:
        """Validate group URL and extract group_id.

        Args:
            url: The GitLab URL to parse
            group_id: The group ID provided by the user

        Returns:
            GroupURLValidationResult containing:
                - The validated group_id (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not group_id:
                errors.append("'group_id' must be provided when 'url' is not")
            return GroupURLValidationResult(
                None if group_id is None else str(group_id), errors
            )

        try:
            # Parse URL and validate netloc against gitlab_host
            url_group_id = GitLabUrlParser.parse_group_url(url, self.gitlab_host)

            # If both URL and group_id are provided, check if they match
            if group_id is not None and str(group_id) != url_group_id:
                errors.append(
                    f"Group ID mismatch: provided '{group_id}' but URL contains '{url_group_id}'"
                )

            # Use the group_id from the URL
            return GroupURLValidationResult(url_group_id, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return GroupURLValidationResult(
                None if group_id is None else str(group_id), errors
            )

    def _validate_epic_url(
        self, url: Optional[str], group_id: Optional[int | str], epic_iid: Optional[int]
    ) -> EpicURLValidationResult:
        """Validate epic URL and extract group_id and epic_iid.

        Args:
            url: The GitLab URL to parse
            group_id: The group ID provided by the user
            epic_iid: The epic IID provided by the user

        Returns:
            EpicURLValidationResult containing:
                - The validated group_id (or None if validation failed)
                - The validated epic_iid (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not group_id:
                errors.append("'group_id' must be provided when 'url' is not")
            if not epic_iid:
                errors.append("'epic_iid' must be provided when 'url' is not")
            return EpicURLValidationResult(
                None if group_id is None else str(group_id), epic_iid, errors
            )

        try:
            # Parse URL and validate netloc against gitlab_host
            url_group_id, url_epic_iid = GitLabUrlParser.parse_epic_url(
                url, self.gitlab_host
            )

            # If both URL and IDs are provided, check if they match
            if group_id is not None and str(group_id) != url_group_id:
                errors.append(
                    f"Group ID mismatch: provided '{group_id}' but URL contains '{url_group_id}'"
                )
            if epic_iid is not None and epic_iid != url_epic_iid:
                errors.append(
                    f"Epic ID mismatch: provided '{epic_iid}' but URL contains '{url_epic_iid}'"
                )

            # Use the IDs from the URL
            return EpicURLValidationResult(url_group_id, url_epic_iid, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return EpicURLValidationResult(
                None if group_id is None else str(group_id), epic_iid, errors
            )

    async def _get_epic_ids_from_url(
        self,
        url: Optional[str],
        group_id: Optional[str],
        epic_iid: Optional[int],
        epic_id: Optional[int],
    ) -> EpicIdsResult:
        """Get group_id, epic_iid and epic_id from URL or direct parameters.

        Args:
            url: The GitLab URL to parse
            group_id: The group ID provided by the user
            epic_iid: The epic IID provided by the user
            epic_id: The epic ID provided by the user

        Returns:
            EpicIdsResult containing:
                - The validated group_id (or None if validation failed)
                - The validated epic_iid (or None if validation failed)
                - The validated/fetched epic_id (or None if validation or fetch failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        # If URL is provided, use validation to get group_id and epic_iid
        if url:
            validation_result = self._validate_epic_url(url, group_id, epic_iid)
            if validation_result.errors:
                return EpicIdsResult(None, None, None, validation_result.errors)

            # Use the validated group_id and epic_iid
            group_id = validation_result.group_id
            epic_iid = validation_result.epic_iid

            # If epic_id is not provided, fetch it from the API using epic_iid
            if epic_id is None and epic_iid is not None:
                try:
                    epic_data = await self.gitlab_client.aget(
                        path=f"/api/v4/groups/{group_id}/epics/{epic_iid}",
                        parse_json=True,
                    )
                    epic_id = epic_data.get("id")
                    if not epic_id:
                        errors.append(
                            f"Could not find epic_id for epic with iid {epic_iid}"
                        )
                        return EpicIdsResult(group_id, epic_iid, None, errors)
                except Exception as e:
                    errors.append(f"Error looking up epic: {str(e)}")
                    return EpicIdsResult(group_id, epic_iid, None, errors)

        if group_id is None or epic_id is None:
            if group_id is None:
                errors.append("'group_id' must be provided when 'url' is not")
            if epic_id is None:
                errors.append("'epic_id' must be provided when 'url' is not")
            return EpicIdsResult(group_id, epic_iid, epic_id, errors)

        return EpicIdsResult(group_id, epic_iid, epic_id, errors)


class GroupResourceInput(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="GitLab URL for the resource. If provided, other ID fields are not required.",
    )
    group_id: Optional[Union[int, str]] = Field(
        default=None,
        description="The ID or URL-encoded path of the group. Required if URL is not provided.",
    )


class EpicResourceInput(GroupResourceInput):
    epic_iid: Optional[int] = Field(
        default=None,
        description="The internal ID of the epic. Required if URL is not provided.",
    )
    epic_id: Optional[int] = Field(
        default=None,
        description="The ID of the epic.",
    )


class WriteEpicInput(EpicResourceInput):
    title: str = Field(description="The title of the epic")
    description: Optional[str] = Field(
        default=None,
        description=f"The description of the epic. Limited to {DESCRIPTION_CHARACTER_LIMIT} characters.",
    )
    labels: Optional[str] = Field(
        default=None, description="The comma-separated list of labels"
    )
    confidential: Optional[bool] = Field(
        default=None, description="Whether the epic should be confidential"
    )
    start_date_is_fixed: Optional[bool] = Field(
        default=None,
        description="Whether start date should be sourced from `start_date_fixed` or from milestones",
    )
    start_date_fixed: Optional[str] = Field(
        default=None, description="The fixed start date of an epic"
    )
    due_date_is_fixed: Optional[bool] = Field(
        default=None,
        description="Whether due date should be sourced from `due_date_fixed` or from milestones",
    )
    due_date_fixed: Optional[str] = Field(
        default=None, description="The fixed due date of an epic"
    )
    parent_id: Optional[int] = Field(
        default=None, description="The ID of a parent epic"
    )


class CreateEpic(EpicBaseTool):
    name: str = "create_epic"
    description: str = f"""Create a new epic in a GitLab group.

    {GROUP_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and title 'New Feature', the tool call would be:
        create_epic(group_id='namespace/group', title='New Feature')
    - Given the URL https://gitlab.com/groups/namespace/group and title 'New Feature', the tool call would be:
        create_epic(url="https://gitlab.com/groups/namespace/group", title='New Feature')
    """
    args_schema: Type[BaseModel] = WriteEpicInput

    async def _arun(self, title: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)

        validation_result = self._validate_group_url(url, group_id)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        data = {"title": title, **{k: v for k, v in kwargs.items() if v is not None}}

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/groups/{validation_result.group_id}/epics",
                body=json.dumps(data),
            )
            return json.dumps({"created_epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: WriteEpicInput) -> str:
        target = args.url if args.url else f"group {args.group_id}"
        return f"Create epic '{args.title}' in {target}"


class ListEpicsInput(GroupResourceInput):
    author_id: Optional[int] = Field(
        default=None, description="Return epics created by the given user id"
    )
    author_username: Optional[str] = Field(
        default=None,
        description="Return epics created by the user with the given username",
    )
    labels: Optional[str] = Field(
        default=None,
        description="Return epics matching a comma-separated list of label names. Label names from the epic group or a parent group can be used",
    )
    with_labels_details: Optional[bool] = Field(
        default=None,
        description="""If true, response returns more details for each label in labels field: :name, :color, :description,
        :description_html, :text_color. Default is false""",
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Return epics ordered by created_at, updated_at, or title fields. Default is created_at",
    )
    sort: Optional[str] = Field(
        default=None,
        description="Return epics sorted in asc or desc order. Default is desc",
    )
    search: Optional[str] = Field(
        default=None, description="Search epics against their title and description"
    )
    state: Optional[str] = Field(
        default=None,
        description="Search epics against their state, possible filters: opened, closed, and all, default: all",
    )
    created_after: Optional[str] = Field(
        default=None,
        description="Return epics created on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)",
    )
    created_before: Optional[str] = Field(
        default=None,
        description="Return epics created on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)",
    )
    updated_before: Optional[str] = Field(
        default=None,
        description="Return epics updated on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)",
    )
    updated_after: Optional[str] = Field(
        default=None,
        description="Return epics updated on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)",
    )
    include_ancestor_groups: Optional[bool] = Field(
        default=None,
        description="Include epics from the requested group's ancestors. Default is false",
    )
    include_descendant_groups: Optional[bool] = Field(
        default=None,
        description="Include epics from the requested group's descendants. Default is true",
    )
    my_reaction_emoji: Optional[str] = Field(
        default=None,
        description="""Return epics reacted by the authenticated user by the given emoji. None returns epics not given a reaction.
        Any returns epics given at least one reaction""",
    )
    negate: Optional[dict] = Field(
        default=None,
        description="Return epics that do not match the parameters supplied. Accepts: author_id, author_username and labels.",
    )


class ListEpics(EpicBaseTool):
    name: str = "list_epics"
    description: str = f"""Get all epics of the requested group and its subgroups.

    {GROUP_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group', the tool call would be:
        list_epics(group_id='namespace/group')
    - Given the URL https://gitlab.com/groups/namespace/group, the tool call would be:
        list_epics(url="https://gitlab.com/groups/namespace/group")
    """
    args_schema: Type[BaseModel] = ListEpicsInput

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)

        validation_result = self._validate_group_url(url, group_id)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        negate = kwargs.pop("negate", None)
        params = {k: v for k, v in kwargs.items() if v is not None}

        if negate:
            params |= {"not": negate}

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{validation_result.group_id}/epics",
                params=params,
                parse_json=False,
            )
            return json.dumps({"epics": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListEpicsInput) -> str:
        target = args.url if args.url else f"group {args.group_id}"
        return f"List epics in {target}"


class GetEpic(EpicBaseTool):
    name: str = "get_epic"
    description: str = f"""Get a single epic in a GitLab group

    {EPIC_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and epic_iid 42, the tool call would be:
        get_epic(group_id='namespace/group', epic_iid=42)
    - Given the URL https://gitlab.com/groups/namespace/group/-/epics/42, the tool call would be:
        get_epic(url="https://gitlab.com/groups/namespace/group/-/epics/42")
    """
    args_schema: Type[BaseModel] = EpicResourceInput

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        group_id = kwargs.get("group_id")
        epic_iid = kwargs.get("epic_iid")

        validation_result = self._validate_epic_url(url, group_id, epic_iid)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{validation_result.group_id}/epics/{validation_result.epic_iid}",
                parse_json=False,
            )
            return json.dumps({"epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: EpicResourceInput) -> str:
        if args.url:
            return f"Read epic {args.url}"
        return f"Read epic #{args.epic_iid} in group {args.group_id}"


class UpdateEpicInput(WriteEpicInput):
    add_labels: Optional[str] = Field(
        default=None, description="Comma-separated label names to add to an epic"
    )
    remove_labels: Optional[str] = Field(
        default=None, description="Comma-separated label names to remove from an epic"
    )
    state_event: Optional[str] = Field(
        default=None,
        description="State event for an epic. Set `close` to close the epic and `reopen` to reopen it",
    )


class UpdateEpic(EpicBaseTool):
    name: str = "update_epic"
    description: str = f"""Update an existing epic in a GitLab group.

    {EPIC_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group', epic_iid 42, and title 'Updated Epic', the tool call would be:
        update_epic(group_id='namespace/group', epic_iid=42, title='Updated Epic')
    - Given the URL https://gitlab.com/groups/namespace/group/-/epics/42 and title 'Updated Epic', the tool call would be:
      update_epic(url="https://gitlab.com/groups/namespace/group/-/epics/42", title='Updated Epic')
    """
    args_schema: Type[BaseModel] = UpdateEpicInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        epic_iid = kwargs.pop("epic_iid", None)

        validation_result = self._validate_epic_url(url, group_id, epic_iid)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        data = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aput(
                path=f"/api/v4/groups/{validation_result.group_id}/epics/{validation_result.epic_iid}",
                body=json.dumps(data),
            )
            return json.dumps({"updated_epic": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: UpdateEpicInput) -> str:
        if args.url:
            return f"Update epic {args.url}"
        return f"Update epic #{args.epic_iid}"


class ListEpicNotesInput(EpicResourceInput):
    sort: Optional[str] = Field(
        default=None,
        description="Return epic notes sorted in asc or desc order. Default is desc.",
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Return epic notes ordered by created_at or updated_at fields. Default is created_at",
    )


class ListEpicNotes(EpicBaseTool):
    name: str = "list_epic_notes"
    description: str = f"""Get a list of all notes (comments) for a specific epic.

    {EPIC_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group' and epic_id 42, the tool call would be:
        list_epic_notes(group_id='namespace/group', epic_id=42)
    - Given the URL https://gitlab.com/groups/namespace/group/-/epics/42, the tool call would be:
        list_epic_notes(url="https://gitlab.com/groups/namespace/group/-/epics/42")

    When using a URL, the tool will automatically fetch the epic details to get the correct epic_id.
    """
    args_schema: Type[BaseModel] = ListEpicNotesInput

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        epic_iid = kwargs.pop("epic_iid", None)
        epic_id = kwargs.pop("epic_id", None)

        result = await self._get_epic_ids_from_url(url, group_id, epic_iid, epic_id)

        if result.errors:
            return json.dumps({"error": "; ".join(result.errors)})
        if result.epic_id is None:
            return json.dumps({"error": "Could not determine epic_id."})

        params = {k: v for k, v in kwargs.items() if v is not None}
        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{result.group_id}/epics/{result.epic_id}/notes",
                params=params,
                parse_json=False,
            )
            return json.dumps({"notes": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListEpicNotesInput) -> str:
        if args.url:
            return f"Read comments on epic {args.url}"
        return f"Read comments on epic #{args.epic_iid} in group {args.group_id}"


class GetEpicNoteInput(EpicResourceInput):
    note_id: int = Field(description="The ID of the note")


class GetEpicNote(EpicBaseTool):
    name: str = "get_epic_note"
    description: str = f"""Get a single note (comment) from a specific epic.

    {EPIC_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given group_id 'namespace/group', epic_id 42, and note_id 5, the tool call would be:
        get_epic_note(group_id='namespace/group', epic_id=42, note_id=5)
    - Given the URL https://gitlab.com/groups/namespace/group/-/epics/42 and note_id 5, the tool call would be:
        get_epic_note(url="https://gitlab.com/groups/namespace/group/-/epics/42", note_id=5)

    When using a URL, the tool will automatically fetch the epic details to get the correct epic_id.
    The note_id parameter is always required.
    """
    args_schema: Type[BaseModel] = GetEpicNoteInput

    async def _arun(self, note_id: int, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        group_id = kwargs.pop("group_id", None)
        epic_iid = kwargs.pop("epic_iid", None)
        epic_id = kwargs.pop("epic_id", None)

        result = await self._get_epic_ids_from_url(url, group_id, epic_iid, epic_id)

        if result.errors:
            return json.dumps({"error": "; ".join(result.errors)})
        if result.epic_id is None:
            return json.dumps({"error": "Could not determine epic_id."})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/groups/{result.group_id}/epics/{result.epic_id}/notes/{note_id}",
                parse_json=False,
            )
            return json.dumps({"note": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: GetEpicNoteInput) -> str:
        if args.url:
            return f"Read comment #{args.note_id} on epic {args.url}"
        return f"Read comment #{args.note_id} on epic #{args.epic_iid} in group {args.group_id}"
