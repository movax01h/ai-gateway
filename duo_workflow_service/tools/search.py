import json
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class BaseSearchInput(BaseModel):
    id: str = Field(description="The ID of the project or group")
    search: str = Field(description="The search term")
    search_type: Literal["projects", "groups"] = Field(
        description="Whether to search in a project or a group"
    )
    order_by: Optional[str] = Field(
        description="Sort results. Allowed value is created_at", default=None
    )
    sort: Optional[str] = Field(
        description="Sort order. Allowed values are asc or desc", default=None
    )


class GitLabSearchBase(DuoBaseTool, ABC):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = BaseModel

    @classmethod
    def _get_description(cls, unique_description: str) -> str:
        return f"""
        {unique_description}

        Search Term Syntax:
        - Use quotes for exact phrase matches: "exact phrase"
        - Use + to specify AND condition: term1+term2
        - Use | for OR condition: term1|term2
        - Use - to exclude terms: -term
        - Use * for wildcard searches: ter*
        - Use \\ to escape special characters: \\#group
        - Use # for searching by issue or merge request ID: #123
        - Use ! for exact word match: !word
        - Use ~ followed by a number for fuzzy search: word~3
        - Parentheses can be used for grouping: (term1+term2)|term3

        Examples:
        - "hello world": Exact phrase match
        - hello+world: Contains both "hello" AND "world"
        - hello|world: Contains either "hello" OR "world"
        - hello -world: Contains "hello" but NOT "world"
        - hel*o: Matches "hello", "helio", etc.
        - \\#group: Searches for the literal "#group"
        - #123: Searches for issue or merge request with ID 123
        - !important: Matches the exact word "important"
        - hello~2: Fuzzy search for "hello" with up to 2 character differences
        - (hello+world)|"greeting message": Complex query

        Note: If a user is not a member of a private project or a private group, this tool is going to result in a 404 status code.
        """

    @abstractmethod
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        pass

    async def _perform_search(self, id: str, params: dict, search_type: str) -> str:
        url = f"/api/v4/{search_type}/{id}/search"
        try:
            response = await self.gitlab_client.aget(path=url, params=params)
            return json.dumps({"search_results": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: BaseSearchInput) -> str:
        return self.format_gitlab_search(self.name, args)

    def format_gitlab_search(self, tool_name, args: BaseSearchInput) -> str:
        search_id = args.id
        search_term = args.search
        search_type = args.search_type

        if "issue" in tool_name:
            scope = "issues"
        elif "merge_request" in tool_name:
            scope = "merge requests"
        elif "milestone" in tool_name:
            scope = "milestones"
        elif "project" in tool_name:
            scope = "projects"
        elif "blob" in tool_name:
            scope = "files"
        elif "commit" in tool_name:
            scope = "commits"
        elif "note" in tool_name:
            scope = "comments"
        elif "user" in tool_name:
            scope = "users"
        else:
            scope = "items"

        return (
            f"Search for {scope} with term '{search_term}' in {search_type} {search_id}"
        )


class GroupProjectSearch(GitLabSearchBase):
    name: str = "gitlab_group_project_search"
    unique_description: str = """
    Search for projects within a specified GitLab group.

    Parameters:
    - id: The ID of the group (required)
    - search_type: Must be "groups" for this search (required)
    - search: The search term (required)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_group_project_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'scope': 'projects',
            'search': 'Duo Workflow',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = BaseSearchInput

    async def _arun(
        self,
        id: str,
        search: str,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "projects",
            "search": search,
        }
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, "groups")


class IssueSearchInput(BaseSearchInput):
    confidential: Optional[bool] = Field(
        description="Filter by confidentiality", default=None
    )
    state: Optional[str] = Field(description="Filter by state", default=None)


class IssueSearch(GitLabSearchBase):
    name: str = "gitlab_issue_search"
    unique_description: str = """
    Search for issues in the specified GitLab project or group.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - confidential: Filter by confidentiality
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc
    - state: Filter by state

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_issue_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'projects',
            'scope': 'issues',
            'search': 'Duo Workflow',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = IssueSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        confidential: Optional[bool] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "issues",
            "search": search,
        }
        if confidential is not None:
            params["confidential"] = str(confidential).lower()
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort
        if state:
            params["state"] = state

        return await self._perform_search(id, params, search_type)


class MergeRequestSearchInput(BaseSearchInput):
    state: Optional[str] = Field(description="Filter by state", default=None)


class MergeRequestSearch(GitLabSearchBase):
    name: str = "gitlab_merge_request_search"
    unique_description: str = """
    Search for merge requests in the specified GitLab project or group.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc
    - state: Filter by state

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_merge_request_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'groups',
            'scope': 'merge_requests',
            'search': 'Duo Workflow',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = MergeRequestSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "merge_requests",
            "search": search,
        }
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort
        if state:
            params["state"] = state

        return await self._perform_search(id, params, search_type)


class MilestoneSearch(GitLabSearchBase):
    name: str = "gitlab_milestone_search"
    unique_description: str = """
    Search for milestones in the specified GitLab project or group.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_milestone_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'projects',
            'scope': 'milestones',
            'search': 'Duo Workflow',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = BaseSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "milestones",
            "search": search,
        }
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)


class UserSearch(GitLabSearchBase):
    name: str = "gitlab__user_search"
    unique_description: str = """
    Search for users in the specified GitLab project or group.

    Parameters:
    - id: The ID of the project or group owned by the authenticated user. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {{
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_project_user_search',
        'type': 'tool_use'
        'input': {{
            'id': 123,
            'search_type': 'groups',
            'scope': 'users',
            'search': 'Duo Workflow User',
        }},
    }}
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = BaseSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "users",
            "search": search,
        }
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)


class RefSearchInput(BaseSearchInput):
    ref: Optional[str] = Field(
        description="The name of a repository branch or tag to search on (only applicable for project searches)",
        default=None,
    )


class BlobSearch(GitLabSearchBase):
    name: str = "gitlab_blob_search"
    unique_description: str = """
    Search for blobs in the specified GitLab group or project. In GitLab, a "blob" refers to a file's content in a specific version of the repository.
    This can include source code files, text files, or any other file type stored in the repository.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - ref: The name of a repository branch or tag to search on. Only applicable for projects search_type
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_blob_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'projects',
            'scope': 'blobs',
            'search': 'Duo Workflow',
            'ref': 'main',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = RefSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects"],
        ref: Optional[str] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "blobs",
            "search": search,
        }
        if ref and search_type == "projects":
            params["ref"] = ref
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)


class CommitSearch(GitLabSearchBase):
    name: str = "gitlab_commit_search"
    unique_description: str = """
    Search for commits in the specified GitLab project or group.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - ref: The name of a repository branch or tag to search on (only applicable for project searches)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_commit_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'projects',
            'scope': 'commits',
            'search': 'Duo Workflow',
            'ref': 'main',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = RefSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        ref: Optional[str] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "commits",
            "search": search,
        }
        if ref and search_type == "projects":
            params["ref"] = ref
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)


class WikiBlobSearch(GitLabSearchBase):
    name: str = "gitlab_wiki_blob_search"
    unique_description: str = """
    Search for wiki blobs in the specified GitLab project or group. In GitLab, a "blob" refers to a file's content in a specific version of the repository.
    This can include source code files, text files, or any other file type stored in the repository.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - ref: The name of a repository branch or tag to search on (only applicable for project searches)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_wiki_blob_search',
        'type': 'tool_use'
        'input': {
            'id': 123,
            'search_type': 'projects',
            'scope': 'wiki_blobs',
            'search': 'Duo Workflow',
            'ref': 'main',
        },
    }
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = RefSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        ref: Optional[str] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "wiki_blobs",
            "search": search,
        }
        if ref and search_type == "projects":
            params["ref"] = ref
        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)


class NoteSearch(GitLabSearchBase):
    name: str = "gitlab_note_search"
    unique_description: str = """
    Search for notes in the specified GitLab project.

    Parameters:
    - id: The ID of the project or group. In GitLab, a namespace and group are used interchangeably,
            so either a group_id or namespace_id can be used to fill this argument. (required)
    - search_type: Whether to search in a project or a group (required)
    - search: The search term (required)
    - order_by: Sort results. Allowed value is created_at
    - sort: Sort order. Allowed values are asc or desc

    An example tool_call is presented below
    {{
        'id': 'toolu_01KqpqRQhTM2pxJrhtTscMWu',
        'name': 'gitlab_project_note_search',
        'type': 'tool_use'
        'input': {{
            'id': 123,
            'search_type': 'groups',
            'scope': 'notes',
            'search': 'Duo Workflow',
        }},
    }}
    """
    description: str = GitLabSearchBase._get_description(unique_description)
    args_schema: Type[BaseModel] = BaseSearchInput

    async def _arun(
        self,
        *,
        id: str,
        search: str,
        search_type: Literal["projects", "groups"],
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> str:
        params = {
            "scope": "notes",
            "search": search,
        }

        if order_by:
            params["order_by"] = order_by
        if sort:
            params["sort"] = sort

        return await self._perform_search(id, params, search_type)
