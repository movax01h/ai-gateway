import json
from abc import abstractmethod
from textwrap import dedent
from typing import Any, Literal, Optional, Type

import structlog
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("search")


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


class GitLabSearchBase(DuoBaseTool):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = BaseModel

    @classmethod
    def _get_description(cls, unique_description: str) -> str:
        # codespell:ignore-begin
        return f"""
        {unique_description}

        Search Term Syntax (for code search only):
        - filename: Search by filename: filename:*spec.rb
        - path: Search by repository location (full or partial matches): path:spec/workers/
        - extension: Search by file extension without . (exact matches only): extension:js
        - blob: Search by Git object ID (exact matches only): blob:998707*
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
        - rails -filename:gemfile.lock: Returns rails in all files except the gemfile.lock file
        - RSpec.describe Resolvers -*builder: Returns RSpec.describe Resolvers that does not start with builder
        - bug | (display +banner): Returns bug or both display and banner
        - helper -extension:yml -extension:js: Returns helper in all files except files with a .yml or .js extension
        - helper path:lib/git: Returns helper in all files with a lib/git* path (for example, spec/lib/gitlab)
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
        # codespell:ignore-end

    @abstractmethod
    async def _execute(self, *args: Any, **kwargs: Any) -> str:
        pass

    async def _perform_search(self, id: str, params: dict, search_type: str) -> str:
        url = f"/api/v4/{search_type}/{id}/search"
        try:
            response = await self.gitlab_client.aget(
                path=url, params=params, use_http_response=True
            )

            if not response.is_success():
                log.error(
                    "Search request failed with status %s: %s",
                    response.status_code,
                    response.body,
                )
                return json.dumps({"error": str(response.body)})

            return json.dumps({"search_results": response.body})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: BaseSearchInput, _tool_response: Any = None
    ) -> str:
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

    async def _execute(
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

    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_ISSUE

    async def _execute(
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

    async def _execute(
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

    async def _execute(
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
    description: str = dedent(
        """
        Search file content in remote GitLab projects.

        Syntax for `search` parameter: keyword [filename:pattern] [path:dir/] [extension:type]
        - keyword: required, case-insensitive
        - filters: optional, space-separated, support `-` prefix excludes, exact literal matching only

        Valid examples:
        - rails filename:gemfile.lock
        - request filename:server -extension:rb path:src/core

        Invalid examples (DO NOT USE):
        - await run filename:*server* (WRONG - contains wildcards)
        """
    )
    args_schema: Type[BaseModel] = RefSearchInput

    def _filter_blob_results(self, results: list) -> list:
        """Filter blob search results using FileExclusionPolicy."""
        if not results:
            return results

        # Apply file exclusion policy and filter results
        policy = FileExclusionPolicy(self.project)
        filtered_results = []
        for result in results:
            file_path = result.get("path") or result.get("filename")
            if not file_path or policy.is_allowed(file_path):
                filtered_results.append(result)

        return filtered_results

    async def _execute(
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

        url = f"/api/v4/{search_type}/{id}/search"
        try:
            response = await self.gitlab_client.aget(
                path=url, params=params, use_http_response=True, parse_json=True
            )

            if not response.is_success():
                log.error(
                    "Blob search request failed",
                    status_code=response.status_code,
                    error=response.body,
                )
                return json.dumps({"search_results": []})
            # Filter blob results using FileExclusionPolicy
            filtered_response = self._filter_blob_results(response.body)
            return json.dumps({"search_results": filtered_response})
        except Exception as e:
            return json.dumps({"error": str(e)})


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

    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_COMMIT

    async def _execute(
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

    async def _execute(
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

    async def _execute(
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
