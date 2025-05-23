import json
from typing import Any, List, NamedTuple, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

PROJECT_IDENTIFICATION_DESCRIPTION = """To identify the project you must provide either:
- project_id parameter, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project
  - https://gitlab.com/namespace/project/-/commits
  - https://gitlab.com/group/subgroup/project
  - https://gitlab.com/group/subgroup/project/-/commits
"""

COMMIT_IDENTIFICATION_DESCRIPTION = """To identify a commit you must provide either:
- project_id and commit_sha, or
- A GitLab URL like:
  - https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6
  - https://gitlab.com/group/subgroup/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6
"""


class CommitResourceInput(ProjectResourceInput):
    commit_sha: Optional[str] = Field(
        default=None,
        description="The SHA hash of the commit. Required if URL is not provided.",
    )


class CommitURLValidationResult(NamedTuple):
    project_id: Optional[str]
    commit_sha: Optional[str]
    errors: List[str]


class CommitBaseTool(DuoBaseTool):
    def _validate_commit_url(
        self, url: Optional[str], project_id: Optional[Any], commit_sha: Optional[str]
    ) -> CommitURLValidationResult:
        """Validate commit URL and extract project_id and commit_sha.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user
            commit_sha: The commit SHA provided by the user

        Returns:
            CommitURLValidationResult containing:
                - The validated project_id (or None if validation failed)
                - The validated commit_sha (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is absent")
            if not commit_sha:
                errors.append("'commit_sha' must be provided when 'url' is absent")
            return CommitURLValidationResult(
                str(project_id) if project_id is not None else None, commit_sha, errors
            )

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id, url_commit_sha = GitLabUrlParser.parse_commit_url(
                url, self.gitlab_host
            )

            # If both URL and IDs are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )
            if commit_sha is not None and commit_sha != url_commit_sha:
                errors.append(
                    f"Commit SHA mismatch: provided '{commit_sha}' but URL contains '{url_commit_sha}'"
                )

            return CommitURLValidationResult(url_project_id, url_commit_sha, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return CommitURLValidationResult(
                str(project_id) if project_id is not None else None, commit_sha, errors
            )


class ListCommitsInput(ProjectResourceInput):
    all: Optional[bool] = Field(
        default=False,
        description="Retrieve every commit from the repository. Default is false.",
    )
    author: Optional[str] = Field(
        default=None,
        description="Search commits by commit author.",
    )
    first_parent: Optional[bool] = Field(
        default=False,
        description="Follow only the first parent commit upon seeing a merge commit. Default is false.",
    )
    order: Optional[str] = Field(
        default=None,
        description="List commits in order. Possible values: default, topo. Default value: default (chronological order).",
    )
    path: Optional[str] = Field(
        default=None,
        description="The file path to filter commits by.",
    )
    ref_name: Optional[str] = Field(
        default=None,
        description="The name of a repository branch or tag to list commits from. Default is the default branch.",
    )
    since: Optional[str] = Field(
        default=None,
        description="Only commits after or on this date are returned in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).",
    )
    trailers: Optional[bool] = Field(
        default=False,
        description="Parse and include Git trailers for every commit. Default is false.",
    )
    until: Optional[str] = Field(
        default=None,
        description="Only commits before or on this date are returned in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).",
    )
    with_stats: Optional[bool] = Field(
        default=False,
        description="Include commit stats. Default is false.",
    )


class ListCommits(CommitBaseTool):
    name: str = "list_commits"
    description: str = f"""List commits in a GitLab project repository.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, the tool call would be:
        list_commits(project_id=13)
    - Given the URL https://gitlab.com/namespace/project, the tool call would be:
        list_commits(url="https://gitlab.com/namespace/project")
    """
    args_schema: Type[BaseModel] = ListCommitsInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        params = {k: v for k, v in kwargs.items() if v is not None}

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/repository/commits",
                params=params,
                parse_json=False,
            )
            return json.dumps({"commits": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListCommitsInput) -> str:
        if args.url:
            return f"List commits in {args.url}"
        return f"List commits in project {args.project_id}"


class GetCommitInput(CommitResourceInput):
    stats: Optional[bool] = Field(
        default=None,
        description="Include commit stats (additions, deletions, total). Default is true.",
    )


class GetCommit(CommitBaseTool):
    name: str = "get_commit"
    description: str = f"""Get a single commit from a GitLab project repository.

    {COMMIT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and commit_sha "6104942438c14ec7bd21c6cd5bd995272b3faff6", the tool call would be:
        get_commit(project_id=13, commit_sha="6104942438c14ec7bd21c6cd5bd995272b3faff6")
    - Given the URL https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6, the tool call would be:
        get_commit(url="https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6")
    """
    args_schema: Type[BaseModel] = GetCommitInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        commit_sha = kwargs.get("commit_sha")
        stats = kwargs.get("stats")

        validation_result = self._validate_commit_url(url, project_id, commit_sha)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        params = {}
        if stats is not None:
            params["stats"] = str(stats).lower()

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/repository/commits/{validation_result.commit_sha}",
                params=params,
                parse_json=False,
            )
            return json.dumps({"commit": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: GetCommitInput) -> str:
        if args.url:
            return f"Read commit {args.url}"
        return f"Read commit {args.commit_sha} in project {args.project_id}"
