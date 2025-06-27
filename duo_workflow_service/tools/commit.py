import json
from typing import Any, List, NamedTuple, Optional, Type

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

# editorconfig-checker-disable
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
# editorconfig-checker-enable


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
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_COMMIT

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
        description="When set to true the ref_name parameter is ignored. Default is false.",
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
        description="List commits in order. Possible values: default, topo. Default value: default "
        "(chronological order).",
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
        description="Only commits after or on this date are returned."
        "Use ISO 8601 format when specifying the date (YYYY-MM-DDTHH:MM:SSZ).",
    )
    trailers: Optional[bool] = Field(
        default=False,
        description="Parse and include Git trailers for every commit. Default is false.",
    )
    until: Optional[str] = Field(
        default=None,
        description="Only commits before or on this date are returned."
        "Use ISO 8601 format when specifying the date (YYYY-MM-DDTHH:MM:SSZ).",
    )
    with_stats: Optional[bool] = Field(
        default=False,
        description="Include commit stats. Default is false.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number. Default is 1.",
    )


class ListCommits(CommitBaseTool):
    name: str = "list_commits"

    # editorconfig-checker-disable
    description: str = f"""List commits in a GitLab project repository.
    By default, only returns the first 20 commits - use pagination to get complete results.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13, the tool call would be:
        list_commits(project_id=13)
    - Given the URL https://gitlab.com/namespace/project, the tool call would be:
        list_commits(url="https://gitlab.com/namespace/project")
    """
    # editorconfig-checker-enable
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
    # pylint: disable=line-too-long
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


class GetCommitDiff(CommitBaseTool):
    name: str = "get_commit_diff"
    description: str = f"""Get the diff of a specific commit in a GitLab project.

    {COMMIT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and commit_sha "6104942438c14ec7bd21c6cd5bd995272b3faff6", the tool call would be:
        get_commit_diff(project_id=13, commit_sha="6104942438c14ec7bd21c6cd5bd995272b3faff6")
    - Given the URL https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6, the tool call would be:
        get_commit_diff(url="https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6")
    """
    args_schema: Type[BaseModel] = CommitResourceInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        commit_sha = kwargs.get("commit_sha")

        project_id, commit_sha, errors = self._validate_commit_url(
            url, project_id, commit_sha
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/repository/commits/{commit_sha}/diff",
                parse_json=False,
            )
            return json.dumps({"diff": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: CommitResourceInput) -> str:
        if args.url:
            return f"Get diff for commit {args.url}"
        return f"Get diff for commit {args.commit_sha} in project {args.project_id}"


class GetCommitComments(CommitBaseTool):
    name: str = "get_commit_comments"
    description: str = f"""Get the comments on a specific commit in a GitLab project.

    {COMMIT_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and commit_sha "6104942438c14ec7bd21c6cd5bd995272b3faff6", the tool call would be:
        get_commit_comments(project_id=13, commit_sha="6104942438c14ec7bd21c6cd5bd995272b3faff6")
    - Given the URL https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6, the tool call would be:
        get_commit_comments(url="https://gitlab.com/namespace/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6")
    """
    args_schema: Type[BaseModel] = CommitResourceInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        commit_sha = kwargs.get("commit_sha")

        project_id, commit_sha, errors = self._validate_commit_url(
            url, project_id, commit_sha
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/repository/commits/{commit_sha}/comments",
                parse_json=False,
            )
            return json.dumps({"comments": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: CommitResourceInput) -> str:
        if args.url:
            return f"Get comments for commit {args.url}"
        return f"Get comments for commit {args.commit_sha} in project {args.project_id}"


class CreateCommitAction(BaseModel):
    """Model representing a single action to be performed in a commit."""

    action: str = Field(
        description="The action to perform: 'create', 'delete', 'move', or 'update'."
    )
    file_path: str = Field(
        description="Full path to the file. For example: 'lib/class.rb'."
    )
    previous_path: Optional[str] = Field(
        default=None,
        description="Original full path to the file being moved. Only considered for 'move' action.",
    )
    content: Optional[str] = Field(
        default=None,
        description="File content, required for 'create' and 'update' actions. For 'move' actions without content, existing file content is preserved.",
    )
    encoding: Optional[str] = Field(
        default=None, description="'text' or 'base64'. Default is 'text'."
    )
    last_commit_id: Optional[str] = Field(
        default=None,
        description="Last known file commit ID. Only considered in 'update', 'move', and 'delete' actions.",
    )


class CreateCommitInput(ProjectResourceInput):
    """Input model for creating a commit in a GitLab repository."""

    branch: str = Field(
        description="Name of the branch to commit into. To create a new branch, also provide either start_branch or start_sha, and optionally start_project."
    )
    commit_message: str = Field(description="Commit message.")
    actions: List[CreateCommitAction] = Field(
        description="JSON array of file actions. Each action requires 'action' and 'file_path'. For 'create' and 'update' actions, 'content' is also required."
    )
    start_branch: Optional[str] = Field(
        default=None, description="Name of the branch to start the new branch from."
    )
    start_sha: Optional[str] = Field(
        default=None, description="SHA of the commit to start the new branch from."
    )
    start_project: Optional[str] = Field(
        default=None,
        description="The ID or URL-encoded path of the project to start the new branch from.",
    )
    author_email: Optional[str] = Field(
        default=None, description="Author's email address."
    )
    author_name: Optional[str] = Field(default=None, description="Author's name.")


class CreateCommit(DuoBaseTool):
    """Tool to create a commit with multiple file actions in a GitLab repository."""

    name: str = "create_commit"

    # editorconfig-checker-disable
    description: str = """Create a commit with multiple file actions in a GitLab repository.

    To identify the project you must provide either:
    - project_id parameter, or
    - A GitLab URL like:
      - https://gitlab.com/namespace/project
      - https://gitlab.com/namespace/project/-/commits
      - https://gitlab.com/group/subgroup/project
      - https://gitlab.com/group/subgroup/project/-/commits

    Actions can include creating, updating, deleting, moving, or changing file permissions.
    Each action requires at minimum an 'action' type and 'file_path'.

    For example:
    - Creating a new file requires 'action': 'create', 'file_path', and 'content'
    - Updating a file requires 'action': 'update', 'file_path', and 'content'
    - Deleting a file requires 'action': 'delete' and 'file_path'
    - Moving a file requires 'action': 'move', 'file_path', and 'previous_path'
    """
    # editorconfig-checker-enable

    args_schema: Type[BaseModel] = CreateCommitInput  # type: ignore

    async def _arun(
        self,
        branch: str,
        commit_message: str,
        actions: List[CreateCommitAction],
        **kwargs: Any,
    ) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        # Convert actions to dictionary format for API
        actions_data = []
        for action in actions:
            if action.action in ["create", "update", "delete", "move"]:
                action_dict = action.model_dump(exclude_none=True)
                actions_data.append(action_dict)

        # Prepare request parameters
        params = {
            "branch": branch,
            "commit_message": commit_message,
            "actions": actions_data,
        }

        # Add optional parameters if provided
        optional_params = [
            "start_branch",
            "start_sha",
            "start_project",
            "author_email",
            "author_name",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                params[param] = kwargs[param]

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/projects/{project_id}/repository/commits",
                body=json.dumps(params),
            )
            return json.dumps(
                {"status": "success", "data": params, "response": response}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: CreateCommitInput) -> str:
        """Format a user-friendly message describing the action being performed."""
        action_types = [action.action for action in args.actions]
        file_count = len(args.actions)

        if args.url:
            return f"Create commit in {args.url} with {file_count} file {self._pluralize('action', file_count)} ({', '.join(action_types)})"
        return f"Create commit in project {args.project_id} with {file_count} file {self._pluralize('action', file_count)} ({', '.join(action_types)})"

    def _pluralize(self, word: str, count: int) -> str:
        """Helper method to pluralize words based on count."""
        return f"{word}s" if count != 1 else word
