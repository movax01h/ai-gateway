from typing import Any, List, NamedTuple, Optional

from langchain.tools import BaseTool

from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser


class ProjectURLValidationResult(NamedTuple):
    project_id: Optional[str]
    errors: List[str]


class MergeRequestValidationResult(NamedTuple):
    project_id: Optional[str]
    merge_request_iid: Optional[int]
    errors: List[str]


def format_tool_display_message(tool: BaseTool, args: Any) -> Optional[str]:
    if not hasattr(tool, "format_display_message"):
        return None

    if not hasattr(tool, "args_schema") or tool.args_schema is None:
        return tool.format_display_message(args)

    try:
        pydantic_args = tool.args_schema(**args)
        return tool.format_display_message(pydantic_args)
    except Exception:
        return tool.format_display_message(args)


class DuoBaseTool(BaseTool):
    @property
    def gitlab_client(self) -> GitlabHttpClient:
        client = self.metadata.get("gitlab_client")  # type: ignore
        if not client:
            raise RuntimeError("gitlab_client is not set")
        return client

    @property
    def gitlab_host(self) -> str:
        host = self.metadata.get("gitlab_host")  # type: ignore
        if not host:
            raise RuntimeError("gitlab_host is not set")
        return host

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool can only be run asynchronously")

    def _validate_project_url(
        self, url: Optional[str], project_id: Optional[int | str]
    ) -> ProjectURLValidationResult:
        """Validate project URL and extract project_id.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user

        Returns:
            ProjectURLValidationResult containing:
                - The validated project_id (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is not")
            return ProjectURLValidationResult(str(project_id), errors)

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id = GitLabUrlParser.parse_project_url(url, self.gitlab_host)

            # If both URL and project_id are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )

            # Use the project_id from the URL
            return ProjectURLValidationResult(url_project_id, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return ProjectURLValidationResult(str(project_id), errors)

    def _validate_merge_request_url(
        self,
        url: Optional[str],
        project_id: Optional[int | str],
        merge_request_iid: Optional[int],
    ) -> MergeRequestValidationResult:
        """Validate merge request URL and extract project_id and merge_request_iid.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user
            merge_request_iid: The merge request IID provided by the user

        Returns:
            MergeRequestValidationResult containing:
                - The validated project_id (or None if validation failed)
                - The validated merge_request_iid (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is not")
            if not merge_request_iid:
                errors.append("'merge_request_iid' must be provided when 'url' is not")
            return MergeRequestValidationResult(
                None if project_id is None else str(project_id),
                merge_request_iid,
                errors,
            )

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id, url_merge_request_iid = (
                GitLabUrlParser.parse_merge_request_url(url, self.gitlab_host)
            )

            # If both URL and IDs are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )
            if (
                merge_request_iid is not None
                and merge_request_iid != url_merge_request_iid
            ):
                errors.append(
                    f"Merge Request ID mismatch: provided '{merge_request_iid}' but URL contains '{url_merge_request_iid}'"
                )

            # Use the IDs from the URL
            return MergeRequestValidationResult(
                url_project_id, url_merge_request_iid, errors
            )
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return MergeRequestValidationResult(
                None if project_id is None else str(project_id),
                merge_request_iid,
                errors,
            )

    def format_display_message(self, args: Any) -> Optional[str]:
        # Handle both dictionary and Pydantic model arguments
        if isinstance(args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        elif hasattr(args, "dict"):
            # Handle Pydantic model instances
            args_str = ", ".join(f"{k}={v}" for k, v in args.dict().items())
        else:
            args_str = str(args)

        return f"Using {self.name}: {args_str}"
