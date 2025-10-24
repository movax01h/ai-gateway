from abc import abstractmethod
from typing import Any, List, NamedTuple, Optional, final

from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.tool_output_manager import truncate_tool_response

DESCRIPTION_CHARACTER_LIMIT = 1_048_576


class ProjectURLValidationResult(NamedTuple):
    project_id: Optional[str]
    errors: List[str]


class PipelineValidationResult(NamedTuple):
    project_id: Optional[str]
    pipeline_iid: Optional[int]
    errors: List[str]


class MergeRequestValidationResult(NamedTuple):
    project_id: Optional[str]
    merge_request_iid: Optional[int]
    errors: List[str]


def format_tool_display_message(
    tool: BaseTool, args: Any, tool_response: Any = None
) -> Optional[str]:
    if not hasattr(tool, "format_display_message"):
        return None

    try:
        schema = getattr(tool, "args_schema", None)

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # type: ignore[arg-type]
            parsed = schema(**args)
            return tool.format_display_message(parsed, tool_response)

    except Exception:
        return DuoBaseTool.format_display_message(tool, args, tool_response)  # type: ignore[arg-type]

    return tool.format_display_message(args, tool_response)


class DuoBaseTool(BaseTool):
    unit_primitive: Optional[GitLabUnitPrimitive] = None
    eval_prompts: Optional[List[str]] = None

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

    @property
    def project(self) -> Project:
        return self.metadata and self.metadata.get("project")  # type: ignore

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool can only be run asynchronously")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_arun" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override _arun. "
                f"Implement _execute instead."
            )

    @final
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper that applies truncation to all tool results.

        This method should NOT be overridden by subclasses.
        """
        tool_result = await self._execute(*args, **kwargs)
        tool_response = truncate_tool_response(
            tool_response=tool_result, tool_name=self.name
        )
        return tool_response

    @abstractmethod
    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """Subclasses MUST implement this method instead of _arun.

        This is where the actual tool logic goes.
        """

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
                    f"Merge Request ID mismatch: provided '{merge_request_iid}' but URL contains "
                    f"'{url_merge_request_iid}'"
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

    def _validate_pipeline_url(
        self,
        url: str,
    ) -> PipelineValidationResult:
        """Validate pipeline URL and extract project_id and pipeline_iid.

        Args:
            url: The GitLab URL to parse

        Returns:
            PipelineValidationResult containing:
                - The validated project_id (or None if validation failed)
                - The validated pipeline_iid (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors: List[str] = []

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id, url_pipeline_iid = GitLabUrlParser.parse_pipeline_url(
                url, self.gitlab_host
            )

            # Use the IDs from the URL
            return PipelineValidationResult(url_project_id, url_pipeline_iid, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return PipelineValidationResult(
                None,
                None,
                errors,
            )

    def format_display_message(
        self, args: Any, _tool_response: Any = None
    ) -> Optional[str]:
        # Handle both dictionary and Pydantic model arguments
        if isinstance(args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        elif isinstance(args, BaseModel):
            # Handle Pydantic model instances
            args_str = ", ".join(f"{k}={v}" for k, v in args.model_dump().items())
        else:
            args_str = str(args)

        return f"Using {self.name}: {args_str}"

    @staticmethod
    def _process_http_response(identifier: str, response: Any) -> Any:
        if not isinstance(response, GitLabHttpResponse):
            return response

        if response.status_code >= 400:
            raise ToolException(
                f"Request failed ({identifier}): HTTP {response.status_code}: {str(response.body)[:300]}"
            )

        return response.body
