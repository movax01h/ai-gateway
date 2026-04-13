import json
from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    List,
    NamedTuple,
    Optional,
    Type,
    final,
    override,
)

import structlog
from langchain_core.tools import BaseTool, ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.tool_output_manager import (
    TruncationConfig,
    truncate_tool_response,
)

log = structlog.stdlib.get_logger("workflow")

DESCRIPTION_CHARACTER_LIMIT = 1_048_576

# editorconfig-checker-disable
QUICK_ACTIONS_WARNING = """
IMPORTANT: Do NOT include GitLab quick actions in the body, title, or description fields. Quick actions
are lines starting with / (such as /label, /assign, /milestone, /close, /reopen, /approve) and will
cause HTTP 403 errors with the message "Quick actions cannot be used with AI workflows". If you encounter
quick actions in a template, remove them and use the dedicated tool parameters instead
(e.g., use the labels parameter for labels, the assignee_ids parameter for assignments).
"""
# editorconfig-checker-enable

# Tools with a version below this threshold are considered experimental
# and are not exposed via the ListTools API.
STABLE_VERSION_THRESHOLD = Version("1.0.0")


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
            parsed = schema(**args)
            return tool.format_display_message(parsed, tool_response)

    except Exception:
        return DuoBaseTool.format_display_message(tool, args, tool_response)  # type: ignore[arg-type]

    return tool.format_display_message(args, tool_response)


class DuoBaseTool(BaseTool):
    eval_prompts: Optional[List[str]] = None

    # Default truncation configuration - tools can override this class attribute
    truncation_config: TruncationConfig = Field(default_factory=TruncationConfig)

    # Trust level for security wrapping (defaults to UNTRUSTED for fail-secure)
    # Tools that access only local filesystem/git should override to TRUSTED_INTERNAL
    trust_level: ToolTrustLevel = ToolTrustLevel.UNTRUSTED_USER_CONTENT

    # Tool class that this tool supersedes (if any)
    supersedes: ClassVar[Optional[Type["DuoBaseTool"]]] = None

    # Client capabilities required to use this tool (if any).
    # All capabilities in the frozenset must be declared by the client.
    required_capability: ClassVar[frozenset[str]] = frozenset()

    # Semantic version of the tool. Tools with version < 1.0.0 are experimental
    # and not exposed via the ListTools API.
    tool_version: ClassVar[Version] = Version("1.0.0")

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

    @override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool can only be run asynchronously")

    @override
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_arun" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override _arun. Implement _execute instead."
            )

    def _apply_tool_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply flow-level tool_options overrides to the given kwargs.

        Tool options take precedence over LLM-provided values for matching
        parameter names. This allows flows to enforce specific parameter
        values (e.g., internal=True) that the LLM cannot override.

        Args:
            kwargs: The keyword arguments to apply overrides to.

        Returns:
            The kwargs dict with overrides applied.
        """
        tool_opts = getattr(self, "_tool_options", {}).get(self.name, {})
        if tool_opts:
            kwargs.update(tool_opts)
        return kwargs

    @override
    @final
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper that applies truncation and security wrapping to all tool results.

        This method should NOT be overridden by subclasses.
        """
        kwargs = self._apply_tool_options(kwargs)
        tool_result = await self._execute(*args, **kwargs)

        # Apply truncation
        tool_response = truncate_tool_response(
            tool_response=tool_result,
            tool_name=self.name,
            truncation_config=self.truncation_config,
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

    async def _paginate_get(
        self,
        endpoint: str,
        per_page: int = 100,
        max_pages: int = 100,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """Fetch all pages from a paginated GitLab REST endpoint.

        Iterates through pages using the ``X-Next-Page`` response header until
        no further pages are available or ``max_pages`` is reached.  The safety
        cap on ``max_pages`` guards against infinite loops caused by a
        misbehaving server that keeps returning a next-page header.

        Args:
            endpoint: The API path to GET (e.g. ``/api/v4/projects/1/issues/2/notes``).
            per_page: Number of items to request per page (default 100, GitLab max).
            max_pages: Maximum number of pages to fetch before stopping (default 100).
            extra_params: Additional query parameters merged into every request.

        Returns:
            A flat list of all items collected across all pages.

        Raises:
            ToolException: If any page request returns a non-success HTTP status,
                the response body cannot be parsed as JSON, or the parsed response
                is not a list.
        """
        items: list[Any] = []
        next_page = "1"
        page_count = 0

        while next_page and page_count < max_pages:
            page_count += 1
            params: dict[str, Any] = {"page": next_page, "per_page": per_page}
            if extra_params:
                params.update(extra_params)

            response = await self.gitlab_client.aget(
                path=endpoint,
                params=params,
                parse_json=False,
            )

            if not response.is_success():
                log.error(
                    "Paginated GET request failed",
                    endpoint=endpoint,
                    status_code=response.status_code,
                    page=next_page,
                )
                raise ToolException(
                    f"Failed to fetch {endpoint}: HTTP {response.status_code}"
                )

            try:
                page_items = json.loads(response.body) if response.body else []
            except json.JSONDecodeError as e:
                body_snippet = (
                    response.body[:200] + "..."
                    if len(response.body) > 200
                    else response.body
                )
                raise ToolException(
                    f"Failed to parse JSON from {endpoint}: HTTP {response.status_code}. "
                    f"Response: {body_snippet}"
                ) from e

            if not isinstance(page_items, list):
                raise ToolException(
                    f"Unexpected response format from {endpoint}: expected list, got {type(page_items).__name__}"
                )
            items.extend(page_items)

            next_page = response.headers.get("X-Next-Page", "")

        return items

    async def _get_discussion_id_from_note_rest(
        self,
        project_id: str,
        resource_type: str,
        resource_iid: int,
        note_id: int,
    ) -> dict:
        """Resolve the REST discussion ID containing the given note_id.

        Args:
            project_id: The project ID
            resource_type: The type of resource ("issues" or "merge_requests")
            resource_iid: The IID of the resource (issue_iid or merge_request_iid)
            note_id: The ID of the note to find

        Returns:
            A dict with discussionId key if found

        Raises:
            ToolException: If the API call fails, the note is not found, or an error occurs
        """
        endpoint = (
            f"/api/v4/projects/{project_id}/{resource_type}/{resource_iid}/discussions"
        )
        try:
            discussions = await self._paginate_get(endpoint)

            for discussion in discussions:
                for note in discussion.get("notes", []):
                    if note.get("id") == note_id:
                        return {"discussionId": discussion["id"]}

            resource_name = (
                "merge request" if resource_type == "merge_requests" else "issue"
            )
            raise ToolException(f"Note {note_id} not found in this {resource_name}.")

        except ToolException:
            raise
        except Exception as e:
            raise ToolException(str(e)) from e

    @staticmethod
    def _process_http_response(
        identifier: str,
        response: Any,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ) -> Any:
        """Process HTTP response, logging errors and raising on failure.

        Args:
            identifier: Description of the operation (e.g., "get_user", "/api/v4/projects/1/issues")
            response: The HTTP response to handle
            logger: Optional logger instance. If not provided, uses structlog default

        Returns:
            response.body on success, or response itself if not a GitLabHttpResponse

        Raises:
            ToolException: If response is not successful (non-2xx status code)
        """
        if not isinstance(response, GitLabHttpResponse):
            return response

        if not response.is_success():
            if logger:
                logger.error(
                    f"{identifier} request failed",
                    status_code=response.status_code,
                    response_body=str(response.body)[:300],
                )

            raise ToolException(
                f"Request failed ({identifier}): HTTP {response.status_code}: {str(response.body)[:300]}"
            )

        return response.body
