import asyncio
import base64
import fnmatch
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from urllib.parse import quote

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, PrivateAttr

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput

log = structlog.stdlib.get_logger(__name__)

DEFAULT_GET_REPOSITORY_FILE_LIMIT = 2000
DEFAULT_REPOSITORY_FILES_CONCURRENCY = 4

_GLOB_METACHARACTERS = frozenset("*?[]")


class RepositoryFileResourceInput(ProjectResourceInput):
    ref: Optional[str] = Field(
        default=None,
        description="The name of branch, tag or commit. Use HEAD to automatically use the default branch.",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Full path to file, such as lib/class.rb.",
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,
        description="Starting line number (0-indexed). Use with limit for reading large files in chunks.",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of lines to read from offset. Use for reading large files in chunks.",
    )


class RepositoryFileBaseTool(DuoBaseTool):
    def _validate_repository_file_url(
        self,
        url: Optional[str],
        project_id: Optional[str],
        ref: Optional[str],
        file_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Validate repository file URL and extract project_id, ref, and file_path.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user
            ref: The ref provided by the user
            file_path: The file path provided by the user

        Returns:
            Tuple containing:
                - The validated project_id (or None if validation failed)
                - The validated ref (or None if validation failed)
                - The validated file_path (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is not")
            if not ref:
                errors.append("'ref' must be provided when 'url' is not")
            if not file_path:
                errors.append("'file_path' must be provided when 'url' is not")
            return project_id, ref, file_path, errors

        try:
            url_project_id, url_ref, url_file_path = (
                GitLabUrlParser.parse_repository_file_url(url, self.gitlab_host)
            )

            # If both URL and IDs are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )
            if ref is not None and ref != url_ref:
                errors.append(
                    f"Ref mismatch: provided '{ref}' but URL contains '{url_ref}'"
                )
            if file_path and file_path != url_file_path:
                errors.append(
                    f"File path mismatch: provided '{file_path}' but URL contains '{url_file_path}'"
                )

            # Use the values from the URL
            return url_project_id, url_ref, url_file_path, errors
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {e!s}")
            return project_id, ref, file_path, errors


class GetRepositoryFile(RepositoryFileBaseTool):
    name: str = "get_repository_file"

    # editorconfig-checker-disable
    description: str = """Get the contents of a file from a remote repository.

    To identify a file you must provide either:
    - project_id, ref, and file_path parameters, or
    - A GitLab SaaS URL like:
      - https://gitlab.com/namespace/project/-/blob/master/README.md
      - https://gitlab.com/group/subgroup/project/-/blob/main/src/file.py
    - A self-managed GitLab URL like:
      - https://gitlab.example.com/namespace/project/-/blob/master/README.md
      - https://gitlab.example.com/group/subgroup/project/-/blob/main/src/file.py

    For large files, use offset and limit to read in chunks:
    - offset: starting line number (0-indexed).
    - limit: number of lines to read from that offset.
    - If the output is truncated, a hint at the end shows the next offset to continue reading.
    - Avoid reading the entire file when you only need a specific section.
    """
    # editorconfig-checker-enable

    args_schema: Type[BaseModel] = RepositoryFileResourceInput

    # gitlab-org/gitlab#604564: remember (project, ref, path) tuples that 404'd this
    # session so the agent's repeated guesses for the same missing file short-circuit
    # instead of re-hitting the API (a failed run made 325 calls, 113 for one missing
    # path, exhausting the step budget).
    _seen_missing_paths: Set[Tuple[str, str, str]] = PrivateAttr(default_factory=set)

    async def _execute(self, **kwargs) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        ref = kwargs.get("ref")
        file_path = kwargs.get("file_path")
        offset = kwargs.get("offset")
        limit = kwargs.get("limit")

        project_id, ref, file_path, errors = self._validate_repository_file_url(
            url, project_id, ref, file_path
        )

        if errors:
            raise ToolException("; ".join(errors))

        if file_path is None:
            raise ToolException("Missing file_path")

        # gitlab-org/gitlab#604564: short-circuit a path already known to 404 this session.
        cache_key = (str(project_id), str(ref), file_path)
        if cache_key in self._seen_missing_paths:
            raise ToolException(
                f"File '{file_path}' (ref '{ref}') already returned 404 earlier in this "
                f"session and does not exist — not re-fetching. Do NOT request it again; "
                f"continue with the files you already have."
            )

        # Check file exclusion policy if project is available
        policy = FileExclusionPolicy(self.project)
        if file_path and not policy.is_allowed(file_path):
            raise ToolException(
                FileExclusionPolicy.format_llm_exclusion_message([file_path])
            )

        encoded_file_path = quote(file_path, safe="")

        response = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/repository/files/{encoded_file_path}",
            params={"ref": ref},
            parse_json=True,
        )

        # gitlab-org/gitlab#604564: cache 404s so identical re-requests short-circuit above.
        if isinstance(response, GitLabHttpResponse) and response.status_code == 404:
            self._seen_missing_paths.add(cache_key)

        body = self._process_http_response("Get repository file", response, log)

        # Note: The GitLab API does not support range queries on file content,
        # so we always fetch the full file and slice client-side.
        content = base64.b64decode(body["content"]).decode("utf-8")

        # Paginate when explicitly requested or when file is large
        if (
            offset is not None
            or limit is not None
            or content.count("\n") >= DEFAULT_GET_REPOSITORY_FILE_LIMIT
        ):
            content = self._paginate(content, offset, limit)

        return json.dumps({"content": content})

    @staticmethod
    def _paginate(
        content: str,
        offset: Optional[int],
        limit: Optional[int],
    ) -> str:
        lines = content.rstrip("\n").splitlines()
        total_lines = len(lines)

        start = offset if offset is not None else 0
        if limit is not None and limit > 0:
            end = min(start + limit, total_lines)
        elif limit is None and total_lines > DEFAULT_GET_REPOSITORY_FILE_LIMIT:
            end = DEFAULT_GET_REPOSITORY_FILE_LIMIT
        else:
            end = total_lines

        if start >= total_lines and (total_lines > 0 or start > 0):
            return (
                f"[Offset {start} is beyond end of file ({total_lines} lines). "
                f"Use offset=0 to read from the start.]"
            )

        result = "\n".join(lines[start:end])

        if end < total_lines:
            result += (
                f"\n\n[Showing lines {start}-{end - 1} of {total_lines} total. "
                f"Use offset={end} to continue reading.]"
            )
        elif start > 0:
            result += f"\n\n[Showing lines {start}-{end - 1} of {total_lines} total.]"

        return result

    def format_display_message(
        self, args: RepositoryFileResourceInput, _tool_response: Any = None
    ) -> str:
        # Check file exclusion policy for display message if project is available
        policy = FileExclusionPolicy(self.project)
        if args.file_path and not policy.is_allowed(args.file_path):
            return FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        if args.url:
            return f"Get repository file content from {args.url}"
        return f"Get repository file {args.file_path} from project {args.project_id} at ref {args.ref}"


class RepositoryTreeResourceInput(ProjectResourceInput):
    path: Optional[str] = Field(
        default=None,
        description="Path inside repository. Used to get content of subdirectories",
    )
    ref: Optional[str] = Field(
        default=None,
        description="The name of a repository branch or tag or, if not given, the default branch",
    )
    recursive: Optional[bool] = Field(
        default=False,
        description="Boolean value for getting a recursive tree",
    )
    page: Optional[int] = Field(
        default=1,
        description="Page number for pagination (min 1)",
        ge=1,
    )
    per_page: Optional[int] = Field(
        default=20,
        description="Results per page for pagination (min 1, max 100)",
        ge=1,
        le=100,
    )


class ListRepositoryTree(DuoBaseTool):
    name: str = "list_repository_tree"
    description: str = """List files and directories in a GitLab repository.

    To identify a project you must provide either:
    - project_id parameter, or
    - A GitLab SaaS URL like:
        - https://gitlab.com/namespace/project
        - https://gitlab.com/group/subgroup/project
    - A self-managed GitLab URL like:
        - https://gitlab.example.com/namespace/project
        - https://gitlab.example.com/group/subgroup/project

    You can specify a path to get contents of a subdirectory, a specific ref (branch/tag),
    and whether to get a recursive tree.

    For example:
    - Given project_id 13, the tool call would be:
        list_repository_tree(project_id=13)
    - To list files in a specific subdirectory with a specific branch:
        list_repository_tree(project_id=13, path="src", ref="main")
    - To recursively list all files in a project:
        list_repository_tree(project_id=13, recursive=True)
    """
    args_schema: Type[BaseModel] = RepositoryTreeResourceInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, **kwargs) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            raise ToolException("; ".join(errors))

        params = {}
        optional_params = ["path", "ref", "page", "per_page"]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                params[param] = kwargs[param]

        recursive = kwargs.get("recursive", None)
        if recursive is not None:
            params["recursive"] = str(recursive).lower()

        response = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/repository/tree",
            params=params,
        )

        if not response.is_success():
            log.error(
                "List repository tree request failed with status %s: %s",
                response.status_code,
                response.body,
            )
            raise ToolException(
                f"List repository tree request failed with status {response.status_code}: {response.body}"
            )

        # Filter results based on file exclusion policy
        policy = FileExclusionPolicy(self.project)

        # Extract file paths from the response objects
        file_paths: List[str] = [
            item.get("path", "")
            for item in response.body
            if isinstance(item.get("path"), str)
        ]
        allowed_paths, _excluded_paths = policy.filter_allowed(file_paths)

        # Filter the original response to only include allowed items
        filtered_response = [
            item for item in response.body if item.get("path") in allowed_paths
        ]

        return json.dumps({"tree": filtered_response})

    def format_display_message(
        self, args: RepositoryTreeResourceInput, _tool_response: Any = None
    ) -> str:
        path_str = f" in path '{args.path}'" if args.path else ""
        ref_str = f" at ref '{args.ref}'" if args.ref else ""
        recursive_str = " recursively" if args.recursive else ""

        if args.url:
            return f"List repository tree{recursive_str}{path_str}{ref_str} from {args.url}"
        return f"List repository tree{recursive_str}{path_str}{ref_str} in project {args.project_id}"


def _is_glob_pattern(path: str) -> bool:
    """Return True if *path* contains any fnmatch glob metacharacter.

    Args:
        path: A file path or glob pattern string.

    Returns:
        True when the path contains ``*``, ``?``, ``[``, or ``]``; False otherwise.
    """
    return bool(_GLOB_METACHARACTERS.intersection(path))


class RepositoryFilesResourceInput(ProjectResourceInput):
    """Input schema for GetRepositoryFiles tool."""

    file_paths: List[str] = Field(
        description=(
            "List of file paths or fnmatch-style glob patterns (e.g. '*.py', 'src/*.py'). "
            "Explicit paths and glob patterns may be mixed freely."
        ),
    )
    ref: Optional[str] = Field(
        default=None,
        description="The name of branch, tag or commit. Use HEAD to automatically use the default branch.",
    )
    url: Optional[str] = Field(
        default=None,
        description="GitLab URL for the project. If provided, project_id is not required.",
    )
    per_page: Optional[int] = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of files to return (default 20, max 50).",
    )


class GetRepositoryFiles(RepositoryFileBaseTool):
    """Bulk-read multiple files from a remote GitLab repository.

    Accepts a mix of explicit paths and fnmatch-style glob patterns.  Glob
    patterns trigger a recursive tree listing to discover matching blobs;
    explicit paths are fetched directly without any tree call.  Content
    fetches are parallelised with a concurrency cap of
    ``DEFAULT_REPOSITORY_FILES_CONCURRENCY``.
    """

    name: str = "get_repository_files"

    # editorconfig-checker-disable
    description: str = """Get the contents of multiple files from a remote GitLab repository in a single call.

    Accepts a mix of explicit file paths and fnmatch-style glob patterns (e.g. '*.py', 'src/*.py').
    Glob patterns trigger a recursive tree listing to discover matching blobs; explicit paths are
    fetched directly without any tree call.

    To identify a project you must provide either:
    - project_id and ref parameters, or
    - A GitLab SaaS URL like:
      - https://gitlab.com/namespace/project/-/blob/master/README.md
    - A self-managed GitLab URL like:
      - https://gitlab.example.com/namespace/project/-/blob/master/README.md

    Parameters:
    - file_paths: list of explicit paths and/or glob patterns (e.g. ['README.md', 'src/*.py'])
    - ref: branch, tag, or commit SHA (required when project_id is used)
    - per_page: maximum number of files to return (default 20, max 50)

    Returns a JSON object mapping each resolved file path to either:
    - {"content": "..."} on success
    - {"error": "..."} on failure (404, decode error, exclusion policy, etc.)

    When the number of matched files exceeds per_page, a "__truncated__" key is added to the
    result indicating how many total matches were found vs. returned.
    """
    # editorconfig-checker-enable

    args_schema: Type[BaseModel] = RepositoryFilesResourceInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, **kwargs: Any) -> str:
        """Execute the bulk file read.

        Args:
            **kwargs: Keyword arguments matching RepositoryFilesResourceInput fields.

        Returns:
            JSON string mapping file paths to their content or error.

        Raises:
            ToolException: When project/ref resolution fails or the tree listing fails.
        """
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        ref = kwargs.get("ref")
        file_paths: List[str] = kwargs["file_paths"]
        per_page: int = kwargs.get("per_page") or 20

        project_id, url_errors = self._validate_project_url(url, project_id)
        if url_errors:
            raise ToolException("; ".join(url_errors))

        if not project_id:
            raise ToolException("'project_id' is required")
        if not ref:
            raise ToolException("'ref' is required")

        explicit_paths = [p for p in file_paths if not _is_glob_pattern(p)]
        glob_patterns = [p for p in file_paths if _is_glob_pattern(p)]

        glob_matched: List[str] = []
        if glob_patterns:
            glob_matched = await self._resolve_glob_patterns(
                project_id, ref, glob_patterns
            )

        seen: Dict[str, None] = {}
        for path in explicit_paths + glob_matched:
            seen[path] = None
        combined = list(seen.keys())

        policy = FileExclusionPolicy(self.project)
        allowed_paths, excluded_paths = policy.filter_allowed(combined)

        total_matched = len(allowed_paths)
        truncated = total_matched > per_page
        paths_to_fetch = allowed_paths[:per_page]

        results: Dict[str, Any] = {}
        semaphore = asyncio.Semaphore(DEFAULT_REPOSITORY_FILES_CONCURRENCY)

        async def fetch_one(path: str) -> Tuple[str, Dict[str, str]]:
            async with semaphore:
                return path, await self._fetch_file_content(project_id, ref, path)

        fetch_tasks = [fetch_one(p) for p in paths_to_fetch]
        fetch_results = await asyncio.gather(*fetch_tasks)
        for path, result in fetch_results:
            results[path] = result

        for path in excluded_paths:
            results[path] = {
                "error": FileExclusionPolicy.format_llm_exclusion_message([path])
            }

        if truncated:
            results["__truncated__"] = {
                "total_matched": total_matched,
                "returned": per_page,
                "message": (
                    f"Results truncated: {total_matched} files matched but only {per_page} returned. "
                    f"Refine your glob pattern or pass explicit paths for the remainder."
                ),
            }

        return json.dumps(results, indent=2)

    async def _resolve_glob_patterns(
        self, project_id: str, ref: str, patterns: List[str]
    ) -> List[str]:
        """Enumerate all blob paths in the repository tree and filter by glob patterns.

        Args:
            project_id: The GitLab project ID or path.
            ref: The branch, tag, or commit SHA.
            patterns: List of fnmatch-style glob patterns to match against.

        Returns:
            Ordered list of blob paths that match at least one pattern.

        Raises:
            ToolException: If the tree listing API call fails.
        """
        tree_entries = await self._paginate_get(
            f"/api/v4/projects/{project_id}/repository/tree",
            per_page=100,
            extra_params={"ref": ref, "recursive": "true"},
        )

        matched: List[str] = []
        for entry in tree_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "blob":
                continue
            entry_path = entry.get("path", "")
            if not entry_path:
                continue
            if any(fnmatch.fnmatch(entry_path, pattern) for pattern in patterns):
                matched.append(entry_path)

        return matched

    async def _fetch_file_content(
        self, project_id: str, ref: str, file_path: str
    ) -> Dict[str, str]:
        """Fetch and decode the content of a single file.

        Args:
            project_id: The GitLab project ID or path.
            ref: The branch, tag, or commit SHA.
            file_path: The path of the file to fetch.

        Returns:
            Dict with ``"content"`` key on success, or ``"error"`` key on failure.
        """
        encoded_path = quote(file_path, safe="")
        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/repository/files/{encoded_path}",
                params={"ref": ref},
                parse_json=True,
            )
            body = self._process_http_response("Get repository file", response, log)
            content = base64.b64decode(body["content"]).decode("utf-8")
            if content.count("\n") >= DEFAULT_GET_REPOSITORY_FILE_LIMIT:
                content = GetRepositoryFile._paginate(content, None, None)
            return {"content": content}
        except ToolException as exc:
            return {"error": str(exc)}
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "Unexpected error fetching file content",
                file_path=file_path,
                exc_info=exc,
            )
            return {"error": str(exc)}

    def format_display_message(
        self, args: RepositoryFilesResourceInput, _tool_response: Any = None
    ) -> str:
        """Return a human-readable summary of the tool invocation.

        Args:
            args: The validated input arguments.
            _tool_response: Unused tool response (reserved for future use).

        Returns:
            A summary string describing what was requested.
        """
        explicit_paths = [p for p in args.file_paths if not _is_glob_pattern(p)]
        glob_patterns = [p for p in args.file_paths if _is_glob_pattern(p)]
        per_page = args.per_page or 20

        parts: List[str] = []
        if explicit_paths:
            parts.append(f"{len(explicit_paths)} explicit path(s)")
        if glob_patterns:
            pattern_list = ", ".join(f"`{p}`" for p in glob_patterns)
            parts.append(f"{len(glob_patterns)} glob pattern(s): {pattern_list}")

        summary = "Get repository files: " + (", ".join(parts) if parts else "no paths")

        if args.url:
            summary += f" from {args.url}"
        else:
            summary += f" from project {args.project_id} at ref {args.ref}"

        summary += f" (up to {per_page} files)"
        return summary
