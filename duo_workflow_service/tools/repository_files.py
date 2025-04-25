import json
from typing import List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput


class RepositoryFileResourceInput(ProjectResourceInput):
    ref: Optional[str] = Field(
        default=None,
        description="The name of branch, tag or commit. Use HEAD to automatically use the default branch.",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="URL encoded full path to file, such as lib%2Fclass%2Erb.",
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
            errors.append(f"Failed to parse URL: {str(e)}")
            return project_id, ref, file_path, errors


class GetRepositoryFile(RepositoryFileBaseTool):
    name: str = "get_repository_file"
    description: str = """Get the contents of a file from a remote repository.

    To identify a file you must provide either:
    - project_id, ref, and file_path parameters, or
    - A GitLab URL like:
      - https://gitlab.com/namespace/project/-/blob/master/README.md
      - https://gitlab.com/group/subgroup/project/-/blob/main/src/file.py
    """
    args_schema: Type[BaseModel] = RepositoryFileResourceInput  # type: ignore

    async def _arun(self, **kwargs) -> str:
        url = kwargs.get("url")
        project_id = kwargs.get("project_id")
        ref = kwargs.get("ref")
        file_path = kwargs.get("file_path")

        project_id, ref, file_path, errors = self._validate_repository_file_url(
            url, project_id, ref, file_path
        )

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/repository/files/{file_path}/raw",
                params={"ref": ref},
                parse_json=False,
            )

            if self._is_binary_string(response):
                return json.dumps(
                    {
                        "error": f"Binary file detected: {file_path}. Only text files are supported."
                    }
                )

            return json.dumps({"content": response})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: RepositoryFileResourceInput) -> str:
        if args.url:
            return f"Get repository file content from {args.url}"
        return f"Get repository file {args.file_path} from project {args.project_id} at ref {args.ref}"

    def _is_binary_string(self, content):
        """
        Detect if content is binary using Perl-inspired heuristic.
        https://code.activestate.com/recipes/173220-test-if-a-file-or-string-is-text-or-binary/
        Returns True if content appears to be binary, False if it's text.
        """
        # Empty content is considered text
        if not content:
            return False

        try:
            # Check for null bytes which are common in binary files
            if b"\x00" in content.encode("utf-8"):
                return True

            text_chars = bytes(range(32, 127)) + b"\n\r\t\b"

            if isinstance(content, str):
                try:
                    # If we can encode/decode as UTF-8, it's likely text
                    content.encode("utf-8").decode("utf-8")
                    return False
                except UnicodeError:
                    # Not valid UTF-8 text
                    content_bytes = content.encode("latin-1")
            else:
                content_bytes = content

            sample = content_bytes[:1024]
            non_text_chars = sum(byte not in text_chars for byte in sample)

            # If more than 30% non-text characters, consider it binary
            if len(sample) > 0 and float(non_text_chars) / float(len(sample)) > 0.3:
                return True

            return False

        except UnicodeError:
            # If we can't encode to UTF-8, it's likely binary
            return True
