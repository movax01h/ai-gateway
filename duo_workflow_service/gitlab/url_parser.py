import re
from typing import List, Tuple
from urllib.parse import quote, unquote, urlparse

PROJECT_URL_REGEX = r"^(.+?)(?:/-/.*)?$"
ISSUE_URL_REGEX = r"^(.+?)/-/issues/(\d+)"
GROUP_URL_REGEX = r"^(?:groups/)?(.+?)(?:/-/.*)?$"
EPIC_URL_REGEX = r"^(?:groups/)?(.+?)/-/epics/(\d+)"
MR_URL_REGEX = r"^(.+?)/-/merge_requests/(\d+)"
JOB_URL_REGEX = r"^(.+?)/-/jobs/(\d+)"
REPOSITORY_FILE_URL_REGEX = r"^(.+?)/-/blob/([^/]+)/(.+)$"
COMMIT_URL_REGEX = r"^(.+?)/-/commit/([a-fA-F0-9]{5,40})"


class GitLabUrlParseError(Exception):
    """Exception raised when a GitLab URL cannot be parsed correctly."""

    pass


class GitLabUrlParser:
    """Utility class for parsing GitLab URLs into their component IDs."""

    @staticmethod
    def _validate_url_netloc(url: str, gitlab_host: str) -> None:
        """Validate that the URL's netloc matches the gitlab_host.

        Args:
            url: The URL to validate
            gitlab_host: The GitLab host to compare against

        Raises:
            GitLabUrlParseError: If the netloc doesn't match gitlab_host
        """
        try:
            parsed_url = urlparse(url)
            netloc = parsed_url.netloc

            if netloc != gitlab_host:
                raise GitLabUrlParseError(
                    f"URL netloc '{netloc}' does not match gitlab_host '{gitlab_host}'"
                )
        except Exception as e:
            if isinstance(e, GitLabUrlParseError):
                raise
            raise GitLabUrlParseError(f"Could not validate URL netloc: {url}") from e

    @staticmethod
    def extract_host_from_url(url: str) -> str:
        """Extract the host from a GitLab URL.

        Args:
            url: The GitLab URL to parse

        Returns:
            The host part of the URL (e.g., 'gitlab.com')

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed
        """
        try:
            parsed_url = urlparse(url)
            host = parsed_url.netloc
            if not host:
                raise GitLabUrlParseError(f"Could not extract host from URL: {url}")
            return host
        except Exception as e:
            raise GitLabUrlParseError(f"Could not extract host from URL: {url}") from e

    @staticmethod
    def _extract_path_components(
        url: str, pattern: str, error_message: str
    ) -> List[str]:
        """Extract components from a URL path using a regex pattern.

        Args:
            url: The GitLab URL to parse
            pattern: Regex pattern to match against the path
            error_message: Error message to use if parsing fails

        Returns:
            List of matched groups from the regex

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed with the given pattern
        """
        try:
            parsed_url = urlparse(url)

            path = parsed_url.path.strip("/")

            if not path:
                raise GitLabUrlParseError(f"{error_message}: {url}")

            # Decode the path to handle already URL-encoded paths
            decoded_path = unquote(path)
            match = re.search(pattern, decoded_path)

            if not match:
                raise GitLabUrlParseError(f"{error_message}: {url}")

            return list(match.groups())
        except Exception as e:
            if isinstance(e, GitLabUrlParseError):
                raise
            raise GitLabUrlParseError(f"{error_message}: {url}") from e

    @staticmethod
    def parse_project_url(url: str, gitlab_host: str) -> str:
        """Extract project path from a GitLab URL.

        Example URLs:
        - https://gitlab.com/namespace/project
        - https://gitlab.example.com/namespace/project

        Args:
            url: The GitLab URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            The URL-encoded project path

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        # Use a pattern that captures everything up to /-/ if present
        components = GitLabUrlParser._extract_path_components(
            url, PROJECT_URL_REGEX, "Could not extract project path from URL"
        )

        # URL-encode the project path for API calls
        return quote(components[0], safe="")

    @staticmethod
    def parse_issue_url(url: str, gitlab_host: str) -> Tuple[str, int]:
        """Extract project path and issue ID from a GitLab issue URL.

        Example URL:
        - https://gitlab.com/namespace/project/-/issues/42
        - https://gitlab.example.com/namespace/project/-/issues/42

        Args:
            url: The GitLab issue URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded project path and the issue ID

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        components = GitLabUrlParser._extract_path_components(
            url, ISSUE_URL_REGEX, "Could not parse issue URL"
        )

        # URL-encode the project path for API calls
        encoded_path = quote(components[0], safe="")
        issue_iid = int(components[1])

        return encoded_path, issue_iid

    @staticmethod
    def parse_group_url(url: str, gitlab_host: str) -> str:
        """Extract group path from a GitLab URL.

        Example URLs:
        - https://gitlab.com/groups/namespace/group
        - https://gitlab.example.com/groups/namespace/group

        Args:
            url: The GitLab URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            The URL-encoded group path

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        # Use a pattern that captures an optional 'groups/' prefix and everything up to /-/ if present
        components = GitLabUrlParser._extract_path_components(
            url,
            GROUP_URL_REGEX,
            "Could not extract group path from URL",
        )

        # URL-encode the group path for API calls
        return quote(components[0], safe="")

    @staticmethod
    def parse_epic_url(url: str, gitlab_host: str) -> Tuple[str, int]:
        """Extract group path and epic ID from a GitLab epic URL.

        Example URLs:
        - https://gitlab.com/groups/namespace/group/-/epics/42
        - https://gitlab.example.com/groups/namespace/group/-/epics/42

        Args:
            url: The GitLab URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded group path and the epic ID

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        # Use a pattern that captures an optional 'groups/' prefix, the group path, and the epic ID
        components = GitLabUrlParser._extract_path_components(
            url, EPIC_URL_REGEX, "Could not parse epic URL"
        )

        # URL-encode the group path for API calls
        encoded_path = quote(components[0], safe="")
        epic_iid = int(components[1])

        return encoded_path, epic_iid

    @staticmethod
    def parse_merge_request_url(url: str, gitlab_host: str) -> Tuple[str, int]:
        """Extract project path and merge request ID from a GitLab merge request URL.

        Example URLs:
        - https://gitlab.com/namespace/project/-/merge_requests/123
        - https://gitlab.example.com/namespace/project/-/merge_requests/123
        - https://gitlab.com/group/subgroup/project/-/merge_requests/123

        Args:
            url: The GitLab merge request URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded project path and the merge request ID

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        components = GitLabUrlParser._extract_path_components(
            url, MR_URL_REGEX, "Could not parse merge request URL"
        )

        # URL-encode the project path for API calls
        encoded_path = quote(components[0], safe="")
        merge_request_iid = int(components[1])

        return encoded_path, merge_request_iid

    @staticmethod
    def parse_job_url(url: str, gitlab_host: str) -> Tuple[str, int]:
        """Extract project path and job ID from a GitLab job URL.

        Example URLs:
        - https://gitlab.com/namespace/project/-/jobs/42
        - https://gitlab.example.com/namespace/project/-/jobs/42

        Args:
            url: The GitLab job URL to parse
            gitlab_host: Optional GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded project path and the job ID

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        components = GitLabUrlParser._extract_path_components(
            url, JOB_URL_REGEX, "Could not parse job URL"
        )

        # URL-encode the project path for API calls
        encoded_path = quote(components[0], safe="")
        job_id = int(components[1])

        return encoded_path, job_id

    @staticmethod
    def parse_repository_file_url(url: str, gitlab_host: str) -> Tuple[str, str, str]:
        """Parse GitLab URL into project path, ref, and file path components.

        Example URLs:
        - https://gitlab.com/namespace/project/-/blob/master/README.md
        - https://gitlab.com/group/subgroup/project/-/blob/main/src/file.py

        Args:
            url: The GitLab URL to parse
            gitlab_host: The GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded project path, the ref, and the file path

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        components = GitLabUrlParser._extract_path_components(
            url, REPOSITORY_FILE_URL_REGEX, "Could not parse repository file URL"
        )

        # URL-encode the project path for API calls
        project_path = quote(components[0], safe="")
        ref = components[1]
        file_path = components[2]

        return project_path, ref, file_path

    @staticmethod
    def parse_commit_url(url: str, gitlab_host: str) -> Tuple[str, str]:
        """Extract project path and commit SHA from a GitLab commit URL.

        Example URLs:
        - https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6
        - https://gitlab.example.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6

        Args:
            url: The GitLab commit URL to parse
            gitlab_host: The GitLab host to validate against

        Returns:
            A tuple containing the URL-encoded project path and the commit SHA

        Raises:
            GitLabUrlParseError: If the URL cannot be parsed or if the netloc doesn't match gitlab_host
        """
        GitLabUrlParser._validate_url_netloc(url, gitlab_host)

        components = GitLabUrlParser._extract_path_components(
            url, COMMIT_URL_REGEX, "Could not parse commit URL"
        )

        # URL-encode the project path for API calls
        encoded_path = quote(components[0], safe="")
        commit_sha = components[1]

        return encoded_path, commit_sha
