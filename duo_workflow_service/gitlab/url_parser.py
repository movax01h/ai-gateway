import re
from typing import List, Tuple
from urllib.parse import quote, unquote, urlparse


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
        - https://gitlab.com/namespace/project/-/issues
        - https://gitlab.example.com/namespace/project
        - https://gitlab.com/group/subgroup/project
        - https://gitlab.com/group/subgroup/project/-/issues

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
            url, r"^(.+?)(?:/-/.*)?$", "Could not extract project path from URL"
        )

        # URL-encode the project path for API calls
        return quote(components[0], safe="")

    @staticmethod
    def parse_issue_url(url: str, gitlab_host: str) -> Tuple[str, int]:
        """Extract project path and issue ID from a GitLab issue URL.

        Example URL:
        - https://gitlab.com/namespace/project/-/issues/42
        - https://gitlab.example.com/namespace/project/-/issues/42
        - https://gitlab.com/group/subgroup/project/-/issues/42

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
            url, r"^(.+?)/-/issues/(\d+)", "Could not parse issue URL"
        )

        # URL-encode the project path for API calls
        encoded_path = quote(components[0], safe="")
        issue_iid = int(components[1])

        return encoded_path, issue_iid
