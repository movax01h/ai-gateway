from unittest.mock import patch
from urllib.parse import quote

import pytest

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser


class TestGitLabUrlParser:
    @pytest.mark.parametrize(
        "url, gitlab_host, should_raise, error_message",
        [
            # Test with matching netloc
            ("https://gitlab.com/namespace/project", "gitlab.com", False, None),
            # Test with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project",
                "gitlab.com",
                True,
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_validate_url_netloc(self, url, gitlab_host, should_raise, error_message):
        if should_raise:
            with pytest.raises(GitLabUrlParseError, match=error_message):
                GitLabUrlParser._validate_url_netloc(url, gitlab_host)
        else:
            # Should not raise an exception
            GitLabUrlParser._validate_url_netloc(url, gitlab_host)

    def test_validate_url_netloc_urlparse_exception_handling(self):
        url = "https://gitlab.com/namespace/project"
        gitlab_host = "gitlab.com"
        error_message = "Could not validate URL netloc"

        # Patch urlparse to raise an exception
        with patch("duo_workflow_service.gitlab.url_parser.urlparse") as mock_urlparse:
            mock_urlparse.side_effect = ValueError("Test exception")

            with pytest.raises(GitLabUrlParseError, match=f"{error_message}: {url}"):
                GitLabUrlParser._validate_url_netloc(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_result",
        [
            # Test standard project URL
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                quote("namespace/project", safe=""),
            ),
            # Test project URL with trailing slash
            (
                "https://gitlab.com/namespace/project/",
                "gitlab.com",
                quote("namespace/project", safe=""),
            ),
            # Test project URL with issues path
            (
                "https://gitlab.com/namespace/project/-/issues",
                "gitlab.com",
                quote("namespace/project", safe=""),
            ),
            # Test project URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/project",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
            ),
            # Test project URL with encoded characters
            (
                "https://gitlab.com/namespace/project-with-dashes",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
            ),
        ],
    )
    def test_parse_project_url(self, url, gitlab_host, expected_result):
        assert GitLabUrlParser.parse_project_url(url, gitlab_host) == expected_result

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no project path)
            (
                "https://gitlab.com/",
                "gitlab.com",
                "Could not extract project path from URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_project_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_project_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_issue_id",
        [
            # Test standard issue URL
            (
                "https://gitlab.com/namespace/project/-/issues/42",
                "gitlab.com",
                quote("namespace/project", safe=""),
                42,
            ),
            # Test issue URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/project/-/issues/42",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
                42,
            ),
            # Test issue URL with encoded characters in project path
            (
                "https://gitlab.com/namespace/project-with-dashes/-/issues/42",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
                42,
            ),
        ],
    )
    def test_parse_issue_url(
        self, url, gitlab_host, expected_project_path, expected_issue_id
    ):
        project_path, issue_id = GitLabUrlParser.parse_issue_url(url, gitlab_host)
        assert project_path == expected_project_path
        assert issue_id == expected_issue_id

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no issue ID)
            (
                "https://gitlab.com/namespace/project/-/issues",
                "gitlab.com",
                "Could not parse issue URL",
            ),
            # Test invalid URL (issue ID not a number)
            (
                "https://gitlab.com/namespace/project/-/issues/abc",
                "gitlab.com",
                "Could not parse issue URL",
            ),
            # Test invalid URL (not an issue URL)
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                "Could not parse issue URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project/-/issues/42",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_issue_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_issue_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, expected_host",
        [
            ("https://gitlab.com/namespace/project.git", "gitlab.com"),
            ("https://gitlab.example.com/namespace/project.git", "gitlab.example.com"),
            ("https://gitlab.com/namespace/project", "gitlab.com"),
            (
                "https://gitlab.example.com:8080/namespace/project",
                "gitlab.example.com:8080",
            ),
        ],
    )
    def test_extract_host_from_url(self, url, expected_host):
        assert GitLabUrlParser.extract_host_from_url(url) == expected_host

    @pytest.mark.parametrize(
        "url",
        [
            "invalid-url",
            "http://",
        ],
    )
    def test_extract_host_from_url_error(self, url):
        with pytest.raises(GitLabUrlParseError):
            GitLabUrlParser.extract_host_from_url(url)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_result",
        [
            # Test project URL with subgroups
            (
                "https://gitlab.com/group/subgroup/project",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
            ),
            # Test project URL with multiple subgroups
            (
                "https://gitlab.com/group/subgroup1/subgroup2/project",
                "gitlab.com",
                quote("group/subgroup1/subgroup2/project", safe=""),
            ),
        ],
    )
    def test_parse_project_url_with_subgroups(self, url, gitlab_host, expected_result):
        assert GitLabUrlParser.parse_project_url(url, gitlab_host) == expected_result

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_issue_id",
        [
            # Test issue URL with subgroups
            (
                "https://gitlab.com/group/subgroup/project/-/issues/42",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
                42,
            ),
            # Test issue URL with multiple subgroups
            (
                "https://gitlab.com/group/subgroup1/subgroup2/project/-/issues/42",
                "gitlab.com",
                quote("group/subgroup1/subgroup2/project", safe=""),
                42,
            ),
        ],
    )
    def test_parse_issue_url_with_subgroups(
        self, url, gitlab_host, expected_project_path, expected_issue_id
    ):
        project_path, issue_id = GitLabUrlParser.parse_issue_url(url, gitlab_host)
        assert project_path == expected_project_path
        assert issue_id == expected_issue_id

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_result",
        [
            # Test project URL with special characters
            (
                "https://gitlab.com/namespace/project with spaces",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
            ),
            # Test project URL with Unicode characters
            (
                "https://gitlab.com/namespace/проект",  # Cyrillic characters
                "gitlab.com",
                quote("namespace/проект", safe=""),
            ),
            # Test project URL with already URL-encoded characters
            (
                "https://gitlab.com/namespace/project%20with%20spaces",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
            ),
        ],
    )
    def test_parse_project_url_with_complex_encoding(
        self, url, gitlab_host, expected_result
    ):
        """Test parsing project URLs with complex encoding scenarios."""
        assert GitLabUrlParser.parse_project_url(url, gitlab_host) == expected_result

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_issue_id",
        [
            # Test issue URL with special characters
            (
                "https://gitlab.com/namespace/project with spaces/-/issues/42",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
                42,
            ),
            # Test issue URL with Unicode characters
            (
                "https://gitlab.com/namespace/проект/-/issues/42",  # Cyrillic characters
                "gitlab.com",
                quote("namespace/проект", safe=""),
                42,
            ),
        ],
    )
    def test_parse_issue_url_with_complex_encoding(
        self, url, gitlab_host, expected_project_path, expected_issue_id
    ):
        project_path, issue_id = GitLabUrlParser.parse_issue_url(url, gitlab_host)
        assert project_path == expected_project_path
        assert issue_id == expected_issue_id

    def test_extract_path_components_exception_handling(self):
        url = "https://gitlab.com/namespace/project"
        pattern = f"]invalid-pattern["
        error_message = "Test error message"

        with pytest.raises(GitLabUrlParseError, match=f"{error_message}: {url}"):
            GitLabUrlParser._extract_path_components(url, pattern, error_message)
