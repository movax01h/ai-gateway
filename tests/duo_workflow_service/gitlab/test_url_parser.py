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

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_merge_request_id",
        [
            # Test standard merge request URL
            (
                "https://gitlab.com/namespace/project/-/merge_requests/123",
                "gitlab.com",
                quote("namespace/project", safe=""),
                123,
            ),
            # Test merge request URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/project/-/merge_requests/123",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
                123,
            ),
            # Test merge request URL with encoded characters in project path
            (
                "https://gitlab.com/namespace/project-with-dashes/-/merge_requests/123",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
                123,
            ),
        ],
    )
    def test_parse_merge_request_url(
        self, url, gitlab_host, expected_project_path, expected_merge_request_id
    ):
        project_path, merge_request_id = GitLabUrlParser.parse_merge_request_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert merge_request_id == expected_merge_request_id

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no merge request ID)
            (
                "https://gitlab.com/namespace/project/-/merge_requests",
                "gitlab.com",
                "Could not parse merge request URL",
            ),
            # Test invalid URL (merge request ID not a number)
            (
                "https://gitlab.com/namespace/project/-/merge_requests/abc",
                "gitlab.com",
                "Could not parse merge request URL",
            ),
            # Test invalid URL (not a merge request URL)
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                "Could not parse merge request URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project/-/merge_requests/123",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_merge_request_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_merge_request_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_merge_request_id",
        [
            # Test merge request URL with subgroups
            (
                "https://gitlab.com/group/subgroup/project/-/merge_requests/123",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
                123,
            ),
            # Test merge request URL with multiple subgroups
            (
                "https://gitlab.com/group/subgroup1/subgroup2/project/-/merge_requests/123",
                "gitlab.com",
                quote("group/subgroup1/subgroup2/project", safe=""),
                123,
            ),
        ],
    )
    def test_parse_merge_request_url_with_subgroups(
        self, url, gitlab_host, expected_project_path, expected_merge_request_id
    ):
        project_path, merge_request_id = GitLabUrlParser.parse_merge_request_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert merge_request_id == expected_merge_request_id

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_merge_request_id",
        [
            # Test merge request URL with special characters
            (
                "https://gitlab.com/namespace/project with spaces/-/merge_requests/123",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
                123,
            ),
            # Test merge request URL with Unicode characters
            (
                "https://gitlab.com/namespace/проект/-/merge_requests/123",  # Cyrillic characters
                "gitlab.com",
                quote("namespace/проект", safe=""),
                123,
            ),
        ],
    )
    def test_parse_merge_request_url_with_complex_encoding(
        self, url, gitlab_host, expected_project_path, expected_merge_request_id
    ):
        project_path, merge_request_id = GitLabUrlParser.parse_merge_request_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert merge_request_id == expected_merge_request_id

    def test_extract_path_components_exception_handling(self):
        url = "https://gitlab.com/namespace/project"
        pattern = f"]invalid-pattern["
        error_message = "Test error message"

        with pytest.raises(GitLabUrlParseError, match=f"{error_message}: {url}"):
            GitLabUrlParser._extract_path_components(url, pattern, error_message)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_group_path",
        [
            # Test standard group URL
            (
                "https://gitlab.com/groups/namespace/group",
                "gitlab.com",
                quote("namespace/group", safe=""),
            ),
            # Test group URL with custom GitLab instance
            (
                "https://gitlab.example.com/groups/namespace/group",
                "gitlab.example.com",
                quote("namespace/group", safe=""),
            ),
            # Test group URL with encoded characters in path
            (
                "https://gitlab.com/groups/namespace/group-with-dashes",
                "gitlab.com",
                quote("namespace/group-with-dashes", safe=""),
            ),
            # Test group URL with epics path
            (
                "https://gitlab.com/groups/namespace/group/-/epics",
                "gitlab.com",
                quote("namespace/group", safe=""),
            ),
            # Test group URL with multiple subgroups
            (
                "https://gitlab.com/groups/parent/subgroup/child",
                "gitlab.com",
                quote("parent/subgroup/child", safe=""),
            ),
        ],
    )
    def test_parse_group_url(self, url, gitlab_host, expected_group_path):
        group_path = GitLabUrlParser.parse_group_url(url, gitlab_host)
        assert group_path == expected_group_path

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (not a group URL)
            (
                "https://gitlab.com",
                "gitlab.com",
                "Could not extract group path from URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/group",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_group_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_group_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_group_path, expected_epic_id",
        [
            # Test standard epic URL
            (
                "https://gitlab.com/groups/namespace/group/-/epics/123",
                "gitlab.com",
                quote("namespace/group", safe=""),
                123,
            ),
            # Test epic URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/group/-/epics/123",
                "gitlab.example.com",
                quote("namespace/group", safe=""),
                123,
            ),
            # Test epic URL with encoded characters in path
            (
                "https://gitlab.com/namespace/group-with-dashes/-/epics/123",
                "gitlab.com",
                quote("namespace/group-with-dashes", safe=""),
                123,
            ),
            # Test epic URL with multiple subgroups
            (
                "https://gitlab.com/parent/subgroup/group/-/epics/123",
                "gitlab.com",
                quote("parent/subgroup/group", safe=""),
                123,
            ),
        ],
    )
    def test_parse_epic_url(
        self, url, gitlab_host, expected_group_path, expected_epic_id
    ):
        group_path, epic_id = GitLabUrlParser.parse_epic_url(url, gitlab_host)
        assert group_path == expected_group_path
        assert epic_id == expected_epic_id

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no epic ID)
            (
                "https://gitlab.com/namespace/group/-/epics",
                "gitlab.com",
                "Could not parse epic URL",
            ),
            # Test invalid URL (epic ID not a number)
            (
                "https://gitlab.com/namespace/group/-/epics/abc",
                "gitlab.com",
                "Could not parse epic URL",
            ),
            # Test invalid URL (not an epic URL)
            (
                "https://gitlab.com/namespace/group",
                "gitlab.com",
                "Could not parse epic URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/group/-/epics/123",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_epic_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_epic_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_job_id",
        [
            # Test standard job URL
            (
                "https://gitlab.com/namespace/project/-/jobs/123",
                "gitlab.com",
                quote("namespace/project", safe=""),
                123,
            ),
            # Test job URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/project/-/jobs/123",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
                123,
            ),
            # Test job URL with encoded characters in path
            (
                "https://gitlab.com/namespace/project-with-dashes/-/jobs/123",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
                123,
            ),
            # Test job URL with subgroups
            (
                "https://gitlab.com/group/subgroup/project/-/jobs/123",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
                123,
            ),
        ],
    )
    def test_parse_job_url(
        self, url, gitlab_host, expected_project_path, expected_job_id
    ):
        project_path, job_id = GitLabUrlParser.parse_job_url(url, gitlab_host)
        assert project_path == expected_project_path
        assert job_id == expected_job_id

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no job ID)
            (
                "https://gitlab.com/namespace/project/-/jobs",
                "gitlab.com",
                "Could not parse job URL",
            ),
            # Test invalid URL (job ID not a number)
            (
                "https://gitlab.com/namespace/project/-/jobs/abc",
                "gitlab.com",
                "Could not parse job URL",
            ),
            # Test invalid URL (not a job URL)
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                "Could not parse job URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project/-/jobs/123",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_job_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_job_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_ref, expected_file_path",
        [
            (
                "https://gitlab.com/namespace/project/-/blob/master/README.md",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "master",
                "README.md",
            ),
            (
                "https://gitlab.example.com/namespace/project/-/blob/main/README.md",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
                "main",
                "README.md",
            ),
            (
                "https://gitlab.com/namespace/project-with-dashes/-/blob/master/README.md",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
                "master",
                "README.md",
            ),
            (
                "https://gitlab.com/namespace/project/-/blob/master/src/main.py",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "master",
                "src/main.py",
            ),
            (
                "https://gitlab.com/namespace/project/-/blob/v1.0.0/README.md",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "v1.0.0",
                "README.md",
            ),
            (
                "https://gitlab.com/namespace/project/-/blob/abc123def456/README.md",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "abc123def456",
                "README.md",
            ),
        ],
        ids=[
            "standard_repository_url",
            "custom_gitlab_instance",
            "project_with_dashes",
            "file_in_subdirectory",
            "tag_as_ref",
            "commit_hash_as_ref",
        ],
    )
    def test_parse_repository_file_url(
        self, url, gitlab_host, expected_project_path, expected_ref, expected_file_path
    ):
        project_path, ref, file_path = GitLabUrlParser.parse_repository_file_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert ref == expected_ref
        assert file_path == expected_file_path

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                "Could not parse repository file URL",
            ),
            (
                "https://gitlab.com/namespace/project/-/blob/master",
                "gitlab.com",
                "Could not parse repository file URL",
            ),
            (
                "https://gitlab.com/namespace/project/-/tree/master/src",
                "gitlab.com",
                "Could not parse repository file URL",
            ),
            (
                "https://gitlab.example.com/namespace/project/-/blob/master/README.md",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
        ids=[
            "no_blob_path",
            "no_file_path",
            "not_blob_url",
            "non_matching_netloc",
        ],
    )
    def test_parse_repository_file_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_repository_file_url(url, gitlab_host)

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_ref, expected_file_path",
        [
            (
                "https://gitlab.com/group/subgroup/project/-/blob/master/README.md",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
                "master",
                "README.md",
            ),
            (
                "https://gitlab.com/group/subgroup1/subgroup2/project/-/blob/master/README.md",
                "gitlab.com",
                quote("group/subgroup1/subgroup2/project", safe=""),
                "master",
                "README.md",
            ),
        ],
        ids=["with_subgroups", "with_multiple_subgroups"],
    )
    def test_parse_repository_file_url_with_subgroups(
        self, url, gitlab_host, expected_project_path, expected_ref, expected_file_path
    ):
        project_path, ref, file_path = GitLabUrlParser.parse_repository_file_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert ref == expected_ref
        assert file_path == expected_file_path

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_ref, expected_file_path",
        [
            (
                "https://gitlab.com/namespace/project/-/blob/master/file with spaces.txt",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "master",
                "file with spaces.txt",
            ),
            (
                "https://gitlab.com/namespace/project with spaces/-/blob/master/README.md",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
                "master",
                "README.md",
            ),
            (
                "https://gitlab.com/namespace/project/-/blob/master/документ.md",  # Cyrillic characters
                "gitlab.com",
                quote("namespace/project", safe=""),
                "master",
                "документ.md",
            ),
            (
                "https://gitlab.com/namespace/проект/-/blob/master/README.md",  # Cyrillic characters
                "gitlab.com",
                quote("namespace/проект", safe=""),
                "master",
                "README.md",
            ),
            (
                "https://gitlab.com/namespace/project%20with%20spaces/-/blob/master/file%20with%20spaces.txt",
                "gitlab.com",
                quote("namespace/project with spaces", safe=""),
                "master",
                "file with spaces.txt",
            ),
        ],
        ids=[
            "special_characters_in_path",
            "special_characters_in_project",
            "unicode_characters_in_path",
            "unicode_characters_in_project",
            "already_url_encoded_characters",
        ],
    )
    def test_parse_repository_file_url_with_complex_encoding(
        self, url, gitlab_host, expected_project_path, expected_ref, expected_file_path
    ):
        project_path, ref, file_path = GitLabUrlParser.parse_repository_file_url(
            url, gitlab_host
        )
        assert project_path == expected_project_path
        assert ref == expected_ref
        assert file_path == expected_file_path

    @pytest.mark.parametrize(
        "url, gitlab_host, expected_project_path, expected_commit_sha",
        [
            # Test standard commit URL
            (
                "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            ),
            # Test commit URL with custom GitLab instance
            (
                "https://gitlab.example.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "gitlab.example.com",
                quote("namespace/project", safe=""),
                "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            ),
            # Test commit URL with subgroups
            (
                "https://gitlab.com/group/subgroup/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "gitlab.com",
                quote("group/subgroup/project", safe=""),
                "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            ),
            # Test commit URL with encoded characters in project path
            (
                "https://gitlab.com/namespace/project-with-dashes/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "gitlab.com",
                quote("namespace/project-with-dashes", safe=""),
                "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            ),
            # Test commit URL with short SHA
            (
                "https://gitlab.com/namespace/project/-/commit/c34bb66f",
                "gitlab.com",
                quote("namespace/project", safe=""),
                "c34bb66f",
            ),
        ],
    )
    def test_parse_commit_url(
        self, url, gitlab_host, expected_project_path, expected_commit_sha
    ):
        project_path, commit_sha = GitLabUrlParser.parse_commit_url(url, gitlab_host)
        assert project_path == expected_project_path
        assert commit_sha == expected_commit_sha

    @pytest.mark.parametrize(
        "url, gitlab_host, error_message",
        [
            # Test invalid URL (no commit SHA)
            (
                "https://gitlab.com/namespace/project/-/commit",
                "gitlab.com",
                "Could not parse commit URL",
            ),
            # Test invalid URL (not a commit URL)
            (
                "https://gitlab.com/namespace/project",
                "gitlab.com",
                "Could not parse commit URL",
            ),
            # Test URL with invalid SHA format
            (
                "https://gitlab.com/namespace/project/-/commit/invalid-sha",
                "gitlab.com",
                "Could not parse commit URL",
            ),
            # Test URL with non-matching netloc
            (
                "https://gitlab.example.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "gitlab.com",
                "URL netloc 'gitlab.example.com' does not match gitlab_host 'gitlab.com'",
            ),
        ],
    )
    def test_parse_commit_url_error(self, url, gitlab_host, error_message):
        with pytest.raises(GitLabUrlParseError, match=error_message):
            GitLabUrlParser.parse_commit_url(url, gitlab_host)
