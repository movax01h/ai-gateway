"""Tests for FileExclusionPolicy class."""

from pathlib import Path
from unittest.mock import patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from lib.feature_flags.context import FeatureFlag


@pytest.fixture(autouse=True)
def mock_feature_flag():
    """Mock feature flag to return True for USE_DUO_CONTEXT_EXCLUSION."""
    with patch(
        "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled"
    ) as mock:
        mock.return_value = True
        yield mock


class TestFileExclusionPolicy:
    """Test cases for FileExclusionPolicy."""

    def test_init_with_no_exclusion_rules(self):
        """Test initialization with project that has no exclusion rules."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=None,
        )

        policy = FileExclusionPolicy(project)
        assert policy._exclusion_rules == []

    def test_path_based_exclusions(self):
        """Test path-based exclusion rules."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["/secrets/**", "logs/", "*.log"],
        )

        policy = FileExclusionPolicy(project)

        # Test file exclusions
        assert not policy.is_allowed("secrets/api_key.txt")
        assert not policy.is_allowed("secrets/db/password.txt")
        assert not policy.is_allowed("logs/app.log")
        assert not policy.is_allowed("logs/error.log")
        assert not policy.is_allowed("app.log")
        assert not policy.is_allowed("debug.log")

        # Test files that should not be excluded
        assert policy.is_allowed("src/main.py")
        assert policy.is_allowed("config.json")
        assert policy.is_allowed("secret.txt")  # Not in secrets/ directory

    def test_extension_based_exclusions(self):
        """Test extension-based exclusion rules."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.env", "**/*.key"],
        )

        policy = FileExclusionPolicy(project)

        # Test file exclusions
        assert not policy.is_allowed(".env")
        assert not policy.is_allowed("config/.env")
        assert not policy.is_allowed("src/config/.env")
        assert not policy.is_allowed("private.key")
        assert not policy.is_allowed("ssl/server.key")

        # Test files that should not be excluded
        assert policy.is_allowed("environment.txt")
        assert policy.is_allowed("keychain.py")
        assert policy.is_allowed("src/main.py")

    def test_mixed_exclusion_types(self):
        """Test project with mixed exclusion rule types."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["/secrets/**", "**/*.env", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)

        # Test various file exclusions
        assert not policy.is_allowed("secrets/api_key.txt")  # Path rule
        assert not policy.is_allowed("config/.env")  # Extension rule
        assert not policy.is_allowed("temp.tmp")  # Path rule with glob

        # Test files that should not be excluded
        assert policy.is_allowed("src/main.py")
        assert policy.is_allowed("config.json")

    def test_complex_path_patterns(self):
        """Test complex path patterns with wildcards."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=[
                "**/node_modules/**",
                "src/**/test_*.py",
                "**/.*",  # Hidden files
            ],
        )

        policy = FileExclusionPolicy(project)

        # Test node_modules exclusion
        assert not policy.is_allowed("node_modules/package/index.js")
        assert not policy.is_allowed("frontend/node_modules/react/index.js")
        assert not policy.is_allowed("deep/path/node_modules/lib/file.js")

        # Test test file exclusion
        assert not policy.is_allowed("src/test_main.py")
        assert not policy.is_allowed("src/utils/test_helper.py")

        # Test hidden files
        assert not policy.is_allowed(".gitignore")
        assert not policy.is_allowed("config/.env")
        assert not policy.is_allowed("src/.hidden")

        # Test files that should not be excluded
        assert policy.is_allowed("src/main.py")
        assert policy.is_allowed("package.json")
        assert policy.is_allowed("modules/helper.py")

    def test_path_normalization(self):
        """Test path normalization for different path formats."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["/temp/**"],
        )

        policy = FileExclusionPolicy(project)

        # Test different path formats
        assert not policy.is_allowed("temp/file.txt")
        assert not policy.is_allowed("/temp/file.txt")
        assert not policy.is_allowed("temp\\file.txt")  # Windows-style path

        # Test that normalization works correctly
        assert policy.is_allowed("temporary/file.txt")

    def test_directory_patterns(self):
        """Test directory-specific patterns."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["build/", "/dist/"],
        )

        policy = FileExclusionPolicy(project)

        # Test directory exclusions
        assert not policy.is_allowed("build/output.js")
        assert not policy.is_allowed("build/assets/style.css")
        assert not policy.is_allowed("dist/bundle.js")
        assert not policy.is_allowed("dist/assets/app.js")

        # Test files that should not be excluded
        assert policy.is_allowed("src/build.py")
        assert policy.is_allowed("distribute.py")

    def test_edge_cases(self):
        """Test edge cases and special patterns."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**", "**/test_*"],  # Match everything
        )

        policy = FileExclusionPolicy(project)

        # The ** pattern should match everything
        assert not policy.is_allowed("any/file.txt")
        assert not policy.is_allowed("test_file.py")
        assert not policy.is_allowed("src/test_helper.py")

    def test_case_sensitivity(self):
        """Test case sensitivity in pattern matching."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.LOG"],
        )

        policy = FileExclusionPolicy(project)

        # Test case sensitivity (should be case-sensitive by default)
        assert not policy.is_allowed("app.LOG")

    def test_filter_allowed_with_no_exclusions(self):
        """Test filter_allowed when no exclusion rules are set."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=None,
        )

        policy = FileExclusionPolicy(project)
        filenames = ["src/main.py", "config.json", "README.md", "test.log"]

        result = policy.filter_allowed(filenames)
        assert result == filenames

    def test_filter_allowed_with_exclusions(self):
        """Test filter_allowed with various exclusion patterns."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)
        filenames = [
            "src/main.py",
            "app.log",
            "secrets/api_key.txt",
            "config.json",
            "temp.tmp",
            "README.md",
            "logs/debug.log",
        ]

        result = policy.filter_allowed(filenames)
        expected = ["src/main.py", "config.json", "README.md"]
        assert result == expected

    def test_filter_allowed_with_empty_list(self):
        """Test filter_allowed with empty filename list."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log"],
        )

        policy = FileExclusionPolicy(project)
        result = policy.filter_allowed([])
        assert result == []

    def test_filter_allowed_with_whitespace_filenames(self):
        """Test filter_allowed handles filenames with whitespace correctly."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log"],
        )

        policy = FileExclusionPolicy(project)
        filenames = [
            "  src/main.py  ",
            " app.log ",
            "",
            "   ",
            "config.json",
        ]

        result = policy.filter_allowed(filenames)
        expected = ["src/main.py", "config.json"]
        assert result == expected

    def test_filter_allowed_all_files_excluded(self):
        """Test filter_allowed when all files are excluded."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**"],
        )

        policy = FileExclusionPolicy(project)
        filenames = ["src/main.py", "config.json", "README.md"]

        result = policy.filter_allowed(filenames)
        assert result == []

    def test_is_allowed_for_project_static_method(self):
        """Test the static is_allowed_for_project method."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**"],
        )

        # Test allowed files
        result = FileExclusionPolicy.is_allowed_for_project(project, "src/main.py")
        assert result is True

        result = FileExclusionPolicy.is_allowed_for_project(project, "config.json")
        assert result is True

        # Test excluded files
        result = FileExclusionPolicy.is_allowed_for_project(project, "app.log")
        assert result is False

        result = FileExclusionPolicy.is_allowed_for_project(
            project, "secrets/api_key.txt"
        )
        assert result is False

    def test_is_allowed_for_project_with_none_project(self):
        """Test is_allowed_for_project with None project."""
        result = FileExclusionPolicy.is_allowed_for_project(None, "src/main.py")
        assert result is True  # Should allow all files when no project/exclusion rules

    def test_format_user_exclusion_message(self):
        """Test format_user_exclusion_message static method."""
        blocked_files = ["secrets/api_key.txt", "app.log", "temp.tmp"]

        result = FileExclusionPolicy.format_user_exclusion_message(blocked_files)
        expected = " - files excluded:\n" "secrets/api_key.txt\n" "app.log\n" "temp.tmp"
        assert result == expected

    def test_format_user_exclusion_message_single_file(self):
        """Test format_user_exclusion_message with single file."""
        blocked_files = ["secrets/api_key.txt"]

        result = FileExclusionPolicy.format_user_exclusion_message(blocked_files)
        expected = " - files excluded:\n" "secrets/api_key.txt"
        assert result == expected

    def test_format_user_exclusion_message_empty_list(self):
        """Test format_user_exclusion_message with empty list."""
        blocked_files = []

        result = FileExclusionPolicy.format_user_exclusion_message(blocked_files)
        expected = " - files excluded:\n"
        assert result == expected

    def test_format_llm_exclusion_message(self):
        """Test format_llm_exclusion_message static method."""
        blocked_files = ["secrets/api_key.txt", "app.log", "temp.tmp"]

        result = FileExclusionPolicy.format_llm_exclusion_message(blocked_files)
        expected = (
            "Files excluded due to policy, continue without files:\n"
            "secrets/api_key.txt\n"
            "app.log\n"
            "temp.tmp"
        )
        assert result == expected

    def test_format_llm_exclusion_message_single_file(self):
        """Test format_llm_exclusion_message with single file."""
        blocked_files = ["secrets/api_key.txt"]

        result = FileExclusionPolicy.format_llm_exclusion_message(blocked_files)
        expected = (
            "Files excluded due to policy, continue without files:\n"
            "secrets/api_key.txt"
        )
        assert result == expected

    def test_format_llm_exclusion_message_empty_list(self):
        """Test format_llm_exclusion_message with empty list."""
        blocked_files = []

        result = FileExclusionPolicy.format_llm_exclusion_message(blocked_files)
        expected = "Files excluded due to policy, continue without files:\n"
        assert result == expected


class TestFileExclusionPolicyFeatureFlag:
    """Test cases for FileExclusionPolicy feature flag behavior."""

    def test_is_allowed_with_feature_flag_disabled(self, mock_feature_flag):
        """Test that is_allowed always returns True when feature flag is disabled."""
        mock_feature_flag.return_value = False

        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)

        # All files should be allowed when feature flag is disabled
        assert policy.is_allowed("app.log") is True
        assert policy.is_allowed("secrets/api_key.txt") is True
        assert policy.is_allowed("temp.tmp") is True
        assert policy.is_allowed("src/main.py") is True

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)

    def test_is_allowed_with_feature_flag_enabled(self, mock_feature_flag):
        """Test that is_allowed respects exclusion rules when feature flag is enabled."""
        mock_feature_flag.return_value = True

        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)

        # Files matching exclusion rules should be blocked
        assert policy.is_allowed("app.log") is False
        assert policy.is_allowed("secrets/api_key.txt") is False
        assert policy.is_allowed("temp.tmp") is False

        # Files not matching exclusion rules should be allowed
        assert policy.is_allowed("src/main.py") is True

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)

    def test_filter_allowed_with_feature_flag_disabled(self, mock_feature_flag):
        """Test that filter_allowed returns all files when feature flag is disabled."""
        mock_feature_flag.return_value = False

        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)
        filenames = [
            "src/main.py",
            "app.log",
            "secrets/api_key.txt",
            "config.json",
            "temp.tmp",
        ]

        result = policy.filter_allowed(filenames)

        # All files should be returned when feature flag is disabled
        assert result == filenames

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)

    def test_filter_allowed_with_feature_flag_enabled(self, mock_feature_flag):
        """Test that filter_allowed respects exclusion rules when feature flag is enabled."""
        mock_feature_flag.return_value = True

        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**", "*.tmp"],
        )

        policy = FileExclusionPolicy(project)
        filenames = [
            "src/main.py",
            "app.log",
            "secrets/api_key.txt",
            "config.json",
            "temp.tmp",
        ]

        result = policy.filter_allowed(filenames)

        # Only files not matching exclusion rules should be returned
        expected = ["src/main.py", "config.json"]
        assert result == expected

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)

    def test_is_allowed_for_project_with_feature_flag_disabled(self, mock_feature_flag):
        """Test static method is_allowed_for_project with feature flag disabled."""
        mock_feature_flag.return_value = False

        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/secrets/**"],
        )

        # All files should be allowed when feature flag is disabled
        assert FileExclusionPolicy.is_allowed_for_project(project, "app.log") is True
        assert (
            FileExclusionPolicy.is_allowed_for_project(project, "secrets/api_key.txt")
            is True
        )
        assert (
            FileExclusionPolicy.is_allowed_for_project(project, "src/main.py") is True
        )

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)
