"""Tests for DiffExclusionPolicy class."""

import json
from unittest.mock import patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.policies.diff_exclusion_policy import DiffExclusionPolicy
from lib.feature_flags.context import FeatureFlag


@pytest.fixture(autouse=True)
def mock_feature_flag():
    """Mock feature flag to return True for USE_DUO_CONTEXT_EXCLUSION."""
    with (
        patch(
            "duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled"
        ) as mock_diff,
        patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled"
        ) as mock_file,
    ):
        mock_diff.return_value = True
        mock_file.return_value = True
        yield mock_diff


class TestDiffExclusionPolicy:
    """Test cases for DiffExclusionPolicy."""

    def test_filter_allowed_with_valid_diff_json(self):
        """Test filter_allowed with valid diff dictionaries."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["*.log", "/secrets/*"],
        )

        policy = DiffExclusionPolicy(project)

        # Create test diff dictionaries
        allowed_diff = {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old line\n+new line",
        }

        excluded_diff_log = {
            "old_path": "app.log",
            "new_path": "app.log",
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        }

        excluded_diff_secrets = {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        diffs = [allowed_diff, excluded_diff_log, excluded_diff_secrets]
        result, excluded = policy.filter_allowed_diffs(diffs)

        assert len(result) == 1
        assert len(excluded) == 2
        assert excluded == ["app.log", "secrets/api_key.txt"]
        assert result[0] == allowed_diff

    def test_filter_allowed_with_different_old_and_new_paths(self):
        """Test filter_allowed when old_path and new_path are different."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log", "/temp/**"],
        )

        policy = DiffExclusionPolicy(project)

        # Both paths allowed
        allowed_diff = {
            "old_path": "src/utils.py",
            "new_path": "src/helpers.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        # Old path excluded, new path allowed
        old_excluded_diff = {
            "old_path": "temp/cache.txt",
            "new_path": "src/cache.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        # Old path allowed, new path excluded
        new_excluded_diff = {
            "old_path": "src/logger.py",
            "new_path": "logs/app.log",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        # Both paths excluded
        both_excluded_diff = {
            "old_path": "temp/old.log",
            "new_path": "logs/new.log",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        diffs = [
            allowed_diff,
            old_excluded_diff,
            new_excluded_diff,
            both_excluded_diff,
        ]
        result, excluded = policy.filter_allowed_diffs(diffs)

        # Only the diff with both paths allowed should be included
        assert len(result) == 1
        assert len(excluded) == 4
        assert result[0] == allowed_diff
        assert excluded == [
            "temp/cache.txt",
            "logs/app.log",
            "temp/old.log",
            "logs/new.log",
        ]

    def test_filter_allowed_with_leading_slashes_in_paths(self):
        """Test filter_allowed properly handles leading slashes in paths."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["/secrets/**"],
        )

        policy = DiffExclusionPolicy(project)

        # Test with leading slashes in paths
        diff_with_leading_slash = {
            "old_path": "/secrets/api_key.txt",
            "new_path": "/secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        # Test without leading slashes
        diff_without_leading_slash = {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        # Test allowed path
        allowed_diff = {
            "old_path": "/src/main.py",
            "new_path": "/src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        diffs = [
            diff_with_leading_slash,
            diff_without_leading_slash,
            allowed_diff,
        ]
        result, excluded = policy.filter_allowed_diffs(diffs)

        # Only the allowed diff should be included
        assert len(result) == 1
        assert len(excluded) == 2
        assert result[0] == allowed_diff
        assert excluded == ["/secrets/api_key.txt", "secrets/api_key.txt"]

    def test_filter_allowed_with_no_exclusion_rules(self):
        """Test filter_allowed when no exclusion rules are configured."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=None,
        )

        policy = DiffExclusionPolicy(project)

        diff1 = {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        diff2 = {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        diffs = [diff1, diff2]
        result, excluded = policy.filter_allowed_diffs(diffs)

        # All diffs should be allowed when no exclusion rules
        assert len(result) == 2
        assert len(excluded) == 0
        assert result == diffs

    def test_filter_allowed_with_empty_list(self):
        """Test filter_allowed with empty diff list."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log"],
        )

        policy = DiffExclusionPolicy(project)
        result, excluded = policy.filter_allowed_diffs([])

        assert result == []
        assert excluded == []

    def test_filter_allowed_complex_exclusion_patterns(self):
        """Test filter_allowed with complex exclusion patterns."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=[
                "**/node_modules/**",
                "**/*.log",
                "/build/**",
                "**/test_*.py",
                "**/.env*",
            ],
        )

        policy = DiffExclusionPolicy(project)

        # Create various test diffs
        allowed_diff = {
            "old_path": "src/main.py",
            "new_path": "src/utils.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        node_modules_diff = {
            "old_path": "frontend/node_modules/react/index.js",
            "new_path": "frontend/node_modules/react/index.js",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        log_diff = {
            "old_path": "logs/app.log",
            "new_path": "logs/app.log",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        build_diff = {
            "old_path": "build/output.js",
            "new_path": "build/output.js",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        test_diff = {
            "old_path": "src/test_helper.py",
            "new_path": "src/test_helper.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        env_diff = {
            "old_path": ".env.local",
            "new_path": ".env.local",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        diffs = [
            allowed_diff,
            node_modules_diff,
            log_diff,
            build_diff,
            test_diff,
            env_diff,
        ]
        result, excluded = policy.filter_allowed_diffs(diffs)

        # Only the allowed diff should remain
        assert len(result) == 1
        assert len(excluded) == 5
        assert result[0] == allowed_diff
        assert excluded == [
            "frontend/node_modules/react/index.js",
            "logs/app.log",
            "build/output.js",
            "src/test_helper.py",
            ".env.local",
        ]


class TestDiffExclusionPolicyFeatureFlag:
    """Test cases for DiffExclusionPolicy feature flag behavior."""

    def test_filter_allowed_with_feature_flag_disabled(self, mock_feature_flag):
        """Test that filter_allowed returns all diffs when feature flag is disabled."""
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

        policy = DiffExclusionPolicy(project)

        allowed_diff = {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        excluded_diff = {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        diffs = [allowed_diff, excluded_diff]
        result, excluded = policy.filter_allowed_diffs(diffs)

        # All diffs should be returned when feature flag is disabled
        assert len(result) == 2
        assert len(excluded) == 0
        assert result == diffs

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)

    def test_filter_allowed_excludes_duplicate_files(self):
        """Test that duplicate excluded files are not added to the excluded_files list."""
        project = Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=["**/*.log"],
        )

        policy = DiffExclusionPolicy(project)

        # Create multiple diffs with the same excluded file
        excluded_diff1 = {
            "old_path": "app.log",
            "new_path": "app.log",
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        }

        excluded_diff2 = {
            "old_path": "app.log",
            "new_path": "debug.log",
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        }

        diffs = [excluded_diff1, excluded_diff2]
        filtered_diffs, excluded_files = policy.filter_allowed_diffs(diffs)

        assert len(filtered_diffs) == 0
        # Should only have unique excluded files
        assert len(excluded_files) == 2
        assert excluded_files == ["app.log", "debug.log"]

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
            exclusion_rules=["**/*.log", "/secrets/**"],
        )

        policy = DiffExclusionPolicy(project)

        allowed_diff = {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        }

        excluded_diff = {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        }

        diffs = [allowed_diff, excluded_diff]
        filtered_diffs, excluded_files = policy.filter_allowed_diffs(diffs)

        # Only allowed diffs should be returned when feature flag is enabled
        assert len(filtered_diffs) == 1
        assert filtered_diffs[0] == allowed_diff
        assert len(excluded_files) == 1
        assert excluded_files == ["secrets/api_key.txt"]

        # Verify the feature flag was checked
        mock_feature_flag.assert_called_with(FeatureFlag.USE_DUO_CONTEXT_EXCLUSION)
