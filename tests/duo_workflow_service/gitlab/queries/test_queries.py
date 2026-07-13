# pylint: disable=file-naming-for-tests
from unittest.mock import patch

import pytest

from duo_workflow_service.gitlab.queries import fetch_query_for_version


class TestQueryVersionSelection:
    """Test fetch_query_for_version returns correct query per GitLab version."""

    def test_query_selection_gitlab_below_18_3(self):
        """GitLab < 18.3 returns the 18.2 query (no namespace, no aiSettings)."""
        with patch("duo_workflow_service.gitlab.queries.log_exception") as _mock_log:
            result = fetch_query_for_version("18.2.0")

        assert "aiSettings" not in result
        assert "namespaceId" not in result
        assert "firstCheckpoint" in result
        assert "latestCheckpoint" not in result

    def test_query_selection_gitlab_18_3_or_above(self):
        """GitLab >= 18.3 returns a query with namespace but no aiSettings."""
        result = fetch_query_for_version("18.7.0")

        assert "aiSettings" not in result
        assert "namespaceId" in result
        assert "firstCheckpoint" in result
        assert "latestCheckpoint" not in result

    def test_query_selection_gitlab_18_8_or_above(self):
        """GitLab >= 18.8 returns a query with aiSettings."""
        result = fetch_query_for_version("18.8.0")

        assert "aiSettings" in result
        assert "latestCheckpoint" in result
        assert "compressedCheckpoint" not in result
        assert "incrementalCheckpointsEnabled" not in result

    def test_query_selection_gitlab_18_9_or_above(self):
        """GitLab >= 18.9 returns a query with aiSettings (per-session tool approvals)."""
        result = fetch_query_for_version("18.9.0")

        assert "aiSettings" in result
        assert "latestCheckpoint" in result
        assert "compressedCheckpoint" not in result
        assert "incrementalCheckpointsEnabled" not in result

    def test_query_selection_gitlab_19_0_or_above(self):
        """GitLab >= 19.0 returns a query with compressedCheckpoint."""
        result = fetch_query_for_version("19.0.0")

        assert "compressedCheckpoint" in result
        assert "incrementalCheckpointsEnabled" not in result

    def test_query_selection_gitlab_19_2_or_above(self):
        """GitLab >= 19.2 returns a query with incrementalCheckpointsEnabled."""
        result = fetch_query_for_version("19.2.0")

        assert "incrementalCheckpointsEnabled" in result
        assert "compressedCheckpoint" in result
        assert "duoWorkflowStatusCheck" in result

    def test_query_selection_unknown_version_returns_fallback(self):
        """Unknown/unparsable version returns the fallback (oldest) query."""
        result = fetch_query_for_version(None)

        # Fallback is the 18.2 query — no aiSettings, no namespace
        assert "aiSettings" not in result
        assert "namespaceId" not in result

    def test_query_selection_invalid_version_string_returns_fallback(self):
        """Invalid version string returns the fallback (oldest) query."""
        result = fetch_query_for_version("not-a-version")

        assert "aiSettings" not in result
        assert "namespaceId" not in result

    def test_query_selection_very_old_version_returns_fallback(self):
        """Version older than 18.2 returns the fallback (oldest) query."""
        result = fetch_query_for_version("17.0.0")

        assert "aiSettings" not in result
        assert "namespaceId" not in result

    def test_query_selection_future_version_returns_latest(self):
        """A future version returns the newest available query."""
        result = fetch_query_for_version("99.0.0")

        assert "incrementalCheckpointsEnabled" in result
        assert "compressedCheckpoint" in result

    @pytest.mark.parametrize(
        "version, expected_feature",
        [
            pytest.param("18.2.0", "firstCheckpoint", id="18.2_has_firstCheckpoint"),
            pytest.param("18.3.0", "archived", id="18.3_has_archived"),
            pytest.param("18.8.0", "aiSettings", id="18.8_has_aiSettings"),
            pytest.param(
                "19.0.0",
                "compressedCheckpoint",
                id="19.0_has_compressedCheckpoint",
            ),
            pytest.param(
                "19.2.0",
                "incrementalCheckpointsEnabled",
                id="19.2_has_incrementalCheckpointsEnabled",
            ),
        ],
    )
    def test_version_gate_features(self, version: str, expected_feature: str):
        """Each version gate introduces a specific feature in the query."""
        result = fetch_query_for_version(version)

        assert expected_feature in result
