"""Tests for the stable-branch patch tagging script."""

import io
import urllib.error
from unittest.mock import patch

import pytest

from scripts.tag_stable_patch import (
    _request,
    head_already_tagged,
    next_tag,
    patch_numbers,
    version_line,
)


def _tag(name, sha):
    return {"name": name, "commit": {"id": sha}}


class TestVersionLine:
    """Derive the version line from a branch name."""

    @pytest.mark.parametrize(
        "branch,expected",
        [
            ("stable-18-2-ee", "18.2"),
            ("stable-18-20-ee", "18.20"),
            ("stable-1-0-ee", "1.0"),
        ],
    )
    def test_stable_branches(self, branch, expected):
        assert version_line(branch) == expected

    @pytest.mark.parametrize(
        "branch",
        ["main", "stable-18-2", "stable-18-2-ee-foo", "18-2-ee", "", None],
    )
    def test_non_stable_branches_return_none(self, branch):
        assert version_line(branch) is None


class TestPatchNumbers:
    """Extract patch integers, ignoring other lines and decorations."""

    def test_filters_to_the_matching_line(self):
        names = [
            "self-hosted-v18.2.0-ee",
            "self-hosted-v18.2.10-ee",
            "self-hosted-v18.20.0-ee",  # different line
            "self-hosted-v1.2.0-ee",  # different line
            "self-hosted-v18.2.0-ee-rc1",  # not a plain patch tag
            "v18.2.5-ee",  # missing prefix
        ]
        assert sorted(patch_numbers(names, "self-hosted-v18.2.")) == [0, 10]


class TestNextTag:
    """Compute the next patch tag for a version line."""

    def test_bumps_highest_patch(self):
        names = ["self-hosted-v18.2.0-ee", "self-hosted-v18.2.10-ee"]
        assert next_tag(names, "self-hosted-v18.2.") == "self-hosted-v18.2.11-ee"

    def test_creates_baseline_when_no_tag_exists(self):
        assert next_tag([], "self-hosted-v18.2.") == "self-hosted-v18.2.0-ee"

    def test_creates_baseline_when_only_other_lines_exist(self):
        names = ["self-hosted-v18.20.5-ee", "self-hosted-v1.2.9-ee"]
        assert next_tag(names, "self-hosted-v18.2.") == "self-hosted-v18.2.0-ee"


class TestHeadAlreadyTagged:
    """Idempotency: a line tag pointing exactly at HEAD means nothing to do."""

    PREFIX = "self-hosted-v18.2."

    def test_true_when_a_line_tag_points_at_head(self):
        tags = [_tag("self-hosted-v18.2.3-ee", "abc123")]
        assert head_already_tagged(tags, "abc123", self.PREFIX) is True

    def test_false_when_line_tag_points_at_another_commit(self):
        tags = [_tag("self-hosted-v18.2.3-ee", "old000")]
        assert head_already_tagged(tags, "abc123", self.PREFIX) is False

    def test_false_when_only_other_lines_point_at_head(self):
        tags = [_tag("self-hosted-v18.20.0-ee", "abc123")]
        assert head_already_tagged(tags, "abc123", self.PREFIX) is False

    def test_ignores_decorated_tags_pointing_at_head(self):
        tags = [_tag("self-hosted-v18.2.0-ee-rc1", "abc123")]
        assert head_already_tagged(tags, "abc123", self.PREFIX) is False


class TestRequestFailsLoudly:
    """API failures log status and body to stderr and exit non-zero."""

    def test_http_error_logs_body_and_exits(self, capsys):
        err = urllib.error.HTTPError(
            url="https://gitlab.example/api",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"403 Forbidden - protected tag"}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                _request("POST", "/tags", "token", {"tag_name": "t", "ref": "sha"})

        assert exc.value.code == 1
        stderr = capsys.readouterr().err
        assert "HTTP 403" in stderr
        assert "protected tag" in stderr

    def test_url_error_logs_reason_and_exits(self, capsys):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(SystemExit) as exc:
                _request("GET", "/tags", "token")

        assert exc.value.code == 1
        assert "connection refused" in capsys.readouterr().err
