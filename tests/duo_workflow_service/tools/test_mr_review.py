"""Tests for SubmitMrReview tool and associated helper functions."""

# pylint: disable=redefined-outer-name,too-many-lines
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.mr_review import (
    DiffLine,
    MrReviewComment,
    SubmitMrReview,
    SubmitMrReviewInput,
    find_line_by_anchor,
    find_line_by_content,
    find_line_by_numbers,
    match_comment_to_diff_line,
    parse_diff_lines,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": None,
    }


# --- Fixtures for common test data ---


@pytest.fixture
def diff_refs():
    return {
        "base_sha": "aaa111",
        "start_sha": "bbb222",
        "head_sha": "ccc333",
    }


@pytest.fixture
def mr_data(diff_refs):
    return {
        "iid": 45,
        "diff_refs": diff_refs,
        "web_url": "https://gitlab.com/group/project/-/merge_requests/45",
    }


@pytest.fixture
def sample_diff():
    return (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -10,6 +10,8 @@\n"
        " context_line_1\n"
        " context_line_2\n"
        "+added_line_1\n"
        "+added_line_2\n"
        " context_line_3\n"
        "-deleted_line\n"
        " context_line_4\n"
    )


@pytest.fixture
def diffs_api_response(sample_diff):
    return [
        {
            "new_path": "app.py",
            "old_path": "app.py",
            "diff": sample_diff,
        }
    ]


# --- Tests for parse_diff_lines ---


class TestParseDiffLines:
    def test_parse_basic_diff(self, sample_diff):
        lines = parse_diff_lines(sample_diff)
        assert len(lines) > 0

        # Check context lines
        context_lines = [dl for dl in lines if dl.line_type == "context"]
        assert len(context_lines) == 4  # context_line_1, 2, 3, 4

        # Check added lines
        added_lines = [dl for dl in lines if dl.line_type == "added"]
        assert len(added_lines) == 2

        # Check deleted lines
        deleted_lines = [dl for dl in lines if dl.line_type == "deleted"]
        assert len(deleted_lines) == 1

    def test_parse_hunk_header_sets_line_numbers(self):
        diff = "@@ -5,3 +10,4 @@\n context\n+added\n-deleted\n"
        lines = parse_diff_lines(diff)

        # First line is context starting at old=5, new=10
        assert lines[0].old_line == 5
        assert lines[0].new_line == 10
        assert lines[0].line_type == "context"

        # Added line has new_line=11, no old_line
        assert lines[1].old_line is None
        assert lines[1].new_line == 11
        assert lines[1].line_type == "added"

        # Deleted line has old_line=6, no new_line
        assert lines[2].old_line == 6
        assert lines[2].new_line is None
        assert lines[2].line_type == "deleted"

    def test_parse_empty_diff(self):
        assert not parse_diff_lines("")

    def test_parse_diff_skips_metadata_lines(self):
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            " unchanged\n"
            "+new\n"
        )
        lines = parse_diff_lines(diff)
        assert len(lines) == 2
        assert lines[0].text == "unchanged"
        assert lines[1].text == "new"

    def test_parse_diff_skips_no_newline_marker(self):
        diff = "@@ -1,1 +1,1 @@\n-old\n+new\n\\ No newline at end of file\n"
        lines = parse_diff_lines(diff)
        assert len(lines) == 2

    def test_parse_skips_lines_before_any_hunk_header(self):
        # Without a parseable @@ header there are no valid line numbers, so
        # nothing is emitted (previously these defaulted to line 1).
        assert not parse_diff_lines("+added without header\n context\n")

    def test_parse_skips_lines_under_malformed_hunk_header(self):
        # A hunk header we can't parse must not attribute its lines to bogus
        # positions; a later valid header resumes normal parsing.
        diff = "@@ bogus header @@\n+orphan line\n@@ -3,2 +7,2 @@\n context\n+added\n"
        lines = parse_diff_lines(diff)
        assert [line.text for line in lines] == ["context", "added"]
        assert lines[0].new_line == 7
        assert lines[1].new_line == 8
        assert all(not dl.text.startswith("\\") for dl in lines)

    def test_parse_multiple_hunks(self):
        diff = (
            "@@ -1,2 +1,2 @@\n"
            " first_context\n"
            "+first_added\n"
            "@@ -50,2 +51,2 @@\n"
            " second_context\n"
            "+second_added\n"
        )
        lines = parse_diff_lines(diff)
        assert len(lines) == 4

        # After second hunk header, line numbers reset
        assert lines[2].old_line == 50
        assert lines[2].new_line == 51
        assert lines[3].new_line == 52

    def test_context_line_without_leading_space(self):
        """Lines that don't start with +, -, or space are treated as context."""
        diff = "@@ -1,1 +1,1 @@\nbare_line\n"
        lines = parse_diff_lines(diff)
        assert len(lines) == 1
        assert lines[0].text == "bare_line"
        assert lines[0].line_type == "context"


# --- Tests for find_line_by_numbers ---


class TestFindLineByNumbers:
    def test_exact_match_both_lines(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
            DiffLine(old_line=6, new_line=11, text="bar", line_type="context"),
        ]
        result = find_line_by_numbers(5, 10, diff_lines)
        assert result is not None
        assert result.text == "foo"

    def test_fallback_to_new_line_only(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
            DiffLine(old_line=None, new_line=12, text="added", line_type="added"),
        ]
        # old_line doesn't match any, but new_line=12 does
        result = find_line_by_numbers(99, 12, diff_lines)
        assert result is not None
        assert result.text == "added"

    def test_no_match(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        result = find_line_by_numbers(99, 99, diff_lines)
        assert result is None

    def test_no_match_when_new_line_is_none(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        result = find_line_by_numbers(99, None, diff_lines)
        assert result is None


# --- Tests for find_line_by_content ---


class TestFindLineByContent:
    def test_content_match_with_enough_lines(self):
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="line_a", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="line_b", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="line_c", line_type="context"),
            DiffLine(old_line=4, new_line=4, text="line_d", line_type="context"),
        ]
        suggestion = ["line_a", "line_b", "line_c"]
        result = find_line_by_content(suggestion, diff_lines)
        assert result is not None
        assert result.text == "line_a"

    def test_below_threshold_returns_none(self):
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="line_a", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="line_b", line_type="context"),
        ]
        suggestion = ["line_a", "line_b"]  # only 2, threshold is 3
        result = find_line_by_content(suggestion, diff_lines)
        assert result is None

    def test_skips_deleted_lines(self):
        diff_lines = [
            DiffLine(old_line=1, new_line=None, text="deleted", line_type="deleted"),
            DiffLine(old_line=None, new_line=1, text="line_a", line_type="added"),
            DiffLine(old_line=2, new_line=2, text="line_b", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="line_c", line_type="context"),
        ]
        suggestion = ["line_a", "line_b", "line_c"]
        result = find_line_by_content(suggestion, diff_lines)
        assert result is not None
        assert result.text == "line_a"

    def test_no_match_returns_none(self):
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="aaa", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="bbb", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="ccc", line_type="context"),
        ]
        suggestion = ["xxx", "yyy", "zzz"]
        result = find_line_by_content(suggestion, diff_lines)
        assert result is None


# --- Tests for match_comment_to_diff_line ---


class TestFindLineByAnchor:
    def test_exact_text_match_corrects_wrong_number(self):
        """The flagged line is located by its text even when the number is wrong."""
        diff_lines = [
            DiffLine(
                old_line=None, new_line=256, text="def show(id):", line_type="added"
            ),
            DiffLine(
                old_line=None,
                new_line=257,
                text="    record = Record.find(id)",
                line_type="added",
            ),
        ]
        # Model reported new_line=247 (an original-file number, not in the diff).
        result = find_line_by_anchor("    record = Record.find(id)", 247, diff_lines)
        assert result is not None
        assert result.new_line == 257

    def test_whitespace_tolerant_match(self):
        """Indentation the model copied imperfectly still matches."""
        diff_lines = [
            DiffLine(
                old_line=None,
                new_line=42,
                text="        do_thing()",
                line_type="added",
            ),
        ]
        result = find_line_by_anchor("do_thing()", None, diff_lines)
        assert result is not None
        assert result.new_line == 42

    def test_multiple_matches_picks_closest_to_claimed_line(self):
        diff_lines = [
            DiffLine(old_line=None, new_line=10, text="    log()", line_type="added"),
            DiffLine(old_line=None, new_line=50, text="    log()", line_type="added"),
            DiffLine(old_line=None, new_line=90, text="    log()", line_type="added"),
        ]
        result = find_line_by_anchor("    log()", 48, diff_lines)
        assert result is not None
        assert result.new_line == 50

    def test_multiple_matches_no_claimed_line_returns_first(self):
        """With no claimed line to disambiguate, the first match wins."""
        diff_lines = [
            DiffLine(old_line=None, new_line=10, text="    log()", line_type="added"),
            DiffLine(old_line=None, new_line=50, text="    log()", line_type="added"),
        ]
        result = find_line_by_anchor("    log()", None, diff_lines)
        assert result is not None
        assert result.new_line == 10

    def test_skips_deleted_lines(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=None, text="gone", line_type="deleted"),
        ]
        assert find_line_by_anchor("gone", None, diff_lines) is None

    def test_no_match_returns_none(self):
        diff_lines = [
            DiffLine(old_line=None, new_line=10, text="foo", line_type="added"),
        ]
        assert find_line_by_anchor("bar", 10, diff_lines) is None

    def test_empty_target_returns_none(self):
        diff_lines = [
            DiffLine(old_line=None, new_line=10, text="foo", line_type="added"),
        ]
        assert find_line_by_anchor("", 10, diff_lines) is None


class TestMatchCommentToDiffLine:
    def test_target_code_corrects_wrong_line_number(self):
        """The bug repro: model's new_line isn't in the diff, target_code rescues it."""
        diff_lines = [
            DiffLine(
                old_line=None, new_line=256, text="def show(id):", line_type="added"
            ),
            DiffLine(
                old_line=None,
                new_line=257,
                text="    record = Record.find(id)",
                line_type="added",
            ),
        ]
        comment = MrReviewComment(
            file="app.py",
            new_line=247,  # original-file number, not in the diff
            body="IDOR",
            target_code="    record = Record.find(id)",
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.new_line == 257

    def test_target_code_absent_keeps_number_behavior(self):
        """Back-compat: without target_code, behavior is unchanged (number match)."""
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        comment = MrReviewComment(file="app.py", new_line=10, old_line=5, body="issue")
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.new_line == 10

    def test_target_code_agreeing_with_number_uses_number_line(self):
        diff_lines = [
            DiffLine(old_line=None, new_line=10, text="foo", line_type="added"),
        ]
        comment = MrReviewComment(
            file="app.py", new_line=10, body="issue", target_code="foo"
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.new_line == 10

    def test_line_number_match_without_suggestion(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        comment = MrReviewComment(file="app.py", new_line=10, old_line=5, body="issue")
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.text == "foo"

    def test_suggestion_below_threshold_uses_line_match(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        comment = MrReviewComment(
            file="app.py", new_line=10, old_line=5, body="issue", suggestion="ab"
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.text == "foo"

    def test_suggestion_match_overrides_wrong_line_numbers(self):
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="line_a", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="line_b", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="line_c", line_type="context"),
            DiffLine(old_line=4, new_line=4, text="line_d", line_type="context"),
        ]
        comment = MrReviewComment(
            file="app.py",
            new_line=99,
            old_line=99,
            body="issue",
            suggestion="line_a\nline_b\nline_c",
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.text == "line_a"
        assert result.new_line == 1

    def test_suggestion_content_matches_line_match(self):
        """When line_match text equals first suggestion line, line_match is returned."""
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="line_a", line_type="context"),
            DiffLine(old_line=6, new_line=11, text="line_b", line_type="context"),
            DiffLine(old_line=7, new_line=12, text="line_c", line_type="context"),
        ]
        comment = MrReviewComment(
            file="app.py",
            new_line=10,
            old_line=5,
            body="issue",
            suggestion="line_a\nline_b\nline_c",
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.text == "line_a"
        assert result.new_line == 10

    def test_no_suggestion_no_line_match(self):
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        comment = MrReviewComment(file="app.py", new_line=99, old_line=99, body="issue")
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is None

    def test_content_fallback_when_line_match_wrong_text(self):
        """When line_match exists but text doesn't match first suggestion line, content search is tried."""
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="wrong", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="target_a", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="target_b", line_type="context"),
            DiffLine(old_line=4, new_line=4, text="target_c", line_type="context"),
        ]
        comment = MrReviewComment(
            file="app.py",
            new_line=1,
            old_line=1,
            body="issue",
            suggestion="target_a\ntarget_b\ntarget_c",
        )
        result = match_comment_to_diff_line(comment, diff_lines)
        assert result is not None
        assert result.text == "target_a"
        assert result.new_line == 2


# --- Tests for SubmitMrReview tool ---


class TestSubmitMrReview:
    @pytest.mark.asyncio
    async def test_execute_summary_only(self, metadata, gitlab_client_mock):
        """Test submitting a review with just a summary and verdict, no inline comments."""
        # _is_public_project -> private
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "private"}),
            )
        )
        # _bulk_publish
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[],
            verdict="reviewed",
            summary="All good",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "reviewed" in parsed["message"]
        assert "summary note posted" in parsed["message"]

    @pytest.mark.asyncio
    async def test_execute_with_inline_comments(
        self, metadata, gitlab_client_mock, mr_data, diffs_api_response
    ):
        """Test submitting a review with inline comments on a private project.

        No fold flag is passed, so the visibility lookup is skipped entirely.
        """
        aget_responses = [
            # _fetch_mr_data
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(mr_data),
            ),
            # _fetch_diffs_by_file
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(diffs_api_response),
            ),
        ]
        gitlab_client_mock.aget = AsyncMock(side_effect=aget_responses)
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "app.py", "new_line": 12, "body": "Security issue here"},
            ],
            verdict="requested_changes",
            summary="Found 1 issue",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "1 inline comment(s) posted" in parsed["message"]
        assert "requested_changes" in parsed["message"]

    @pytest.mark.asyncio
    async def test_execute_draft_note_failure_is_counted_not_fatal(
        self, metadata, gitlab_client_mock, mr_data, diffs_api_response
    ):
        """A draft note that fails to post is counted as failed; the review still completes."""
        aget_responses = [
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_api_response)),
        ]
        gitlab_client_mock.aget = AsyncMock(side_effect=aget_responses)
        # First apost (draft note) fails with 403; second apost (bulk_publish) succeeds.
        gitlab_client_mock.apost = AsyncMock(
            side_effect=[
                GitLabHttpResponse(status_code=403, body="Forbidden"),
                GitLabHttpResponse(status_code=200, body="{}"),
            ]
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "app.py", "new_line": 12, "body": "Security issue here"},
            ],
            verdict="reviewed",
            summary="Done",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "1 inline comment(s) could not be posted" in parsed["message"]
        assert "inline comment(s) posted" not in parsed["message"]
        # bulk_publish still ran despite the failed draft note
        post_calls = gitlab_client_mock.apost.call_args_list
        bulk = [c for c in post_calls if "bulk_publish" in str(c)]
        assert len(bulk) == 1
        # The comment that failed to post inline is folded into the summary (forced
        # internal) rather than silently dropped.
        body = json.loads(bulk[0].kwargs["body"])
        assert body["internal"] is True
        assert "Done" in body["note"]
        assert "Security issue here" in body["note"]
        assert "could not be anchored" in body["note"]

    @pytest.mark.asyncio
    async def test_execute_public_project_folds_comments_into_summary(
        self, metadata, gitlab_client_mock
    ):
        """With the opt-in flag, public projects fold inline comments into an internal summary.

        Without a custom title the neutral default heading is used (no domain wording).
        """
        gitlab_client_mock.aget = AsyncMock(
            # only _is_public_project -> public; no MR fetch since comments are folded
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "public"}),
            )
        )
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "app.py", "new_line": 10, "body": "Vulnerability found"},
            ],
            verdict="requested_changes",
            summary="Original summary",
            fold_inline_into_summary_when_public=True,
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Comments were folded, so 0 inline comments
        assert "inline comment" not in parsed["message"]
        # But summary was posted
        assert "internal summary note posted" in parsed["message"]

        # Verify bulk_publish was called with the folded summary
        post_calls = gitlab_client_mock.apost.call_args_list
        bulk_publish_call = [c for c in post_calls if "bulk_publish" in str(c)]
        assert len(bulk_publish_call) == 1
        body = json.loads(bulk_publish_call[0].kwargs["body"])
        assert body["internal"] is True
        # Neutral default heading — the tool carries no security-specific wording
        assert "Inline findings" in body["note"]
        assert "Security Findings" not in body["note"]
        assert "Original summary" in body["note"]

    @pytest.mark.asyncio
    async def test_execute_public_project_fold_uses_caller_title(
        self, metadata, gitlab_client_mock
    ):
        """The caller can supply the folded-findings heading verbatim (e.g. the security flow)."""
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "public"}),
            )
        )
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        title = "**Security Findings** (internal only — this project is public)"
        tool = SubmitMrReview(metadata=metadata)
        await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "app.py", "new_line": 10, "body": "Vulnerability found"},
            ],
            verdict="requested_changes",
            summary="Original summary",
            fold_inline_into_summary_when_public=True,
            inline_findings_title=title,
        )

        post_calls = gitlab_client_mock.apost.call_args_list
        bulk_publish_call = [c for c in post_calls if "bulk_publish" in str(c)]
        body = json.loads(bulk_publish_call[0].kwargs["body"])
        assert title in body["note"]

    @pytest.mark.asyncio
    async def test_execute_public_project_without_flag_posts_inline(
        self, metadata, gitlab_client_mock, mr_data, diffs_api_response
    ):
        """Without the opt-in flag, inline comments are posted as-is even on a public project.

        No visibility lookup happens, and summary_internal is not forced.
        """
        aget_responses = [
            # _fetch_mr_data
            GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
            # _fetch_diffs_by_file
            GitLabHttpResponse(status_code=200, body=json.dumps(diffs_api_response)),
        ]
        gitlab_client_mock.aget = AsyncMock(side_effect=aget_responses)
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "app.py", "new_line": 12, "body": "Issue here"},
            ],
            verdict="requested_changes",
            summary="Public summary",
        )

        parsed = json.loads(result)
        assert "1 inline comment(s) posted" in parsed["message"]
        # No visibility lookup was performed (no projects API call)
        assert all(
            "/projects/123" not in str(c) or "merge_requests" in str(c)
            for c in gitlab_client_mock.aget.call_args_list
        )
        # summary_internal not forced — posted as a public note
        post_calls = gitlab_client_mock.apost.call_args_list
        bulk_publish_call = [c for c in post_calls if "bulk_publish" in str(c)]
        body = json.loads(bulk_publish_call[0].kwargs["body"])
        assert body["internal"] is False

    @pytest.mark.asyncio
    async def test_execute_no_diff_refs_raises(self, metadata, gitlab_client_mock):
        """When MR has no diff_refs, a ToolException should be raised."""
        aget_responses = [
            # _fetch_mr_data - no diff_refs
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"iid": 45}),
            ),
        ]
        gitlab_client_mock.aget = AsyncMock(side_effect=aget_responses)

        tool = SubmitMrReview(metadata=metadata)
        with pytest.raises(ToolException, match="diff_refs"):
            await tool._execute(
                project_id=123,
                merge_request_iid=45,
                comments=[{"file": "app.py", "new_line": 10, "body": "issue"}],
            )

    @pytest.mark.asyncio
    async def test_execute_skips_unresolvable_comments(
        self, metadata, gitlab_client_mock, mr_data
    ):
        """Comments that can't be resolved to a diff line are skipped."""
        aget_responses = [
            # _fetch_mr_data
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps(mr_data),
            ),
            # _fetch_diffs_by_file - empty diffs
            GitLabHttpResponse(
                status_code=200,
                body=json.dumps([]),
            ),
        ]
        gitlab_client_mock.aget = AsyncMock(side_effect=aget_responses)
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {"file": "nonexistent.py", "new_line": 1, "body": "issue"},
            ],
            verdict="reviewed",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # nothing posted since file not in diffs, and the failure is reported honestly
        assert "inline comment(s) posted" not in parsed["message"]
        assert "1 inline comment(s) could not be posted" in parsed["message"]

    @pytest.mark.asyncio
    async def test_execute_target_code_corrects_line_and_posts_inline(
        self, metadata, gitlab_client_mock, mr_data, diffs_api_response
    ):
        """A finding whose new_line isn't in the diff still posts inline, on the line located by target_code (the
        #603790 fix)."""
        gitlab_client_mock.aget = AsyncMock(
            side_effect=[
                GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
                GitLabHttpResponse(
                    status_code=200, body=json.dumps(diffs_api_response)
                ),
            ]
        )
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {
                    "file": "app.py",
                    "new_line": 999,  # not in the diff (original-file numbering)
                    "target_code": "added_line_1",  # added_line_1 is at new_line=12
                    "body": "Security issue here",
                },
            ],
            verdict="requested_changes",
            summary="Done",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "1 inline comment(s) posted" in parsed["message"]
        assert "could not be posted" not in parsed["message"]

        # The draft note was created on the corrected line (12), not the reported 999.
        post_calls = gitlab_client_mock.apost.call_args_list
        draft_note_calls = [
            c
            for c in post_calls
            if "draft_notes" in str(c) and "bulk_publish" not in str(c)
        ]
        assert len(draft_note_calls) == 1
        body = json.loads(draft_note_calls[0].kwargs["body"])
        assert body["position"]["new_line"] == 12

    @pytest.mark.asyncio
    async def test_execute_unanchorable_finding_folded_into_summary(
        self, metadata, gitlab_client_mock, mr_data, diffs_api_response
    ):
        """A finding that can't be anchored is folded into the internal summary (never silently dropped) and the count
        stays honest."""
        gitlab_client_mock.aget = AsyncMock(
            side_effect=[
                GitLabHttpResponse(status_code=200, body=json.dumps(mr_data)),
                GitLabHttpResponse(
                    status_code=200, body=json.dumps(diffs_api_response)
                ),
            ]
        )
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            comments=[
                {
                    "file": "app.py",
                    "new_line": 999,  # not in the diff
                    "target_code": "this text is nowhere in the diff",
                    "body": "Unanchorable critical finding",
                },
            ],
            verdict="requested_changes",
            summary="Original summary",
            summary_internal=False,  # tool must force this True when folding
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "1 inline comment(s) could not be posted" in parsed["message"]

        post_calls = gitlab_client_mock.apost.call_args_list
        # No draft note was created — only bulk_publish ran.
        assert not [
            c
            for c in post_calls
            if "draft_notes" in str(c) and "bulk_publish" not in str(c)
        ]
        bulk = [c for c in post_calls if "bulk_publish" in str(c)]
        assert len(bulk) == 1
        body = json.loads(bulk[0].kwargs["body"])
        # Folded into the summary, forced internal, finding body preserved.
        assert body["internal"] is True
        assert "Original summary" in body["note"]
        assert "Unanchorable critical finding" in body["note"]
        assert "could not be anchored" in body["note"]

    @pytest.mark.asyncio
    async def test_execute_malformed_comment_raises(self, metadata, gitlab_client_mock):
        """Malformed comments should raise so the agent framework can retry."""
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "private"}),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        with pytest.raises(Exception):
            await tool._execute(
                project_id=123,
                merge_request_iid=45,
                comments=[{"file": "app.py", "new_line": 10}],
                verdict="reviewed",
            )

    @pytest.mark.asyncio
    async def test_execute_no_actions(self, metadata, gitlab_client_mock):
        """No comments, no verdict, no summary -> 'no actions taken'."""
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "private"}),
            )
        )
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "no actions taken" in parsed["message"]


class TestSubmitMrReviewFetchMrData:
    @pytest.mark.asyncio
    async def test_fetch_mr_data(self, metadata, gitlab_client_mock, mr_data):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(mr_data),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._fetch_mr_data(123, 45)

        assert result["iid"] == 45
        assert "diff_refs" in result
        gitlab_client_mock.aget.assert_called_once_with(
            "/api/v4/projects/123/merge_requests/45", parse_json=False
        )


class TestSubmitMrReviewFetchDiffsByFile:
    @pytest.mark.asyncio
    async def test_fetch_diffs_by_file(
        self, metadata, gitlab_client_mock, diffs_api_response
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(diffs_api_response),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._fetch_diffs_by_file(123, 45)

        assert "app.py" in result
        assert len(result["app.py"]) > 0
        gitlab_client_mock.aget.assert_called_once_with(
            "/api/v4/projects/123/merge_requests/45/diffs", parse_json=False
        )

    @pytest.mark.asyncio
    async def test_fetch_diffs_skips_empty_diffs(self, metadata, gitlab_client_mock):
        diffs_data = [
            {"new_path": "empty.py", "old_path": "empty.py", "diff": ""},
            {
                "new_path": "valid.py",
                "old_path": "valid.py",
                "diff": "@@ -1,1 +1,1 @@\n+line\n",
            },
        ]
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(diffs_data),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._fetch_diffs_by_file(123, 45)

        assert "empty.py" not in result
        assert "valid.py" in result


class TestSubmitMrReviewResolveCommentPosition:
    def test_resolve_exact_match(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        diffs_by_file = {"app.py": diff_lines}
        comment = MrReviewComment(file="app.py", new_line=10, old_line=5, body="issue")

        result = tool._resolve_comment_position(comment, diffs_by_file)
        assert result is not None
        assert result.new_line == 10
        assert result.old_line == 5

    def test_resolve_corrects_line_numbers(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        diff_lines = [
            DiffLine(old_line=1, new_line=1, text="target_a", line_type="context"),
            DiffLine(old_line=2, new_line=2, text="target_b", line_type="context"),
            DiffLine(old_line=3, new_line=3, text="target_c", line_type="context"),
        ]
        diffs_by_file = {"app.py": diff_lines}
        comment = MrReviewComment(
            file="app.py",
            new_line=99,
            old_line=99,
            body="issue",
            suggestion="target_a\ntarget_b\ntarget_c",
        )

        result = tool._resolve_comment_position(comment, diffs_by_file)
        assert result is not None
        assert result.new_line == 1
        assert result.old_line == 1

    def test_resolve_file_not_in_diffs(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(
            file="missing.py", new_line=10, old_line=5, body="issue"
        )

        result = tool._resolve_comment_position(comment, {})
        assert result is None

    def test_resolve_no_matching_line(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        diff_lines = [
            DiffLine(old_line=5, new_line=10, text="foo", line_type="context"),
        ]
        diffs_by_file = {"app.py": diff_lines}
        comment = MrReviewComment(
            file="app.py", new_line=999, old_line=999, body="issue"
        )

        result = tool._resolve_comment_position(comment, diffs_by_file)
        assert result is None


class TestSubmitMrReviewCreateDraftNote:
    @pytest.mark.asyncio
    async def test_create_draft_note_with_position(
        self, metadata, gitlab_client_mock, diff_refs
    ):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=201, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(
            file="app.py", new_line=10, old_line=5, body="Security issue"
        )
        await tool._create_draft_note(123, 45, diff_refs, comment)

        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        assert call_args.kwargs["path"] == (
            "/api/v4/projects/123/merge_requests/45/draft_notes"
        )
        payload = json.loads(call_args.kwargs["body"])
        assert payload["note"] == "Security issue"
        assert payload["position"]["new_line"] == 10
        assert payload["position"]["old_line"] == 5
        assert payload["position"]["base_sha"] == "aaa111"
        assert payload["position"]["new_path"] == "app.py"

    @pytest.mark.asyncio
    async def test_create_draft_note_with_suggestion(
        self, metadata, gitlab_client_mock, diff_refs
    ):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=201, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(
            file="app.py",
            new_line=10,
            body="Replace this",
            suggestion="fixed_code()",
        )
        await tool._create_draft_note(123, 45, diff_refs, comment)

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert "```suggestion:-0+0" in payload["note"]
        assert "fixed_code()" in payload["note"]

    @pytest.mark.asyncio
    async def test_create_draft_note_new_line_only(
        self, metadata, gitlab_client_mock, diff_refs
    ):
        """When only new_line is set, old_line should not be in the position."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=201, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(file="app.py", new_line=10, body="Added line issue")
        await tool._create_draft_note(123, 45, diff_refs, comment)

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert "new_line" in payload["position"]
        assert "old_line" not in payload["position"]

    @pytest.mark.asyncio
    async def test_create_draft_note_fixes_newlines_in_body(
        self, metadata, gitlab_client_mock, diff_refs
    ):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=201, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(file="app.py", new_line=10, body="line1\\nline2")
        await tool._create_draft_note(123, 45, diff_refs, comment)

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert payload["note"] == "line1\nline2"

    @pytest.mark.asyncio
    async def test_create_draft_note_raises_on_error(
        self, metadata, gitlab_client_mock, diff_refs
    ):
        """A non-2xx response (which the client does not raise on for 4xx) must surface."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=403, body="Forbidden")
        )

        tool = SubmitMrReview(metadata=metadata)
        comment = MrReviewComment(file="app.py", new_line=10, body="issue")
        with pytest.raises(ToolException):
            await tool._create_draft_note(123, 45, diff_refs, comment)


class TestSubmitMrReviewBulkPublish:
    @pytest.mark.asyncio
    async def test_bulk_publish_with_verdict_and_summary(
        self, metadata, gitlab_client_mock
    ):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        await tool._bulk_publish(123, 45, "requested_changes", "Found issues", True)

        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        assert "bulk_publish" in call_args.kwargs["path"]
        payload = json.loads(call_args.kwargs["body"])
        assert payload["reviewer_state"] == "requested_changes"
        # Summary is posted exactly as the caller supplied it — no appended disclaimer.
        assert payload["note"] == "Found issues"
        assert payload["internal"] is True

    @pytest.mark.asyncio
    async def test_bulk_publish_raises_on_error(self, metadata, gitlab_client_mock):
        """bulk_publish is the critical publish — a non-2xx must raise, not report success."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=400, body="Bad Request")
        )

        tool = SubmitMrReview(metadata=metadata)
        with pytest.raises(ToolException):
            await tool._bulk_publish(123, 45, "reviewed", None, False)


class TestSubmitMrReviewTrustLevel:
    def test_defaults_to_untrusted_user_content(self, metadata):
        # SubmitMrReview writes to remote MRs (not local fs/git), so it must not be
        # TRUSTED_INTERNAL — its output goes through prompt-injection scanning.
        tool = SubmitMrReview(metadata=metadata)
        assert tool.trust_level == ToolTrustLevel.UNTRUSTED_USER_CONTENT

    @pytest.mark.asyncio
    async def test_bulk_publish_no_verdict_no_summary(
        self, metadata, gitlab_client_mock
    ):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        await tool._bulk_publish(123, 45, None, None, False)

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert "reviewer_state" not in payload
        assert "note" not in payload

    @pytest.mark.asyncio
    async def test_bulk_publish_summary_posted_as_is(
        self, metadata, gitlab_client_mock
    ):
        """The tool posts the summary verbatim — any disclaimer/footer is the caller's job."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SubmitMrReview(metadata=metadata)
        await tool._bulk_publish(123, 45, None, "Summary text", False)

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert payload["note"] == "Summary text"
        assert "SAST" not in payload["note"]
        assert payload["internal"] is False


class TestIsPublicProject:
    @pytest.mark.asyncio
    async def test_public_project(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "public"}),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._is_public_project(123)
        assert result is True

    @pytest.mark.asyncio
    async def test_private_project(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "private"}),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._is_public_project(123)
        assert result is False

    @pytest.mark.asyncio
    async def test_internal_project(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps({"visibility": "internal"}),
            )
        )

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._is_public_project(123)
        assert result is False

    @pytest.mark.asyncio
    async def test_api_error_fails_secure_to_public(self, metadata, gitlab_client_mock):
        # If visibility can't be determined we fail secure: assume public so findings
        # stay in the internal-only note rather than being posted as public comments.
        gitlab_client_mock.aget = AsyncMock(side_effect=Exception("API error"))

        tool = SubmitMrReview(metadata=metadata)
        result = await tool._is_public_project(123)
        assert result is True


class TestBuildPublicProjectSummary:
    def test_with_existing_summary(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comments = [
            MrReviewComment(
                file="app.py", new_line=10, body="Vuln found", suggestion=None
            ),
        ]
        result = tool._build_public_project_summary(comments, "Existing summary")

        assert "Existing summary" in result
        # Neutral default heading — no domain-specific wording in the tool
        assert "Inline findings" in result
        assert "Security Findings" not in result
        assert "app.py" in result
        assert "`10`" in result

    def test_uses_caller_supplied_title(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comments = [
            MrReviewComment(file="app.py", new_line=10, body="Vuln found"),
        ]
        title = "**Security Findings** (internal only — this project is public)"
        result = tool._build_public_project_summary(comments, None, title)

        assert title in result
        assert "Inline findings" not in result

    def test_without_existing_summary(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comments = [
            MrReviewComment(file="app.py", new_line=10, body="Vuln found"),
            MrReviewComment(file="lib.py", body="Another issue"),
        ]
        result = tool._build_public_project_summary(comments, None)

        assert "Inline findings" in result
        assert "app.py" in result
        assert "lib.py" in result
        assert "**1.**" in result
        assert "**2.**" in result

    def test_comment_without_new_line(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comments = [
            MrReviewComment(file="app.py", body="General issue"),
        ]
        result = tool._build_public_project_summary(comments, None)

        assert "app.py" in result
        # No line reference when new_line is None
        assert ":`" not in result

    def test_fixes_newlines_in_body(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        comments = [
            MrReviewComment(file="app.py", new_line=1, body="line1\\nline2"),
        ]
        result = tool._build_public_project_summary(comments, None)

        assert "line1\nline2" in result

    def test_fixes_newlines_in_existing_summary(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        result = tool._build_public_project_summary([], "summary\\ntext")

        assert "summary\ntext" in result


class TestFixNewlines:
    def test_replaces_literal_backslash_n(self):
        assert SubmitMrReview._fix_newlines("hello\\nworld") == "hello\nworld"

    def test_no_change_needed(self):
        assert SubmitMrReview._fix_newlines("hello world") == "hello world"

    def test_multiple_replacements(self):
        assert SubmitMrReview._fix_newlines("a\\nb\\nc") == "a\nb\nc"

    def test_empty_string(self):
        assert SubmitMrReview._fix_newlines("") == ""


class TestBuildResponse:
    def test_with_all_fields(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        result = tool._build_response(45, 3, 0, "requested_changes", "Summary", False)
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "3 inline comment(s) posted" in parsed["message"]
        assert "requested_changes" in parsed["message"]
        assert "public summary note posted" in parsed["message"]

    def test_internal_summary(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        result = tool._build_response(45, 0, 0, None, "Summary", True)
        parsed = json.loads(result)
        assert "internal summary note posted" in parsed["message"]

    def test_no_actions(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        result = tool._build_response(45, 0, 0, None, None, False)
        parsed = json.loads(result)
        assert "no actions taken" in parsed["message"]

    def test_reports_failed_comments(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        result = tool._build_response(45, 2, 1, "reviewed", None, False)
        parsed = json.loads(result)
        assert "2 inline comment(s) posted" in parsed["message"]
        assert "1 inline comment(s) could not be posted" in parsed["message"]


class TestFormatDisplayMessage:
    def test_format_display_message(self, metadata):
        tool = SubmitMrReview(metadata=metadata)
        args = SubmitMrReviewInput(project_id=123, merge_request_iid=45)
        message = tool.format_display_message(args)

        assert "!45" in message
        assert "123" in message
