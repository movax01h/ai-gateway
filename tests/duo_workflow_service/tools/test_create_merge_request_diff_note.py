# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    CreateMergeRequestDiffNote,
    CreateMergeRequestDiffNoteInput,
)

URL_ERROR_CASES = [
    # URL and project_id both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        "different%2Fproject",
        123,
        "Project ID mismatch",
    ),
    # URL and merge_request_iid both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        "namespace%2Fproject",
        456,
        "Merge Request ID mismatch",
    ),
    # URL given isn't a merge request URL (it's just a project URL)
    (
        "https://gitlab.com/namespace/project",
        None,
        None,
        "Failed to parse URL",
    ),
]

DIFF_REFS = {
    "base_sha": "abc123",
    "head_sha": "def456",
    "start_sha": "ghi789",
}


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=["**/*.log", "/secrets/**", "**/node_modules/**"],
    )


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project_mock,
    }


@pytest.fixture(name="mr_response")
def mr_response_fixture():
    """Mock MR GET response containing diff_refs."""
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps({"id": 9, "diff_refs": DIFF_REFS}),
    )


@pytest.fixture(name="discussion_response")
def discussion_response_fixture():
    """Mock discussion POST response."""
    return GitLabHttpResponse(
        status_code=200,
        body={
            "id": "disc123",
            "notes": [{"id": 1, "body": "suggestion note", "type": "DiffNote"}],
        },
    )


@pytest.fixture(name="empty_diffs_response")
def empty_diffs_response_fixture():
    """Mock MR /diffs GET response with no diffs — keeps the no-op guard inert."""
    return GitLabHttpResponse(status_code=200, body=json.dumps([]))


def _aget_dispatch(*, mr_response, diffs_response):
    """Build an aget side_effect that returns the right response per endpoint."""

    async def _side_effect(path, **_kwargs):
        if path.endswith("/diffs"):
            return diffs_response
        return mr_response

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.asyncio
async def test_create_diff_note_with_new_line(
    gitlab_client_mock,
    metadata,
    mr_response,
    discussion_response,
    empty_diffs_response,
):
    """Test posting an inline note on an added line (new_line only)."""
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response, diffs_response=empty_diffs_response
    )
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="```suggestion:-0+0\nfixed code\n```",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=42,
    )

    result = json.loads(response)
    assert "created_diff_note" in result
    assert result["created_diff_note"]["id"] == "disc123"

    # Verify the MR was fetched for diff_refs
    gitlab_client_mock.aget.assert_any_call(
        path="/api/v4/projects/1/merge_requests/9",
        parse_json=False,
    )

    # Verify the discussion was created with correct payload
    expected_payload = {
        "body": "```suggestion:-0+0\nfixed code\n```",
        "position": {
            "position_type": "text",
            "base_sha": "abc123",
            "head_sha": "def456",
            "start_sha": "ghi789",
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "new_line": 42,
        },
    }
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/9/discussions",
        body=json.dumps(expected_payload),
    )


@pytest.mark.asyncio
async def test_create_diff_note_with_old_line(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting an inline note on a deleted line (old_line only)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="This line should not be removed",
        old_path="src/main.py",
        new_path="src/main.py",
        old_line=10,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_line"] == 10
    assert "new_line" not in payload["position"]


@pytest.mark.asyncio
async def test_create_diff_note_with_both_lines(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting an inline note on an unchanged/context line (both old_line and new_line)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Comment on context line",
        old_path="src/main.py",
        new_path="src/main.py",
        old_line=5,
        new_line=5,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_line"] == 5
    assert payload["position"]["new_line"] == 5


@pytest.mark.asyncio
async def test_create_diff_note_with_renamed_file(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting a note where old_path differs from new_path (file rename)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Note on renamed file",
        old_path="src/old_name.py",
        new_path="src/new_name.py",
        new_line=1,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_path"] == "src/old_name.py"
    assert payload["position"]["new_path"] == "src/new_name.py"


@pytest.mark.asyncio
async def test_create_diff_note_no_line_provided(metadata):
    """Test error when neither old_line nor new_line is provided."""
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body="Missing line info",
            old_path="src/main.py",
            new_path="src/main.py",
        )

    assert "old_line or new_line" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_diff_note_diff_refs_fetch_failure(gitlab_client_mock, metadata):
    """Test error when fetching diff_refs fails."""
    mr_error_response = GitLabHttpResponse(
        status_code=404,
        body="Not Found",
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mr_error_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException):
        await tool._arun(
            project_id=1,
            merge_request_iid=999,
            body="Note",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=1,
        )


@pytest.mark.asyncio
async def test_create_diff_note_diff_refs_missing(gitlab_client_mock, metadata):
    """Test error when MR response has no diff_refs field."""
    mr_response_no_refs = GitLabHttpResponse(
        status_code=200,
        body=json.dumps({"id": 9}),
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response_no_refs)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body="Note",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=1,
        )

    assert "diff_refs not available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_diff_note_post_exception(
    gitlab_client_mock, metadata, mr_response
):
    """Test exception handling when the discussion POST fails."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API error"))

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body="Note",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=1,
        )

    assert "API error" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_create_diff_note_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            url=url,
            project_id=project_id,
            merge_request_iid=merge_request_iid,
            body="Note",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=1,
        )

    assert error_contains in str(exc_info.value)

    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_mr_path,expected_disc_path",
    [
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions",
        ),
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions",
        ),
    ],
)
async def test_create_diff_note_with_url_success(
    url,
    project_id,
    merge_request_iid,
    expected_mr_path,
    expected_disc_path,
    gitlab_client_mock,
    metadata,
    mr_response,
    discussion_response,
):
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        body="Inline note",
        old_path="file.py",
        new_path="file.py",
        new_line=10,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_mr_path,
        parse_json=False,
    )
    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_disc_path,
        body=json.dumps(
            {
                "body": "Inline note",
                "position": {
                    "position_type": "text",
                    "base_sha": "abc123",
                    "head_sha": "def456",
                    "start_sha": "ghi789",
                    "old_path": "file.py",
                    "new_path": "file.py",
                    "new_line": 10,
                },
            }
        ),
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateMergeRequestDiffNoteInput(
                project_id=42,
                merge_request_iid=9,
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                new_line=10,
            ),
            "Add inline diff note to src/main.py:new_line=10 in merge request !9 in project 42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                project_id=42,
                merge_request_iid=9,
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                old_line=5,
            ),
            "Add inline diff note to src/main.py:old_line=5 in merge request !9 in project 42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                new_line=10,
            ),
            "Add inline diff note to src/main.py:new_line=10 in merge request "
            "https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                old_line=5,
            ),
            "Add inline diff note to src/main.py:old_line=5 in merge request "
            "https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_create_diff_note_format_display_message(input_data, expected_message):
    tool = CreateMergeRequestDiffNote(description="Create diff note")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.parametrize(
    "body,expected",
    [
        ("```suggestion\nfoo\n```", [(0, 0, ["foo"])]),
        # Embedded in body, indentation preserved
        ("ctx\n```suggestion\n  foo\n```\nmore", [(0, 0, ["  foo"])]),
        # Multi-line content with default offsets
        ("```suggestion\nfoo\nbar\n```", [(0, 0, ["foo", "bar"])]),
        # Non-default offsets
        ("```suggestion:-1+2\nfoo\n```", [(1, 2, ["foo"])]),
        # No block returns empty list
        ("just a comment", []),
        # Multiple blocks: each is parsed independently
        (
            "```suggestion\nfoo\n```\n```suggestion:-0+1\nbar\nbaz\n```",
            [(0, 0, ["foo"]), (0, 1, ["bar", "baz"])],
        ),
    ],
)
def test_extract_suggestion_blocks(body, expected):
    assert CreateMergeRequestDiffNote._extract_suggestion_blocks(body) == expected


_MULTI_HUNK_DIFF = "\n".join(
    [
        "@@ -1,1 +1,1 @@",
        " first",
        "@@ -10,3 +20,3 @@",
        " a",
        "+inserted",
        " b",
    ]
)
_RANGE_DIFF = "\n".join(
    [
        "@@ -40,3 +40,4 @@",
        " line40",
        " line41",
        "+added line",
        " line42",
    ]
)


@pytest.mark.parametrize(
    "diff,target_old,target_new,before,after,expected",
    [
        # Single added line by new_line
        (_RANGE_DIFF, None, 42, 0, 0, ["added line"]),
        # Single deleted line by old_line
        (
            "@@ -1,3 +1,2 @@\n line1\n-removed\n line2\n",
            2,
            None,
            0,
            0,
            ["removed"],
        ),
        # Multi-line range across context + added line
        (_RANGE_DIFF, None, 42, 1, 1, ["line41", "added line", "line42"]),
        # Range fully within a later hunk
        (_MULTI_HUNK_DIFF, None, 21, 0, 0, ["inserted"]),
        # Range outside diff context → None
        (_RANGE_DIFF, None, 42, 5, 0, None),
        # Both targets None → None
        ("@@ -1,1 +1,1 @@\n a\n", None, None, 0, 0, None),
        # Start before line 1 → None
        ("@@ -1,1 +1,1 @@\n a\n", None, 1, 5, 0, None),
    ],
)
def test_lines_at_range(diff, target_old, target_new, before, after, expected):
    assert (
        CreateMergeRequestDiffNote._lines_at_range(
            diff,
            target_old=target_old,
            target_new=target_new,
            before=before,
            after=after,
        )
        == expected
    )


@pytest.fixture(name="diffs_response_main_py")
def diffs_response_main_py_fixture():
    """Mock /diffs response with src/main.py having an added line at new_line=42."""
    diff = "\n".join(
        [
            "@@ -40,3 +40,4 @@",
            " line40",
            " line41",
            "+this.$emit('on-agents-load', this.agentsCount);",
            " line43",
        ]
    )
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            [
                {
                    "old_path": "src/main.py",
                    "new_path": "src/main.py",
                    "diff": diff,
                }
            ]
        ),
    )


@pytest.fixture(name="diffs_response_main_py_multiline")
def diffs_response_main_py_multiline_fixture():
    """Mock /diffs response covering enough context for multi-line suggestion checks."""
    diff = "\n".join(
        [
            "@@ -38,7 +38,7 @@",
            " line38",
            " line39",
            " line40",
            "-old41",
            "+line41",
            "+line42",
            " line43",
            " line44",
            " line45",
        ]
    )
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            [
                {
                    "old_path": "src/main.py",
                    "new_path": "src/main.py",
                    "diff": diff,
                }
            ]
        ),
    )


_TARGET_LINE = "this.$emit('on-agents-load', this.agentsCount);"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,refused",
    [
        # Single-line content matches the targeted line — refused.
        (f"```suggestion:-0+0\n{_TARGET_LINE}\n```", True),
        # Indentation-only change (linter fix) is allowed under strict equality.
        (f"```suggestion:-0+0\n    {_TARGET_LINE}\n```", False),
        # Different content — allowed.
        ("```suggestion:-0+0\nthis.$emit('agent-loaded');\n```", False),
        # Multi-line suggestion where any line differs — allowed.
        (f"```suggestion:-0+1\n{_TARGET_LINE}\ndifferent\n```", False),
        # Range extends past the diff context window — fail open, allowed.
        (f"```suggestion:-5+5\n{_TARGET_LINE}\n```", False),
        # No suggestion block — guard short-circuits, allowed.
        ("just a plain comment", False),
    ],
)
async def test_create_diff_note_no_op_guard(
    body,
    refused,
    gitlab_client_mock,
    metadata,
    mr_response,
    diffs_response_main_py,
    discussion_response,
):
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response, diffs_response=diffs_response_main_py
    )
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)
    tool = CreateMergeRequestDiffNote(metadata=metadata)
    args = {
        "project_id": 1,
        "merge_request_iid": 9,
        "body": body,
        "old_path": "src/main.py",
        "new_path": "src/main.py",
        "new_line": 42,
    }

    if refused:
        with pytest.raises(ToolException, match="identical to the targeted line"):
            await tool._arun(**args)
        gitlab_client_mock.apost.assert_not_called()
    else:
        response = await tool._arun(**args)
        assert "created_diff_note" in json.loads(response)
        gitlab_client_mock.apost.assert_called_once()


@pytest.mark.asyncio
async def test_create_diff_note_refuses_multi_line_no_op(
    gitlab_client_mock, metadata, mr_response, diffs_response_main_py_multiline
):
    """Multi-line suggestion whose every replacement line equals the target is refused."""
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response, diffs_response=diffs_response_main_py_multiline
    )
    gitlab_client_mock.apost = AsyncMock()
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    # The fixture's new file is line38..line45; suggestion replaces lines 41-43.
    with pytest.raises(ToolException, match="identical to the targeted line"):
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body="```suggestion:-1+1\nline41\nline42\nline43\n```",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=42,
        )
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_create_diff_note_fails_open_when_diffs_unavailable(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """When /diffs errors the guard fails open and the suggestion is posted."""
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response,
        diffs_response=GitLabHttpResponse(status_code=500, body="boom"),
    )
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body=f"```suggestion:-0+0\n{_TARGET_LINE}\n```",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=42,
    )
    assert "created_diff_note" in json.loads(response)
    gitlab_client_mock.apost.assert_called_once()


@pytest.mark.asyncio
async def test_create_diff_note_allows_when_any_block_is_meaningful(
    gitlab_client_mock,
    metadata,
    mr_response,
    diffs_response_main_py,
    discussion_response,
):
    """A comment with one no-op block plus one meaningful block is allowed."""
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response, diffs_response=diffs_response_main_py
    )
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    body = (
        f"```suggestion:-0+0\n{_TARGET_LINE}\n```\n"
        "```suggestion:-0+0\nthis.$emit('agent-loaded');\n```"
    )
    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body=body,
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=42,
    )
    assert "created_diff_note" in json.loads(response)
    gitlab_client_mock.apost.assert_called_once()


@pytest.mark.asyncio
async def test_create_diff_note_refuses_when_all_blocks_are_no_ops(
    gitlab_client_mock, metadata, mr_response, diffs_response_main_py_multiline
):
    """Multiple suggestion blocks that are all no-ops are refused."""
    gitlab_client_mock.aget = _aget_dispatch(
        mr_response=mr_response, diffs_response=diffs_response_main_py_multiline
    )
    gitlab_client_mock.apost = AsyncMock()
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    # First block is a single-line no-op at new_line=42; second is a 3-line no-op
    # spanning lines 41-43. Both target the same anchor (new_line=42).
    body = (
        "```suggestion:-0+0\nline42\n```\n"
        "```suggestion:-1+1\nline41\nline42\nline43\n```"
    )
    with pytest.raises(ToolException, match="identical to the targeted line"):
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body=body,
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=42,
        )
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_create_diff_note_paginates_diffs(
    gitlab_client_mock, metadata, mr_response, diffs_response_main_py
):
    """Guard finds the file diff even when it sits on a later /diffs page."""
    page_one = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            [
                {
                    "old_path": "other/file.py",
                    "new_path": "other/file.py",
                    "diff": "@@ -1,1 +1,1 @@\n line\n",
                }
            ]
        ),
        headers={"X-Next-Page": "2"},
    )

    async def _side_effect(path, **kwargs):
        if path.endswith("/diffs"):
            page = (kwargs.get("params") or {}).get("page", "1")
            return page_one if str(page) == "1" else diffs_response_main_py
        return mr_response

    gitlab_client_mock.aget = AsyncMock(side_effect=_side_effect)
    gitlab_client_mock.apost = AsyncMock()
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    with pytest.raises(ToolException, match="identical to the targeted line"):
        await tool._arun(
            project_id=1,
            merge_request_iid=9,
            body=f"```suggestion:-0+0\n{_TARGET_LINE}\n```",
            old_path="src/main.py",
            new_path="src/main.py",
            new_line=42,
        )
    gitlab_client_mock.apost.assert_not_called()
    # /diffs was fetched twice (page 1 then page 2).
    diffs_calls = [
        c
        for c in gitlab_client_mock.aget.call_args_list
        if c.kwargs["path"].endswith("/diffs")
    ]
    assert len(diffs_calls) == 2
