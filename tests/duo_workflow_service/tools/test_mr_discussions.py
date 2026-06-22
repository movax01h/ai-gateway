"""Tests for MR discussion tools: ListMrDiscussions, ReplyToDiscussion, SetDiscussionResolved."""

# pylint: disable=redefined-outer-name
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.mr_discussions import (
    NOTE_BODY_LIMIT,
    ListMrDiscussions,
    ListMrDiscussionsInput,
    ReplyToDiscussion,
    ReplyToDiscussionInput,
    SetDiscussionResolved,
    SetDiscussionResolvedInput,
    _truncate_note_body,
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


@pytest.fixture
def sample_discussions():
    """A list of discussions as returned by the GitLab API."""
    return [
        {
            "id": "disc_aaa111",
            "notes": [
                {
                    "id": 1001,
                    "author": {"username": "duo-security-reviewer"},
                    "body": "Security issue found in authentication",
                    "resolvable": True,
                    "resolved": False,
                    "position": {
                        "new_path": "app/auth.py",
                        "new_line": 42,
                        "old_line": 40,
                    },
                },
                {
                    "id": 1002,
                    "author": {"username": "developer"},
                    "body": "Thanks, will fix",
                },
            ],
        },
        {
            "id": "disc_bbb222",
            "notes": [
                {
                    "id": 2001,
                    "author": {"username": "developer"},
                    "body": "General comment on the MR",
                    "resolvable": False,
                    "resolved": False,
                },
            ],
        },
        {
            "id": "disc_ccc333",
            "notes": [
                {
                    "id": 3001,
                    "author": {"username": "duo-security-reviewer"},
                    "body": "Another finding resolved",
                    "resolvable": True,
                    "resolved": True,
                    "position": {
                        "new_path": "lib/utils.py",
                        "new_line": 10,
                        "old_line": None,
                    },
                },
            ],
        },
        {
            "id": "disc_empty",
            "notes": [],
        },
    ]


# --- Tests for ListMrDiscussions ---


class TestListMrDiscussions:
    @pytest.mark.asyncio
    async def test_list_all_discussions(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(project_id=123, merge_request_iid=45)

        parsed = json.loads(result)
        # 3 discussions returned (the empty one is skipped)
        assert len(parsed) == 3

        gitlab_client_mock.aget.assert_called_once_with(
            "/api/v4/projects/123/merge_requests/45/discussions",
            parse_json=False,
        )

    @pytest.mark.asyncio
    async def test_list_discussions_with_position(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(project_id=123, merge_request_iid=45)

        parsed = json.loads(result)
        first_disc = parsed[0]
        assert first_disc["discussion_id"] == "disc_aaa111"
        assert first_disc["file"] == "app/auth.py"
        assert first_disc["new_line"] == 42
        assert first_disc["old_line"] == 40
        assert first_disc["resolved"] is False
        assert first_disc["author"] == "duo-security-reviewer"
        assert len(first_disc["notes"]) == 2

    @pytest.mark.asyncio
    async def test_list_discussions_without_position(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(project_id=123, merge_request_iid=45)

        parsed = json.loads(result)
        # Second discussion has no position
        general_disc = parsed[1]
        assert general_disc["discussion_id"] == "disc_bbb222"
        assert "file" not in general_disc
        assert "new_line" not in general_disc

    @pytest.mark.asyncio
    async def test_filter_by_author(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            only_from_author="duo-security-reviewer",
        )

        parsed = json.loads(result)
        assert len(parsed) == 2
        assert all(d["author"] == "duo-security-reviewer" for d in parsed)

    @pytest.mark.asyncio
    async def test_filter_only_resolvable(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            only_resolvable=True,
        )

        parsed = json.loads(result)
        # Only disc_aaa111 and disc_ccc333 are resolvable
        assert len(parsed) == 2
        assert parsed[0]["discussion_id"] == "disc_aaa111"
        assert parsed[1]["discussion_id"] == "disc_ccc333"

    @pytest.mark.asyncio
    async def test_filter_author_and_resolvable(
        self, metadata, gitlab_client_mock, sample_discussions
    ):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(sample_discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            only_from_author="developer",
            only_resolvable=True,
        )

        parsed = json.loads(result)
        # developer has no resolvable discussions (disc_bbb222 is not resolvable)
        assert len(parsed) == 0

    @pytest.mark.asyncio
    async def test_empty_discussions(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps([]),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(project_id=123, merge_request_iid=45)

        parsed = json.loads(result)
        assert parsed == []

    @pytest.mark.asyncio
    async def test_note_body_truncated_with_marker(self, metadata, gitlab_client_mock):
        long_body = "x" * (NOTE_BODY_LIMIT + 1000)
        discussions = [
            {
                "id": "disc_long",
                "notes": [
                    {
                        "id": 9999,
                        "author": {"username": "user"},
                        "body": long_body,
                        "resolvable": False,
                        "resolved": False,
                    }
                ],
            }
        ]
        gitlab_client_mock.aget = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=200,
                body=json.dumps(discussions),
            )
        )

        tool = ListMrDiscussions(metadata=metadata)
        result = await tool._execute(project_id=123, merge_request_iid=45)

        parsed = json.loads(result)
        body = parsed[0]["notes"][0]["body"]
        # kept content is capped at the limit, and the drop is made explicit
        assert body.startswith("x" * NOTE_BODY_LIMIT)
        assert "TRUNCATED" in body
        assert "1000 CHARACTERS DROPPED" in body

    def test_format_display_message(self, metadata):
        tool = ListMrDiscussions(metadata=metadata)
        args = ListMrDiscussionsInput(project_id=123, merge_request_iid=45)
        message = tool.format_display_message(args)

        assert "!45" in message
        assert "123" in message


# --- Tests for ReplyToDiscussion ---


class TestReplyToDiscussion:
    @pytest.mark.asyncio
    async def test_reply_success(self, metadata, gitlab_client_mock):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=201,
                body=json.dumps({"id": 5001, "body": "Great, fixed!"}),
            )
        )

        tool = ReplyToDiscussion(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222ccc333",
            body="Great, fixed!",
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["note_id"] == 5001
        assert "disc_aaa111b" in parsed["message"]

        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        assert "/discussions/disc_aaa111bbb222ccc333/notes" in call_args.kwargs["path"]
        payload = json.loads(call_args.kwargs["body"])
        assert payload["body"] == "Great, fixed!"

    @pytest.mark.asyncio
    async def test_reply_fixes_newlines(self, metadata, gitlab_client_mock):
        gitlab_client_mock.apost = AsyncMock(
            return_value=GitLabHttpResponse(
                status_code=201,
                body=json.dumps({"id": 5002}),
            )
        )

        tool = ReplyToDiscussion(metadata=metadata)
        await tool._execute(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_123",
            body="line1\\nline2\\nline3",
        )

        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert payload["body"] == "line1\nline2\nline3"

    def test_fix_newlines_static(self):
        assert ReplyToDiscussion._fix_newlines("a\\nb") == "a\nb"
        assert ReplyToDiscussion._fix_newlines("no change") == "no change"
        assert ReplyToDiscussion._fix_newlines("") == ""

    def test_format_display_message(self, metadata):
        tool = ReplyToDiscussion(metadata=metadata)
        args = ReplyToDiscussionInput(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222",
            body="reply text",
        )
        message = tool.format_display_message(args)

        assert "!45" in message
        assert "disc_aaa111b" in message

    def test_format_display_message_short_id(self, metadata):
        tool = ReplyToDiscussion(metadata=metadata)
        args = ReplyToDiscussionInput(
            project_id=123,
            merge_request_iid=45,
            discussion_id="short",
            body="reply",
        )
        message = tool.format_display_message(args)
        assert "short" in message


# --- Tests for SetDiscussionResolved ---


class TestSetDiscussionResolved:
    @pytest.mark.asyncio
    async def test_resolve_success(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aput = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SetDiscussionResolved(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222ccc333",
            resolved=True,
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "resolved" in parsed["message"]

        gitlab_client_mock.aput.assert_called_once()
        call_args = gitlab_client_mock.aput.call_args
        assert "/discussions/disc_aaa111bbb222ccc333" in call_args.kwargs["path"]
        payload = json.loads(call_args.kwargs["body"])
        assert payload["resolved"] is True

    @pytest.mark.asyncio
    async def test_unresolve_success(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aput = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SetDiscussionResolved(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222ccc333",
            resolved=False,
        )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "unresolve" in parsed["message"]

    @pytest.mark.asyncio
    async def test_resolve_default_is_true(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aput = AsyncMock(
            return_value=GitLabHttpResponse(status_code=200, body="{}")
        )

        tool = SetDiscussionResolved(metadata=metadata)
        result = await tool._execute(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_123",
        )

        parsed = json.loads(result)
        assert "resolved" in parsed["message"]

        call_args = gitlab_client_mock.aput.call_args
        payload = json.loads(call_args.kwargs["body"])
        assert payload["resolved"] is True

    def test_format_display_message_resolve(self, metadata):
        tool = SetDiscussionResolved(metadata=metadata)
        args = SetDiscussionResolvedInput(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222",
            resolved=True,
        )
        message = tool.format_display_message(args)

        assert "Resolve" in message
        assert "!45" in message
        assert "disc_aaa111b" in message

    def test_format_display_message_unresolve(self, metadata):
        tool = SetDiscussionResolved(metadata=metadata)
        args = SetDiscussionResolvedInput(
            project_id=123,
            merge_request_iid=45,
            discussion_id="disc_aaa111bbb222",
            resolved=False,
        )
        message = tool.format_display_message(args)

        assert "Unresolve" in message
        assert "!45" in message

    @pytest.mark.asyncio
    async def test_resolve_raises_on_error(self, metadata, gitlab_client_mock):
        """A 4xx (e.g. no permission to resolve) must surface, not report a false success."""
        gitlab_client_mock.aput = AsyncMock(
            return_value=GitLabHttpResponse(status_code=403, body="Forbidden")
        )

        tool = SetDiscussionResolved(metadata=metadata)
        with pytest.raises(ToolException):
            await tool._execute(
                project_id=123,
                merge_request_iid=45,
                discussion_id="disc_aaa111bbb222ccc333",
                resolved=True,
            )


class TestMrDiscussionTrustLevels:
    """These tools read/write remote MR data, not local fs/git, so they must not be TRUSTED_INTERNAL — their output must
    go through prompt-injection scanning."""

    @pytest.mark.parametrize(
        "tool_cls", [ListMrDiscussions, ReplyToDiscussion, SetDiscussionResolved]
    )
    def test_defaults_to_untrusted_user_content(self, metadata, tool_cls):
        tool = tool_cls(metadata=metadata)
        assert tool.trust_level == ToolTrustLevel.UNTRUSTED_USER_CONTENT


class TestTruncateNoteBody:
    def test_short_body_unchanged(self):
        assert _truncate_note_body("short") == "short"
        at_limit = "x" * NOTE_BODY_LIMIT
        assert _truncate_note_body(at_limit) == at_limit

    def test_long_body_gets_marker(self):
        result = _truncate_note_body("x" * (NOTE_BODY_LIMIT + 250))
        assert result.startswith("x" * NOTE_BODY_LIMIT)
        assert "250 CHARACTERS DROPPED" in result
