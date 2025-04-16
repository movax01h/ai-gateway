import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.merge_request import (
    CreateMergeRequest,
    CreateMergeRequestInput,
    CreateMergeRequestNote,
    CreateMergeRequestNoteInput,
    GetMergeRequest,
    GetMergeRequestInput,
    ListAllMergeRequestNotes,
    ListAllMergeRequestNotesInput,
    ListMergeRequestDiffs,
    ListMergeRequestDiffsInput,
    UpdateMergeRequest,
    UpdateMergeRequestInput,
)


@pytest.mark.asyncio
async def test_create_merge_request():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}'
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateMergeRequest(metadata=metadata)  # type: ignore

    input_data = {
        "project_id": 1,
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
        "description": "Feature description",
        "assignee_ids": [123],
        "reviewer_ids": [456],
        "remove_source_branch": True,
        "squash": True,
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
        "description": "Feature description",
        "assignee_ids": [123],
        "reviewer_ids": [456],
        "remove_source_branch": True,
        "squash": True,
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_data,
            "response": '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}',
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests", body=json.dumps(expected_data)
    )


@pytest.mark.asyncio
async def test_create_merge_request_minimal_params():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}'
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateMergeRequest(metadata=metadata)  # type: ignore

    input_data = {
        "project_id": 1,
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_data,
            "response": '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}',
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests", body=json.dumps(expected_data)
    )


@pytest.mark.asyncio
async def test_get_merge_request():

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = ["{}"]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetMergeRequest(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": "1", "merge_request_iid": "123"})

    assert response == "{}"

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123", parse_json=False
    )


@pytest.mark.asyncio
async def test_list_merge_request_diffs():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = ['{"diffs": []}']
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = ListMergeRequestDiffs(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": 1, "merge_request_iid": 123})

    assert response == '{"diffs": []}'

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/diffs", parse_json=False
    )


@pytest.mark.asyncio
async def test_create_merge_request_note():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.side_effect = ['{"id": 1, "body": "Test note"}']
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateMergeRequestNote(metadata=metadata)  # type: ignore

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": "Test note"}
    )

    expected_response = json.dumps(
        {
            "status": "success",
            "body": "Test note",
            "response": '{"id": 1, "body": "Test note"}',
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body='{"body": "Test note"}',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note",
    [
        "/merge",
        "/close",
        "/label ~bug",
        "/assign @user",
        "/milestone %v1.0",
        "/remove_source_branch",
        "/target_branch main",
        "/title Update title",
        "/board_move ~doing",
        "/copy_metadata from !123",
        "This is a multi-line note\n/merge",
        "Line 1\n/close\nLine 3",
        "/MErGE",
    ],
)
async def test_create_merge_request_note_blocks_quick_actions(note):
    gitlab_client_mock = AsyncMock()
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateMergeRequestNote(metadata=metadata)  # type: ignore

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": note}
    )

    expected_response = json.dumps(
        {
            "status": "error",
            "message": """Notes containing GitLab quick actions are not allowed. Quick actions are text-based shortcuts for common GitLab actions.
                                  They are commands that are on their own line and start with a backslash. Examples include /merge, /approve, /close, etc.""",
        }
    )

    assert response == expected_response
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note",
    [
        "This is a regular note",
        "This note talks about /merge in the middle",
        "https://gitlab.com",
        "gitlab-org/gitlab",
        "URL: https://example.com/merge",
        "Text with slash/merge in middle",
        "Line 1\nLine 2\nLine 3",
        "Discussion about\nmerge\nand \nclose\n without slashes",
    ],
)
async def test_create_merge_request_note_allows_regular_notes(note):
    gitlab_client_mock = AsyncMock()
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateMergeRequestNote(metadata=metadata)  # type: ignore

    gitlab_client_mock.apost.side_effect = ['{"id": 1, "body": "' + note + '"}']

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": note}
    )

    expected_response = json.dumps(
        {
            "status": "success",
            "body": note,
            "response": '{"id": 1, "body": "' + note + '"}',
        }
    )

    assert response == expected_response
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body=json.dumps({"body": note}),
    )


@pytest.mark.asyncio
async def test_list_all_merge_request_notes():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = [
        '[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'
    ]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = ListAllMergeRequestNotes(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": 1, "merge_request_iid": 123})

    assert response == '[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes", parse_json=False
    )


@pytest.mark.asyncio
async def test_update_merge_request():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aput.return_value = (
        '{"id": 123, "title": "Updated MR", "description": "New description"}'
    )
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }
    tool = UpdateMergeRequest(metadata=metadata)  # type: ignore

    response = await tool.arun(
        {
            "project_id": 1,
            "merge_request_iid": 123,
            "title": "Updated MR",
            "description": "New description",
            "remove_source_branch": True,
        }
    )

    expected_response = (
        '{"id": 123, "title": "Updated MR", "description": "New description"}'
    )
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123",
        body=json.dumps(
            {
                "description": "New description",
                "remove_source_branch": True,
                "title": "Updated MR",
            }
        ),
    )


def test_create_merge_request_format_display_message():
    tool = CreateMergeRequest(description="Create merge request")

    input_data = CreateMergeRequestInput(
        project_id=42,
        source_branch="feature-branch",
        target_branch="main",
        title="New feature implementation",
        description="This implements the new feature",
        assignee_ids=[123, 456],
        reviewer_ids=[789],
        remove_source_branch=True,
        squash=True,
    )

    message = tool.format_display_message(input_data)

    expected_message = (
        "Create merge request from 'feature-branch' to 'main' in project 42"
    )
    assert message == expected_message


def test_get_merge_request_format_display_message():
    tool = GetMergeRequest(description="Get merge request description")

    input_data = GetMergeRequestInput(project_id=42, merge_request_iid=123)

    message = tool.format_display_message(input_data)

    expected_message = "Read merge request !123 in project 42"
    assert message == expected_message


def test_list_merge_request_diffs_format_display_message():
    tool = ListMergeRequestDiffs(description="List merge request diffs")

    input_data = ListMergeRequestDiffsInput(project_id=42, merge_request_iid=123)

    message = tool.format_display_message(input_data)

    expected_message = "View changes in merge request !123 in project 42"
    assert message == expected_message


def test_create_merge_request_note_format_display_message():
    tool = CreateMergeRequestNote(description="Create merge request note")

    input_data = CreateMergeRequestNoteInput(
        project_id=42, merge_request_iid=123, body="This is a note on the merge request"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Add comment to merge request !123 in project 42"
    assert message == expected_message


def test_list_all_merge_request_notes_format_display_message():
    tool = ListAllMergeRequestNotes(description="List merge request notes")

    input_data = ListAllMergeRequestNotesInput(project_id=42, merge_request_iid=123)

    message = tool.format_display_message(input_data)

    expected_message = "Read comments on merge request !123 in project 42"
    assert message == expected_message


def test_update_merge_request_format_display_message():
    tool = UpdateMergeRequest(description="Update merge request")

    input_data = UpdateMergeRequestInput(
        project_id=42,
        merge_request_iid=123,
        title="Updated feature implementation",
        description="Updated description",
        allow_collaboration=True,
        assignee_ids=[123, 456],
        discussion_locked=True,
        milestone_id=10,
        remove_source_branch=True,
        reviewer_ids=[789],
        squash=True,
        state_event="close",
        target_branch="develop",
    )

    message = tool.format_display_message(input_data)

    expected_message = "Update merge request !123 in project 42"
    assert message == expected_message
