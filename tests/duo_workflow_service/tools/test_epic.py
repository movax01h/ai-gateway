import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.epic import (
    CreateEpic,
    GetEpic,
    GetEpicInput,
    ListEpics,
    ListEpicsInput,
    UpdateEpic,
    UpdateEpicInput,
    WriteEpicInput,
)


@pytest.mark.asyncio
async def test_get_epic():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.return_value = {
        "id": 1,
        "iid": 5,
        "group_id": 1,
        "title": "Test Epic",
        "description": "This is a test epic",
        "state": "opened",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
    }
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id=1, epic_iid=5)

    expected_response = json.dumps(
        {
            "epic": {
                "id": 1,
                "iid": 5,
                "group_id": 1,
                "title": "Test Epic",
                "description": "This is a test epic",
                "state": "opened",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/1/epics/5", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_epic_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = Exception("Epic not found")
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id=1, epic_iid=999)

    expected_response = json.dumps({"error": "Epic not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/1/epics/999", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_epic_with_string_group_id():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.return_value = {
        "id": 1,
        "iid": 5,
        "group_id": 1,
        "title": "Test Epic",
        "description": "This is a test epic",
        "state": "opened",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
    }
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id="group%2Fsubgroup", epic_iid=5)

    expected_response = json.dumps(
        {
            "epic": {
                "id": 1,
                "iid": 5,
                "group_id": 1,
                "title": "Test Epic",
                "description": "This is a test epic",
                "state": "opened",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/group%2Fsubgroup/epics/5", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_epic_with_string_group_id_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = Exception("Group not found")
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id="nonexistent%2Fgroup", epic_iid=5)

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/nonexistent%2Fgroup/epics/5", parse_json=False
    )


def test_get_epic_format_display_message():
    tool = GetEpic(description="Get epic description")

    input_data = GetEpicInput(group_id=123, epic_iid=5)

    message = tool.format_display_message(input_data)

    expected_message = "Read epic #5 in group 123"
    assert message == expected_message


@pytest.mark.asyncio
async def test_list_epics_success():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.return_value = [
        {
            "id": 1,
            "iid": 5,
            "group_id": 1,
            "title": "Test Epic 1",
            "description": "This is test epic 1",
            "state": "opened",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        },
        {
            "id": 2,
            "iid": 6,
            "group_id": 1,
            "title": "Test Epic 2",
            "description": "This is test epic 2",
            "state": "opened",
            "created_at": "2024-01-02T12:00:00Z",
            "updated_at": "2024-01-02T12:00:00Z",
        },
    ]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(
        group_id=1,
        author_id=123,
        labels="bug,feature",
        state="opened",
        search="test",
        sort="asc",
        with_labels_details=True,
        include_ancestor_groups=True,
        include_descendant_groups=True,
        my_reaction_emoji="thumbsup",
    )

    expected_response = json.dumps(
        {
            "epics": [
                {
                    "id": 1,
                    "iid": 5,
                    "group_id": 1,
                    "title": "Test Epic 1",
                    "description": "This is test epic 1",
                    "state": "opened",
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:00:00Z",
                },
                {
                    "id": 2,
                    "iid": 6,
                    "group_id": 1,
                    "title": "Test Epic 2",
                    "description": "This is test epic 2",
                    "state": "opened",
                    "created_at": "2024-01-02T12:00:00Z",
                    "updated_at": "2024-01-02T12:00:00Z",
                },
            ]
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/1/epics",
        params={
            "author_id": 123,
            "labels": "bug,feature",
            "state": "opened",
            "search": "test",
            "sort": "asc",
            "with_labels_details": True,
            "include_ancestor_groups": True,
            "include_descendant_groups": True,
            "my_reaction_emoji": "thumbsup",
        },
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_list_epics_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = Exception("Group not found")
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(group_id=999)

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/999/epics",
        params={},
        parse_json=False,
    )


def test_list_epic_format_display_message():
    tool = ListEpics(description="list epics description")

    input_data = ListEpicsInput(
        group_id=123,
        author_id=123,
        labels="bug,feature",
        state="opened",
        search="test",
        sort="asc",
        with_labels_details=True,
        include_ancestor_groups=True,
        include_descendant_groups=True,
        my_reaction_emoji="thumbsup",
        author_username=None,
        order_by=None,
        created_after=None,
        created_before=None,
        updated_before=None,
        updated_after=None,
        negate=None,
    )

    message = tool.format_display_message(input_data)

    expected_message = "List epics in group 123"
    assert message == expected_message


@pytest.mark.asyncio
async def test_create_epic_success():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = {
        "id": 1,
        "iid": 5,
        "group_id": 1,
        "title": "New Test Epic",
        "description": "This is a new test epic",
        "state": "opened",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
        "labels": ["bug", "feature"],
        "start_date_fixed": "2024-01-01",
        "due_date_fixed": "2024-12-31",
        "confidential": True,
        "parent_id": 10,
    }
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateEpic(description="create epic description", metadata=metadata)

    response = await tool._arun(
        group_id=1,
        title="New Test Epic",
        description="This is a new test epic",
        labels="bug,feature",
        start_date_fixed="2024-01-01",
        due_date_fixed="2024-12-31",
        confidential=True,
        parent_id=10,
        start_date_is_fixed=True,
        due_date_is_fixed=True,
    )

    expected_response = json.dumps(
        {"created_epic": gitlab_client_mock.apost.return_value}
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/groups/1/epics",
        body=json.dumps(
            {
                "title": "New Test Epic",
                "description": "This is a new test epic",
                "labels": "bug,feature",
                "start_date_fixed": "2024-01-01",
                "due_date_fixed": "2024-12-31",
                "confidential": True,
                "parent_id": 10,
                "start_date_is_fixed": True,
                "due_date_is_fixed": True,
            }
        ),
    )


@pytest.mark.asyncio
async def test_create_epic_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.side_effect = Exception("Group not found")
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = CreateEpic(description="create epic description", metadata=metadata)

    response = await tool._arun(group_id=999, title="Test Epic")

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/groups/999/epics",
        body=json.dumps({"title": "Test Epic"}),
    )


def test_create_epic_format_display_message():
    tool = CreateEpic(description="list epics description")

    input_data = WriteEpicInput(
        group_id=123,
        title="New epic",
        description="epic description",
        labels=None,
        confidential=None,
        start_date_fixed=None,
        start_date_is_fixed=None,
        due_date_fixed=None,
        due_date_is_fixed=None,
        parent_id=None,
    )

    message = tool.format_display_message(input_data)

    expected_message = "Create epic 'New epic' in group 123"
    assert message == expected_message


@pytest.mark.asyncio
async def test_update_epic_success():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aput.return_value = {
        "id": 1,
        "iid": 5,
        "group_id": 1,
        "title": "Updated Test Epic",
        "description": "This is an updated test epic",
        "state": "closed",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-02T12:00:00Z",
        "labels": ["bug", "urgent"],
        "start_date_fixed": "2024-02-01",
        "due_date_fixed": "2024-12-31",
        "confidential": True,
        "parent_id": 10,
    }
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = UpdateEpic(description="update epic description", metadata=metadata)

    response = await tool._arun(
        group_id=1,
        epic_iid=5,
        title="Updated Test Epic",
        description="This is an updated test epic",
        labels="bug,urgent",
        start_date_fixed="2024-02-01",
        due_date_fixed="2024-12-31",
        state_event="close",
        confidential=True,
        parent_id=10,
        start_date_is_fixed=True,
        due_date_is_fixed=True,
    )

    expected_response = json.dumps(
        {"updated_epic": gitlab_client_mock.aput.return_value}
    )
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/groups/1/epics/5",
        body=json.dumps(
            {
                "title": "Updated Test Epic",
                "description": "This is an updated test epic",
                "labels": "bug,urgent",
                "start_date_fixed": "2024-02-01",
                "due_date_fixed": "2024-12-31",
                "state_event": "close",
                "confidential": True,
                "parent_id": 10,
                "start_date_is_fixed": True,
                "due_date_is_fixed": True,
            }
        ),
    )


@pytest.mark.asyncio
async def test_update_epic_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aput.side_effect = Exception("Epic not found")
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = UpdateEpic(description="update epic description", metadata=metadata)

    response = await tool._arun(group_id=1, epic_iid=999, title="Updated Epic")

    expected_response = json.dumps({"error": "Epic not found"})
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/groups/1/epics/999", body=json.dumps({"title": "Updated Epic"})
    )


def test_update_epic_format_display_message():
    tool = UpdateEpic(description="list epics description")

    input_data = UpdateEpicInput(
        epic_iid=456,
        add_labels=None,
        remove_labels=None,
        state_event=None,
        group_id=123,
        title="New epic",
        description="epic description",
        labels=None,
        confidential=None,
        start_date_fixed=None,
        start_date_is_fixed=None,
        due_date_fixed=None,
        due_date_is_fixed=None,
        parent_id=None,
    )

    message = tool.format_display_message(input_data)

    expected_message = "Update epic #456"
    assert message == expected_message
