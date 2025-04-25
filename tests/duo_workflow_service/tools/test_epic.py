import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.epic import (
    CreateEpic,
    EpicResourceInput,
    GetEpic,
    GroupResourceInput,
    ListEpics,
    ListEpicsInput,
    UpdateEpic,
    UpdateEpicInput,
    WriteEpicInput,
)


@pytest.fixture
def gitlab_client_mock():
    mock = Mock()
    return mock


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.mark.asyncio
async def test_get_epic(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value={
            "id": 1,
            "iid": 5,
            "group_id": 1,
            "title": "Test Epic",
            "description": "This is a test epic",
            "state": "opened",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        }
    )

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
async def test_get_epic_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Epic not found"))

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id=1, epic_iid=999)

    expected_response = json.dumps({"error": "Epic not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/1/epics/999", parse_json=False
    )


@pytest.mark.asyncio
async def test_get_epic_with_string_group_id(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value={
            "id": 1,
            "iid": 5,
            "group_id": 1,
            "title": "Test Epic",
            "description": "This is a test epic",
            "state": "opened",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        }
    )

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
async def test_get_epic_with_string_group_id_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Group not found"))

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(group_id="nonexistent%2Fgroup", epic_iid=5)

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/nonexistent%2Fgroup/epics/5", parse_json=False
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            EpicResourceInput(group_id=123, epic_iid=5),
            "Read epic #5 in group 123",
        ),
        (
            EpicResourceInput(
                url="https://gitlab.com/groups/namespace/group/-/epics/42"
            ),
            "Read epic https://gitlab.com/groups/namespace/group/-/epics/42",
        ),
    ],
)
def test_get_epic_format_display_message(input_data, expected_message):
    tool = GetEpic(description="Get epic description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,group_id,epic_iid,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            None,
            None,
            "/api/v4/groups/namespace%2Fgroup/epics/123",
        ),
        # Test with URL and matching group_id and epic_iid
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "namespace%2Fgroup",
            123,
            "/api/v4/groups/namespace%2Fgroup/epics/123",
        ),
    ],
)
async def test_get_epic_with_url_success(
    url, group_id, epic_iid, expected_path, gitlab_client_mock, metadata
):
    mock_response = {
        "id": 1,
        "iid": 123,
        "group_id": "namespace%2Fgroup",
        "title": "Test Epic",
        "description": "This is a test epic",
        "state": "opened",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
    }
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(url=url, group_id=group_id, epic_iid=epic_iid)

    expected_response = json.dumps({"epic": mock_response})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
async def test_validate_group_url_no_url_no_group_id(metadata):
    tool = CreateEpic(description="create epic description", metadata=metadata)

    validation_result = tool._validate_group_url(url=None, group_id=None)

    assert validation_result.group_id is None
    assert len(validation_result.errors) == 1
    assert (
        validation_result.errors[0] == "'group_id' must be provided when 'url' is not"
    )


@pytest.mark.asyncio
async def test_validate_epic_url_no_url_no_ids(metadata):
    tool = GetEpic(description="get epic description", metadata=metadata)

    validation_result = tool._validate_epic_url(url=None, group_id=None, epic_iid=None)

    assert validation_result.group_id is None
    assert validation_result.epic_iid is None
    assert len(validation_result.errors) == 2
    assert "'group_id' must be provided when 'url' is not" in validation_result.errors
    assert "'epic_iid' must be provided when 'url' is not" in validation_result.errors


@pytest.mark.asyncio
async def test_validate_epic_url_no_url_no_group_id(metadata):
    tool = GetEpic(description="get epic description", metadata=metadata)

    validation_result = tool._validate_epic_url(url=None, group_id=None, epic_iid=42)

    assert validation_result.group_id is None
    assert validation_result.epic_iid == 42
    assert len(validation_result.errors) == 1
    assert (
        validation_result.errors[0] == "'group_id' must be provided when 'url' is not"
    )


@pytest.mark.asyncio
async def test_validate_epic_url_no_url_no_epic_iid(metadata):
    tool = GetEpic(description="get epic description", metadata=metadata)

    validation_result = tool._validate_epic_url(
        url=None, group_id="group/subgroup", epic_iid=None
    )

    assert validation_result.group_id == "group/subgroup"
    assert validation_result.epic_iid is None
    assert len(validation_result.errors) == 1
    assert (
        validation_result.errors[0] == "'epic_iid' must be provided when 'url' is not"
    )


@pytest.mark.asyncio
async def test_list_epics_with_negate_parameter(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value=[
            {
                "id": 1,
                "iid": 5,
                "group_id": 1,
                "title": "Test Epic 1",
                "description": "This is test epic 1",
                "state": "opened",
            },
        ]
    )
    negate_params = {"author_id": 456, "labels": "wontfix"}

    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(group_id=1, negate=negate_params)

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
                },
            ]
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/1/epics",
        params={"not": negate_params},
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,group_id,epic_iid,error_contains",
    [
        # URL and group_id both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "different%2Fgroup",
            123,
            "Group ID mismatch",
        ),
        # URL and epic_iid both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "namespace%2Fgroup",
            456,
            "Epic ID mismatch",
        ),
        # URL given isn't an epic URL (it's just a group URL)
        (
            "https://gitlab.com/groups/namespace/group",
            None,
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_get_epic_with_url_error(
    url, group_id, epic_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetEpic(description="get epic description", metadata=metadata)

    response = await tool._arun(url=url, group_id=group_id, epic_iid=epic_iid)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_epics_success(gitlab_client_mock, metadata):
    sample_epics = [
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
    gitlab_client_mock.aget = AsyncMock(return_value=sample_epics)

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

    expected_response = json.dumps({"epics": sample_epics})
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
async def test_list_epics_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Group not found"))

    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(group_id=999)

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/999/epics",
        params={},
        parse_json=False,
    )


OPTIONAL_LIST_EPICS_INPUTS = {
    "author_id": 123,
    "labels": "bug,feature",
    "state": "opened",
    "search": "test",
    "sort": "asc",
    "with_labels_details": True,
    "include_ancestor_groups": True,
    "include_descendant_groups": True,
    "my_reaction_emoji": "thumbsup",
    "author_username": None,
    "order_by": None,
    "created_after": None,
    "created_before": None,
    "updated_before": None,
    "updated_after": None,
    "negate": None,
}


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListEpicsInput(group_id=123, **OPTIONAL_LIST_EPICS_INPUTS),  # type: ignore
            "List epics in group 123",
        ),
        (
            ListEpicsInput(
                url="https://gitlab.com/groups/namespace/group", **OPTIONAL_LIST_EPICS_INPUTS  # type: ignore
            ),
            "List epics in https://gitlab.com/groups/namespace/group",
        ),
    ],
)
def test_list_epic_format_display_message(input_data, expected_message):
    tool = ListEpics(description="list epics description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,group_id,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/groups/namespace/group",
            None,
            "/api/v4/groups/namespace%2Fgroup/epics",
        ),
        # Test with URL and matching group_id
        (
            "https://gitlab.com/groups/namespace/group",
            "namespace%2Fgroup",
            "/api/v4/groups/namespace%2Fgroup/epics",
        ),
    ],
)
async def test_list_epics_with_url_success(
    url, group_id, expected_path, gitlab_client_mock, metadata
):
    mock_response = [
        {
            "id": 1,
            "iid": 5,
            "group_id": "namespace%2Fgroup",
            "title": "Test Epic 1",
            "description": "This is test epic 1",
            "state": "opened",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        },
        {
            "id": 2,
            "iid": 6,
            "group_id": "namespace%2Fgroup",
            "title": "Test Epic 2",
            "description": "This is test epic 2",
            "state": "opened",
            "created_at": "2024-01-02T12:00:00Z",
            "updated_at": "2024-01-02T12:00:00Z",
        },
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(
        url=url,
        group_id=group_id,
        labels="bug,feature",
        state="opened",
        search="test",
        sort="asc",
        with_labels_details=True,
        include_ancestor_groups=True,
        include_descendant_groups=True,
        my_reaction_emoji="thumbsup",
    )

    expected_response = json.dumps({"epics": mock_response})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={
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
@pytest.mark.parametrize(
    "url,group_id,error_contains",
    [
        # URL and group_id both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group",
            "different%2Fgroup",
            "Group ID mismatch",
        ),
        # Invalid URL
        (
            "gitlab.com/namespace",
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_list_epics_with_url_error(
    url, group_id, error_contains, gitlab_client_mock, metadata
):
    tool = ListEpics(description="list epics description", metadata=metadata)

    response = await tool._arun(url=url, group_id=group_id)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_create_epic_success(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
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
    )

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
async def test_create_epic_error(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Group not found"))

    tool = CreateEpic(description="create epic description", metadata=metadata)

    response = await tool._arun(group_id=999, title="Test Epic")

    expected_response = json.dumps({"error": "Group not found"})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/groups/999/epics",
        body=json.dumps({"title": "Test Epic"}),
    )


OPTIONAL_WRITE_EPIC_INPUTS = {
    "title": "New epic",
    "description": "epic description",
    "labels": None,
    "confidential": None,
    "start_date_fixed": None,
    "start_date_is_fixed": None,
    "due_date_fixed": None,
    "due_date_is_fixed": None,
    "parent_id": None,
}


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            WriteEpicInput(
                group_id=123,
                **OPTIONAL_WRITE_EPIC_INPUTS,  # type: ignore
            ),
            "Create epic 'New epic' in group 123",
        ),
        (
            WriteEpicInput(
                url="https://gitlab.com/groups/namespace/group",
                **OPTIONAL_WRITE_EPIC_INPUTS,  # type: ignore
            ),
            "Create epic 'New epic' in https://gitlab.com/groups/namespace/group",
        ),
    ],
)
def test_create_epic_format_display_message(input_data, expected_message):
    tool = CreateEpic(description="create epic description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,group_id,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/groups/namespace/group",
            None,
            "/api/v4/groups/namespace%2Fgroup/epics",
        ),
        # Test with URL and matching group_id
        (
            "https://gitlab.com/groups/namespace/group",
            "namespace%2Fgroup",
            "/api/v4/groups/namespace%2Fgroup/epics",
        ),
    ],
)
async def test_create_epic_with_url_success(
    url, group_id, expected_path, gitlab_client_mock, metadata
):
    mock_response = {
        "id": 1,
        "iid": 5,
        "group_id": "namespace%2Fgroup",
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
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateEpic(description="create epic description", metadata=metadata)

    response = await tool._arun(
        url=url,
        group_id=group_id,
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

    expected_response = json.dumps({"created_epic": mock_response})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
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
@pytest.mark.parametrize(
    "url,group_id,error_contains",
    [
        # URL and group_id both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group",
            "different%2Fgroup",
            "Group ID mismatch",
        ),
        # Invalid URL
        (
            "gitlab.com/namespace",
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_create_epic_with_url_error(
    url, group_id, error_contains, gitlab_client_mock, metadata
):
    tool = CreateEpic(description="create epic description", metadata=metadata)

    response = await tool._arun(
        url=url,
        group_id=group_id,
        title="Test Epic",
    )
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_update_epic_success(gitlab_client_mock, metadata):
    gitlab_client_mock.aput = AsyncMock(
        return_value={
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
    )

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
async def test_update_epic_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aput = AsyncMock(side_effect=Exception("Epic not found"))

    tool = UpdateEpic(description="update epic description", metadata=metadata)

    response = await tool._arun(group_id=1, epic_iid=999, title="Updated Epic")

    expected_response = json.dumps({"error": "Epic not found"})
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/groups/1/epics/999", body=json.dumps({"title": "Updated Epic"})
    )


OPTIONAL_UPDATE_EPIC_INPUTS = {
    "title": "Updated epic",
    "description": "epic description",
    "labels": None,
    "confidential": None,
    "start_date_fixed": None,
    "start_date_is_fixed": None,
    "due_date_fixed": None,
    "due_date_is_fixed": None,
    "parent_id": None,
    "add_labels": None,
    "remove_labels": None,
    "state_event": None,
}


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            UpdateEpicInput(
                group_id=123,
                epic_iid=456,
                **OPTIONAL_UPDATE_EPIC_INPUTS,  # type: ignore
            ),
            "Update epic #456",
        ),
        (
            UpdateEpicInput(
                url="https://gitlab.com/groups/namespace/group/-/epics/42",
                **OPTIONAL_UPDATE_EPIC_INPUTS,  # type: ignore
            ),
            "Update epic https://gitlab.com/groups/namespace/group/-/epics/42",
        ),
    ],
)
def test_update_epic_format_display_message(input_data, expected_message):
    tool = UpdateEpic(description="update epic description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,group_id,epic_iid,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            None,
            None,
            "/api/v4/groups/namespace%2Fgroup/epics/123",
        ),
        # Test with URL and matching group_id and epic_iid
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "namespace%2Fgroup",
            123,
            "/api/v4/groups/namespace%2Fgroup/epics/123",
        ),
    ],
)
async def test_update_epic_with_url_success(
    url, group_id, epic_iid, expected_path, gitlab_client_mock, metadata
):
    mock_response = {
        "id": 1,
        "iid": 123,
        "group_id": "namespace%2Fgroup",
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
    gitlab_client_mock.aput = AsyncMock(return_value=mock_response)

    tool = UpdateEpic(description="update epic description", metadata=metadata)

    response = await tool._arun(
        url=url,
        group_id=group_id,
        epic_iid=epic_iid,
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

    expected_response = json.dumps({"updated_epic": mock_response})
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path=expected_path,
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
@pytest.mark.parametrize(
    "url,group_id,epic_iid,error_contains",
    [
        # URL and group_id both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "different%2Fgroup",
            123,
            "Group ID mismatch",
        ),
        # URL and epic_iid both given, but don't match
        (
            "https://gitlab.com/groups/namespace/group/-/epics/123",
            "namespace%2Fgroup",
            456,
            "Epic ID mismatch",
        ),
        # URL given isn't an epic URL (it's just a group URL)
        (
            "https://gitlab.com/groups/namespace/group",
            None,
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_update_epic_with_url_error(
    url, group_id, epic_iid, error_contains, gitlab_client_mock, metadata
):
    tool = UpdateEpic(description="update epic description", metadata=metadata)

    response = await tool._arun(
        url=url,
        group_id=group_id,
        epic_iid=epic_iid,
        title="Updated Epic",
    )
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aput.assert_not_called()
