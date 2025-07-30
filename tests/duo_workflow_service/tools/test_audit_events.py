import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.audit_events import (
    ListGroupAuditEvents,
    ListGroupAuditEventsInput,
    ListInstanceAuditEvents,
    ListInstanceAuditEventsInput,
    ListProjectAuditEvents,
    ListProjectAuditEventsInput,
)


@pytest.fixture(name="instance_audit_event_data")
def instance_audit_event_data_fixture():
    """Fixture for instance audit event data."""
    return [
        {
            "id": 1,
            "author_id": 1,
            "entity_id": 6,
            "entity_type": "Project",
            "details": {
                "custom_message": "Project archived",
                "author_name": "Administrator",
                "author_email": "admin@example.com",
                "target_id": "flightjs/flight",
                "target_type": "Project",
                "target_details": "flightjs/flight",
                "ip_address": "127.0.0.1",
                "entity_path": "flightjs/flight",
            },
            "created_at": "2019-08-30T07:00:41.885Z",
        }
    ]


@pytest.fixture(name="group_audit_event_data")
def group_audit_event_data_fixture():
    """Fixture for group audit event data."""
    return [
        {
            "id": 2,
            "author_id": 1,
            "entity_id": 60,
            "entity_type": "Group",
            "details": {
                "custom_message": "Group marked for deletion",
                "author_name": "Administrator",
                "author_email": "admin@example.com",
                "target_id": "flightjs",
                "target_type": "Group",
                "target_details": "flightjs",
                "ip_address": "127.0.0.1",
                "entity_path": "flightjs",
            },
            "created_at": "2019-08-28T19:36:44.162Z",
        }
    ]


@pytest.fixture(name="project_audit_event_data")
def project_audit_event_data_fixture():
    """Fixture for project audit event data."""
    return [
        {
            "id": 5,
            "author_id": 1,
            "entity_id": 7,
            "entity_type": "Project",
            "details": {
                "change": "prevent merge request approval from committers",
                "from": "",
                "to": "true",
                "author_name": "Administrator",
                "author_email": "admin@example.com",
                "target_id": 7,
                "target_type": "Project",
                "target_details": "twitter/typeahead-js",
                "ip_address": "127.0.0.1",
                "entity_path": "twitter/typeahead-js",
            },
            "created_at": "2020-05-26T22:55:04.230Z",
        }
    ]


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


# Instance Audit Events Tests


@pytest.mark.asyncio
async def test_list_instance_audit_events(
    gitlab_client_mock, metadata, instance_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=instance_audit_event_data)
    # Mock last_response.headers.get to return a valid integer string
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListInstanceAuditEvents(metadata=metadata)

    input_data = {}

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "audit_events": instance_audit_event_data,
            "pagination": {
                "total_items": len(instance_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/audit_events",
        params={"per_page": 20, "page": 1},
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_instance_audit_events_with_filters(
    gitlab_client_mock, metadata, instance_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=instance_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListInstanceAuditEvents(metadata=metadata)

    input_data = {
        "entity_type": "Project",
        "entity_id": 6,
        "created_after": "2019-01-01T00:00:00Z",
        "created_before": "2019-12-31T23:59:59Z",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "audit_events": instance_audit_event_data,
            "pagination": {
                "total_items": len(instance_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/audit_events",
        params={
            "entity_type": "Project",
            "entity_id": 6,
            "created_after": "2019-01-01T00:00:00Z",
            "created_before": "2019-12-31T23:59:59Z",
            "per_page": 20,
            "page": 1,
        },
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_instance_audit_events_entity_id_without_type(
    gitlab_client_mock, metadata
):
    tool = ListInstanceAuditEvents(metadata=metadata)

    response = await tool.arun({"entity_id": 6})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "entity_id requires entity_type to be specified" in error_response["error"]


@pytest.mark.asyncio
async def test_list_instance_audit_events_pagination(
    gitlab_client_mock, metadata, instance_audit_event_data
):
    # Test fetching all pages
    page1_data = instance_audit_event_data
    page2_data = [
        {
            "id": 2,
            "author_id": 1,
            "entity_id": 60,
            "entity_type": "Group",
            "details": {},
            "created_at": "2019-08-27T18:36:44.162Z",
        }
    ]

    gitlab_client_mock.aget = AsyncMock(side_effect=[page1_data, page2_data])
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "2"}

    tool = ListInstanceAuditEvents(metadata=metadata)

    response = await tool.arun({"fetch_all_pages": True, "per_page": 1})

    result = json.loads(response)
    assert len(result["audit_events"]) == 2
    assert result["pagination"]["total_items"] == 2
    assert result["pagination"]["total_pages"] == 2


@pytest.mark.asyncio
async def test_list_instance_audit_events_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("API Error"))

    tool = ListInstanceAuditEvents(metadata=metadata)

    with pytest.raises(Exception, match="API Error"):
        await tool.arun({})


@pytest.mark.asyncio
async def test_list_instance_audit_events_api_error_response(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(return_value={"message": "401 Unauthorized"})

    tool = ListInstanceAuditEvents(metadata=metadata)

    response = await tool.arun({})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "401 Unauthorized" in error_response["error"]


@pytest.mark.asyncio
async def test_list_instance_audit_events_api_error_key_response(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(return_value={"error": "Access denied"})

    tool = ListInstanceAuditEvents(metadata=metadata)

    response = await tool.arun({})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Access denied" in error_response["error"]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListInstanceAuditEventsInput(),
            "List instance audit events",
        ),
        (
            ListInstanceAuditEventsInput(entity_type="Project", entity_id=6),
            "List instance audit events for Project 6",
        ),
    ],
)
def test_list_instance_audit_events_format_display_message(
    input_data, expected_message
):
    tool = ListInstanceAuditEvents(metadata={})
    assert tool.format_display_message(input_data) == expected_message


# Group Audit Events Tests


@pytest.mark.asyncio
async def test_list_group_audit_events_with_id(
    gitlab_client_mock, metadata, group_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=group_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListGroupAuditEvents(metadata=metadata)

    input_data = {"group_id": 60}

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "audit_events": group_audit_event_data,
            "pagination": {
                "total_items": len(group_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/60/audit_events",
        params={"per_page": 20, "page": 1},
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_group_audit_events_with_path(
    gitlab_client_mock, metadata, group_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=group_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListGroupAuditEvents(metadata=metadata)

    input_data = {"group_path": "gitlab-org/gitlab"}

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "audit_events": group_audit_event_data,
            "pagination": {
                "total_items": len(group_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/gitlab-org/gitlab/audit_events",
        params={"per_page": 20, "page": 1},
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_group_audit_events_no_identifier(gitlab_client_mock, metadata):
    tool = ListGroupAuditEvents(metadata=metadata)

    response = await tool.arun({})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Either group_id or group_path must be provided" in error_response["error"]


@pytest.mark.asyncio
async def test_list_group_audit_events_with_filters(
    gitlab_client_mock, metadata, group_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=group_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListGroupAuditEvents(metadata=metadata)

    input_data = {
        "group_id": 60,
        "created_after": "2019-08-01T00:00:00Z",
        "created_before": "2019-08-31T23:59:59Z",
    }

    response = await tool.arun(input_data)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/groups/60/audit_events",
        params={
            "created_after": "2019-08-01T00:00:00Z",
            "created_before": "2019-08-31T23:59:59Z",
            "per_page": 20,
            "page": 1,
        },
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_group_audit_events_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("API Error"))

    tool = ListGroupAuditEvents(metadata=metadata)

    with pytest.raises(Exception, match="API Error"):
        await tool.arun({"group_id": 60})


@pytest.mark.asyncio
async def test_list_group_audit_events_api_error_response(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(return_value={"message": "403 Forbidden"})

    tool = ListGroupAuditEvents(metadata=metadata)

    response = await tool.arun({"group_id": 60})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "403 Forbidden" in error_response["error"]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListGroupAuditEventsInput(group_id=60),
            "List audit events for group 60",
        ),
        (
            ListGroupAuditEventsInput(group_path="gitlab-org/gitlab"),
            "List audit events for group gitlab-org/gitlab",
        ),
    ],
)
def test_list_group_audit_events_format_display_message(input_data, expected_message):
    tool = ListGroupAuditEvents(metadata={})
    assert tool.format_display_message(input_data) == expected_message


# Project Audit Events Tests


@pytest.mark.asyncio
async def test_list_project_audit_events(
    gitlab_client_mock, metadata, project_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=project_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListProjectAuditEvents(metadata=metadata)

    input_data = {"project_id": 7}

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "audit_events": project_audit_event_data,
            "pagination": {
                "total_items": len(project_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/7/audit_events",
        params={"per_page": 20, "page": 1},
        parse_json=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project",
            None,
            "/api/v4/projects/namespace%2Fproject/audit_events",
        ),
        # Test with URL and matching project_id
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            "/api/v4/projects/namespace%2Fproject/audit_events",
        ),
    ],
)
async def test_list_project_audit_events_with_url_success(
    url,
    project_id,
    expected_path,
    gitlab_client_mock,
    metadata,
    project_audit_event_data,
):
    gitlab_client_mock.aget = AsyncMock(return_value=project_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListProjectAuditEvents(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id)

    expected_response = json.dumps(
        {
            "audit_events": project_audit_event_data,
            "pagination": {
                "total_items": len(project_audit_event_data),
                "total_pages": 1,
                "current_page": 1,
                "per_page": 20,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"per_page": 20, "page": 1},
        parse_json=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,error_contains",
    [
        # URL and project_id both given, but don't match
        (
            "https://gitlab.com/namespace/project",
            "different%2Fproject",
            "Project ID mismatch",
        ),
        # URL given isn't a valid GitLab URL
        (
            "https://example.com/not-gitlab",
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_list_project_audit_events_with_url_error(
    url, project_id, error_contains, gitlab_client_mock, metadata
):
    tool = ListProjectAuditEvents(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id)

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_contains in error_response["error"]

    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_project_audit_events_with_filters(
    gitlab_client_mock, metadata, project_audit_event_data
):
    gitlab_client_mock.aget = AsyncMock(return_value=project_audit_event_data)
    gitlab_client_mock.last_response = Mock()
    gitlab_client_mock.last_response.headers = {"X-Total-Pages": "1"}

    tool = ListProjectAuditEvents(metadata=metadata)

    input_data = {
        "project_id": 7,
        "created_after": "2020-05-01T00:00:00Z",
        "created_before": "2020-05-31T23:59:59Z",
    }

    response = await tool.arun(input_data)

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/7/audit_events",
        params={
            "created_after": "2020-05-01T00:00:00Z",
            "created_before": "2020-05-31T23:59:59Z",
            "per_page": 20,
            "page": 1,
        },
        parse_json=True,
    )


@pytest.mark.asyncio
async def test_list_project_audit_events_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("API Error"))

    tool = ListProjectAuditEvents(metadata=metadata)

    with pytest.raises(Exception, match="API Error"):
        await tool.arun({"project_id": 7})


@pytest.mark.asyncio
async def test_list_project_audit_events_api_error_response(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(return_value={"message": "404 Not Found"})

    tool = ListProjectAuditEvents(metadata=metadata)

    response = await tool.arun({"project_id": 7})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "404 Not Found" in error_response["error"]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListProjectAuditEventsInput(project_id=7),
            "List audit events for project 7",
        ),
        (
            ListProjectAuditEventsInput(url="https://gitlab.com/namespace/project"),
            "List audit events for https://gitlab.com/namespace/project",
        ),
    ],
)
def test_list_project_audit_events_format_display_message(input_data, expected_message):
    tool = ListProjectAuditEvents(metadata={})
    assert tool.format_display_message(input_data) == expected_message
