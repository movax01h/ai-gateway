import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.audit_events import (
    MAX_AUDIT_EVENT_PAGES,
    ListGroupAuditEvents,
    ListGroupAuditEventsInput,
    ListInstanceAuditEvents,
    ListInstanceAuditEventsInput,
    ListProjectAuditEvents,
    ListProjectAuditEventsInput,
    _parse_int,
    _read_pagination_headers,
)

SINGLE_PAGE_HEADERS = {"x-total-pages": "1", "x-next-page": ""}

# (tool class, tool input, expected API path) for each tool sharing the pagination path.
ALL_TOOLS = [
    (ListInstanceAuditEvents, {}, "/api/v4/audit_events"),
    (ListGroupAuditEvents, {"group_id": 60}, "/api/v4/groups/60/audit_events"),
    (ListProjectAuditEvents, {"project_id": 7}, "/api/v4/projects/7/audit_events"),
]


def _events(count: int, start: int = 0) -> list:
    return [{"id": start + i} for i in range(count)]


def _ok(body: list, headers: dict | None = None) -> GitLabHttpResponse:
    return GitLabHttpResponse(status_code=200, body=body, headers=headers or {})


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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=instance_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=instance_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
@pytest.mark.usefixtures("gitlab_client_mock")
async def test_list_instance_audit_events_entity_id_without_type(metadata):
    """Test that validation error for entity_id without entity_type raises ToolException."""
    tool = ListInstanceAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(entity_id=6)

    assert "entity_id requires entity_type to be specified" in str(exc_info.value)


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

    mock_response1 = GitLabHttpResponse(
        status_code=200,
        body=page1_data,
        headers={"x-total-pages": "2", "x-next-page": "2"},
    )
    mock_response2 = GitLabHttpResponse(
        status_code=200,
        body=page2_data,
        headers={"x-total-pages": "2", "x-next-page": ""},
    )
    gitlab_client_mock.aget = AsyncMock(side_effect=[mock_response1, mock_response2])

    tool = ListInstanceAuditEvents(metadata=metadata)

    response = await tool.arun({"fetch_all_pages": True, "per_page": 1})

    result = json.loads(response)
    assert len(result["audit_events"]) == 2
    assert result["pagination"] == {
        "total_items": 2,
        "total_pages": 2,
        "current_page": 2,
        "per_page": 1,
        "truncated": False,
        "has_more": False,
    }
    assert gitlab_client_mock.aget.call_count == 2


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
    """Test that API errors raise ToolException."""
    mock_response = GitLabHttpResponse(
        status_code=401,
        body={"message": "401 Unauthorized"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListInstanceAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun()

    assert "401 Unauthorized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_instance_audit_events_api_error_key_response(
    gitlab_client_mock, metadata
):
    """Test that API errors with 'error' key raise ToolException."""
    mock_response = GitLabHttpResponse(
        status_code=403,
        body={"error": "Access denied"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListInstanceAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun()

    assert "Access denied" in str(exc_info.value)


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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=group_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=group_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
@pytest.mark.usefixtures("gitlab_client_mock")
async def test_list_group_audit_events_no_identifier(metadata):
    """Test that missing group identifier raises ToolException."""
    tool = ListGroupAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun()

    assert "Either group_id or group_path must be provided" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_group_audit_events_with_filters(
    gitlab_client_mock, metadata, group_audit_event_data
):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=group_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListGroupAuditEvents(metadata=metadata)

    input_data = {
        "group_id": 60,
        "created_after": "2019-08-01T00:00:00Z",
        "created_before": "2019-08-31T23:59:59Z",
    }

    await tool.arun(input_data)

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
    """Test that API errors for group audit events raise ToolException."""
    mock_response = GitLabHttpResponse(
        status_code=403,
        body={"message": "403 Forbidden"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListGroupAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(group_id=60)

    assert "403 Forbidden" in str(exc_info.value)


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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=project_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=project_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

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
                "truncated": False,
                "has_more": False,
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
    """Test that URL validation errors raise ToolException."""
    tool = ListProjectAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(url=url, project_id=project_id)

    assert error_contains in str(exc_info.value)

    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_project_audit_events_with_filters(
    gitlab_client_mock, metadata, project_audit_event_data
):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=project_audit_event_data,
        headers=SINGLE_PAGE_HEADERS,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListProjectAuditEvents(metadata=metadata)

    input_data = {
        "project_id": 7,
        "created_after": "2020-05-01T00:00:00Z",
        "created_before": "2020-05-31T23:59:59Z",
    }

    await tool.arun(input_data)

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
    """Test that API errors for project audit events raise ToolException."""
    mock_response = GitLabHttpResponse(
        status_code=404,
        body={"message": "404 Not Found"},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListProjectAuditEvents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_id=7)

    assert "404 Not Found" in str(exc_info.value)


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


# Shared pagination behaviour


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls,tool_input,api_path", ALL_TOOLS)
@pytest.mark.parametrize(
    "next_page_headers",
    [
        pytest.param(lambda page: {"x-next-page": str(page + 1)}, id="x-next-page"),
        pytest.param(lambda page: {}, id="no-x-next-page"),
    ],
)
async def test_fetch_all_pages_stops_at_page_cap(
    tool_cls, tool_input, api_path, next_page_headers, gitlab_client_mock, metadata
):
    per_page = 2
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            _ok(_events(per_page, start=page * per_page), next_page_headers(page))
            for page in range(1, MAX_AUDIT_EVENT_PAGES + 2)
        ]
    )

    tool = tool_cls(metadata=metadata)
    response = await tool.arun(
        {**tool_input, "fetch_all_pages": True, "per_page": per_page}
    )

    result = json.loads(response)
    assert gitlab_client_mock.aget.call_count == MAX_AUDIT_EVENT_PAGES
    assert gitlab_client_mock.aget.call_args.kwargs["path"] == api_path
    assert result["pagination"] == {
        "total_items": per_page * MAX_AUDIT_EVENT_PAGES,
        "total_pages": None,
        "current_page": MAX_AUDIT_EVENT_PAGES,
        "per_page": per_page,
        "truncated": True,
        "has_more": True,
    }


@pytest.mark.asyncio
async def test_fetch_all_pages_cap_reached_on_last_page_is_not_truncated(
    gitlab_client_mock, metadata
):
    """X-next-page on the last fetched page decides has_more when the cap is reached."""
    per_page = 2
    responses = [
        _ok(_events(per_page), {"x-next-page": str(page + 1)})
        for page in range(1, MAX_AUDIT_EVENT_PAGES)
    ]
    responses.append(_ok(_events(per_page), {"x-next-page": ""}))
    gitlab_client_mock.aget = AsyncMock(side_effect=responses)

    tool = ListInstanceAuditEvents(metadata=metadata)
    result = json.loads(
        await tool.arun({"fetch_all_pages": True, "per_page": per_page})
    )

    assert gitlab_client_mock.aget.call_count == MAX_AUDIT_EVENT_PAGES
    assert result["pagination"]["truncated"] is False
    assert result["pagination"]["has_more"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls,tool_input,api_path", ALL_TOOLS)
async def test_fetch_all_pages_later_page_failure_returns_partial_results(
    tool_cls, tool_input, api_path, gitlab_client_mock, metadata
):
    per_page = 2
    first_page = _events(per_page)
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[
            _ok(first_page, {"x-next-page": "2"}),
            GitLabHttpResponse(
                status_code=429, body={"message": "429 Too Many Requests"}
            ),
        ]
    )

    tool = tool_cls(metadata=metadata)
    with patch("duo_workflow_service.tools.audit_events.logger") as mock_logger:
        result = json.loads(
            await tool.arun(
                {**tool_input, "fetch_all_pages": True, "per_page": per_page}
            )
        )

    assert gitlab_client_mock.aget.call_count == 2
    assert gitlab_client_mock.aget.call_args.kwargs["path"] == api_path
    assert result["audit_events"] == first_page
    pagination = result["pagination"]
    assert pagination["total_items"] == per_page
    assert pagination["current_page"] == 1
    assert pagination["truncated"] is True
    assert pagination["has_more"] is True
    assert "HTTP 429" in pagination["error"]
    # The failure is logged once: _process_http_response does not log it as well.
    mock_logger.warning.assert_called_once_with(
        "Audit events pagination stopped early",
        page=2,
        error=pagination["error"],
    )
    mock_logger.error.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fetch_all_pages", [True, False])
async def test_first_page_failure_raises(fetch_all_pages, gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=429, body={"message": "429 Too Many Requests"}
        )
    )

    tool = ListGroupAuditEvents(metadata=metadata)

    with pytest.raises(ToolException, match="HTTP 429"):
        await tool._arun(group_id=60, fetch_all_pages=fetch_all_pages)

    gitlab_client_mock.aget.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,expected_has_more",
    [
        ({"x-next-page": "2", "x-total-pages": "4"}, True),
        ({}, True),
        ({"x-next-page": "", "x-total-pages": "1"}, False),
    ],
)
async def test_single_page_fetch_makes_one_call(
    headers, expected_has_more, gitlab_client_mock, metadata
):
    per_page = 2
    gitlab_client_mock.aget = AsyncMock(return_value=_ok(_events(per_page), headers))

    tool = ListProjectAuditEvents(metadata=metadata)
    result = json.loads(await tool.arun({"project_id": 7, "per_page": per_page}))

    gitlab_client_mock.aget.assert_called_once()
    assert result["pagination"]["truncated"] is False
    assert result["pagination"]["has_more"] is expected_has_more


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"x-next-page": "", "x-total-pages": "3"}, id="lowercase"),
        pytest.param({"X-Next-Page": "", "X-Total-Pages": "3"}, id="capitalised"),
    ],
)
async def test_fetch_all_pages_stops_on_empty_next_page_header(
    headers, gitlab_client_mock, metadata
):
    """A full page with an empty x-next-page is the last page, whatever the header case."""
    per_page = 2
    gitlab_client_mock.aget = AsyncMock(
        side_effect=[_ok(_events(per_page), headers), _ok(_events(per_page))]
    )

    tool = ListInstanceAuditEvents(metadata=metadata)
    result = json.loads(
        await tool.arun({"fetch_all_pages": True, "per_page": per_page})
    )

    gitlab_client_mock.aget.assert_called_once()
    assert result["pagination"]["total_pages"] == 3
    assert result["pagination"]["truncated"] is False
    assert result["pagination"]["has_more"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_headers",
    [
        # x-next-page absent: the short-page check ends the loop.
        pytest.param([{}, {}, {}], id="no-x-next-page"),
        # Large dataset: GitLab omits x-total-pages, which must not stop the loop.
        pytest.param(
            [{"x-next-page": "2"}, {"x-next-page": "3"}, {"x-next-page": "4"}],
            id="no-x-total-pages",
        ),
    ],
)
async def test_fetch_all_pages_stops_on_short_page_before_cap(
    page_headers, gitlab_client_mock, metadata
):
    per_page = 2
    responses = iter(
        [
            _ok(_events(per_page), page_headers[0]),
            _ok(_events(per_page, start=2), page_headers[1]),
            _ok(_events(1, start=4), page_headers[2]),
        ]
    )
    requested_pages = []

    async def aget(path, params, parse_json):
        # params is reused across requests, so record the page at call time.
        requested_pages.append(params["page"])
        return next(responses)

    gitlab_client_mock.aget = AsyncMock(side_effect=aget)

    tool = ListGroupAuditEvents(metadata=metadata)
    result = json.loads(
        await tool.arun({"group_id": 60, "fetch_all_pages": True, "per_page": per_page})
    )

    assert gitlab_client_mock.aget.call_count == 3
    assert requested_pages == [1, 2, 3]
    assert result["audit_events"] == _events(5)
    assert result["pagination"] == {
        "total_items": 5,
        "total_pages": None,
        "current_page": 3,
        "per_page": per_page,
        "truncated": False,
        "has_more": False,
    }


@pytest.mark.parametrize(
    "response,expected",
    [
        (GitLabHttpResponse(status_code=200, body=[], headers=None), {}),
        (Mock(spec=[]), {}),
        (Mock(headers=object()), {}),
        (
            GitLabHttpResponse(
                status_code=200,
                body=[],
                headers={"X-Next-Page": "2", "x-total-pages": "3"},
            ),
            {"x-next-page": "2", "x-total-pages": "3"},
        ),
    ],
)
def test_read_pagination_headers(response, expected):
    assert _read_pagination_headers(response) == expected


@pytest.mark.parametrize(
    "value,expected", [("3", 3), ("0", 0), ("", None), ("abc", None)]
)
def test_parse_int(value, expected):
    assert _parse_int(value) == expected
