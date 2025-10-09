from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.events import get_event
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse


@pytest.mark.asyncio
async def test_get_events():
    gitlab_client = AsyncMock()

    data = [
        {"id": 1, "event_type": "example_event"},
        {"id": 2, "event_type": "example_event_2"},
    ]

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=data,
    )
    gitlab_client.aget = AsyncMock(return_value=mock_response)

    workflow_id = "123"
    event = await get_event(gitlab_client, workflow_id)

    # Check if the correct API path was called
    gitlab_client.aget.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/123/events",
        parse_json=True,
        use_http_response=True,
    )

    # Check if the events are returned correctly
    assert event is not None
    assert event["id"] == 1


@pytest.mark.asyncio
async def test_get_events_with_ack():
    gitlab_client = AsyncMock()

    data = [
        {"id": 1, "event_type": "example_event"},
    ]

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=data,
    )
    gitlab_client.aget = AsyncMock(return_value=mock_response)

    workflow_id = "123"

    await get_event(gitlab_client, workflow_id, ack=True)

    # Check if the acknowledgement API path was called
    gitlab_client.aput.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/123/events/1",
        body='{"event_status": "delivered"}',
        use_http_response=True,
    )


@pytest.mark.asyncio
async def test_get_events_with_bad_request():
    gitlab_client = AsyncMock()

    mock_response = GitLabHttpResponse(
        status_code=400,
        body={"status": 400, "reason": "Bad Request"},
    )
    gitlab_client.aget = AsyncMock(return_value=mock_response)

    workflow_id = "123"

    event = await get_event(gitlab_client, workflow_id, ack=True)

    assert event is None
