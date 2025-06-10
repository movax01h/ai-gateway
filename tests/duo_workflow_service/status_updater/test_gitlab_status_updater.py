from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.status_updater.gitlab_status_updater import (
    GitLabStatusUpdater,
)


@pytest.fixture
def http_client():
    return AsyncMock()


@pytest.fixture
def gitlab_status_updater(http_client):
    return GitLabStatusUpdater(http_client)


@pytest.mark.asyncio
async def test_get_workflow_status(gitlab_status_updater):
    gitlab_status_updater._client.aget = AsyncMock(
        return_value={"id": 391, "status": "running"}
    )

    result = await gitlab_status_updater.get_workflow_status(workflow_id="391")

    assert result is "running"


@pytest.mark.asyncio
async def test_update_workflow_status(gitlab_status_updater):
    gitlab_status_updater._client.apatch = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=200, body={"workflow": {"id": 391, "status": 3}}
        )
    )

    result = await gitlab_status_updater.update_workflow_status(
        workflow_id="391", status_event="drop"
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_workflow_status_when_response_error(gitlab_status_updater):
    gitlab_status_updater._client.apatch = AsyncMock(
        return_value={
            "status": 400,
            "message": "Can not drop workflow that has status failed",
        }
    )

    with pytest.raises(Exception):
        result = await gitlab_status_updater.update_workflow_status(
            workflow_id="391", status_event="drop"
        )

        assert result is None
