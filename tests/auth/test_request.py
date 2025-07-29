from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks, HTTPException, Request
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.api.v1.proxy.request import authorize_with_unit_primitive_header


@pytest.fixture(name="mock_request")
def mock_request_fixture():
    request = Mock(spec=Request)
    request.headers = {}
    request.user = Mock(spec=StarletteUser)
    return request


@pytest.fixture(name="mock_background_tasks")
def mock_background_tasks_fixture():
    background_tasks = Mock(spec=BackgroundTasks)
    return background_tasks


@pytest.fixture(name="mock_abuse_detector")
def mock_abuse_detector_fixture():
    abuse_detector = Mock(spec=AbuseDetector)
    return abuse_detector


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_missing_header(
    mock_request, mock_background_tasks, mock_abuse_detector
):
    @authorize_with_unit_primitive_header()
    async def dummy_func(request, background_tasks, abuse_detector):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request, mock_background_tasks, mock_abuse_detector)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == f"Missing {X_GITLAB_UNIT_PRIMITIVE} header"
    assert not mock_background_tasks.add_task.called


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_unknown(
    mock_request, mock_background_tasks, mock_abuse_detector
):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = "odd_feature"

    @authorize_with_unit_primitive_header()
    async def dummy_func(request, background_tasks, abuse_detector):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request, mock_background_tasks, mock_abuse_detector)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unknown unit primitive header odd_feature"
    assert not mock_background_tasks.add_task.called


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_unauthorized(
    mock_request, mock_background_tasks, mock_abuse_detector
):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = GitLabUnitPrimitive.DUO_CHAT
    mock_request.user.can.return_value = False

    @authorize_with_unit_primitive_header()
    async def dummy_func(request, background_tasks, abuse_detector):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request, mock_background_tasks, mock_abuse_detector)

    assert exc_info.value.status_code == 403
    assert (
        exc_info.value.detail
        == f"Unauthorized to access {GitLabUnitPrimitive.DUO_CHAT}"
    )
    assert not mock_background_tasks.add_task.called


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_authorized(
    mock_request, mock_background_tasks, mock_abuse_detector
):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = GitLabUnitPrimitive.DUO_CHAT
    mock_request.user.can.return_value = True
    mock_abuse_detector.should_detect.return_value = True

    @authorize_with_unit_primitive_header()
    async def dummy_func(request, background_tasks, abuse_detector):
        return "Success"

    result = await dummy_func(mock_request, mock_background_tasks, mock_abuse_detector)
    assert result == "Success"
    assert mock_background_tasks.add_task.called


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_authorized_without_abuse_detection(
    mock_request, mock_background_tasks, mock_abuse_detector
):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = GitLabUnitPrimitive.DUO_CHAT
    mock_request.user.can.return_value = True
    mock_abuse_detector.should_detect.return_value = False

    @authorize_with_unit_primitive_header()
    async def dummy_func(request, background_tasks, abuse_detector):
        return "Success"

    result = await dummy_func(mock_request, mock_background_tasks, mock_abuse_detector)
    assert result == "Success"
    assert not mock_background_tasks.add_task.called
