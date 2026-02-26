from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.api.v1.proxy.request import authorize_with_unit_primitive_header


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_missing_header(mock_request):
    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == f"Missing {X_GITLAB_UNIT_PRIMITIVE} header"


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_unknown(mock_request):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = "odd_feature"

    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unknown unit primitive header odd_feature"


@pytest.mark.asyncio
async def test_authorize_with_unit_primitive_header_unauthorized(mock_request):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = GitLabUnitPrimitive.DUO_CHAT

    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    with pytest.raises(HTTPException) as exc_info:
        await dummy_func(mock_request)

    assert exc_info.value.status_code == 403
    assert (
        exc_info.value.detail
        == f"Unauthorized to access {GitLabUnitPrimitive.DUO_CHAT}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", [["duo_chat"]])
async def test_authorize_with_unit_primitive_header_authorized(mock_request):
    mock_request.headers[X_GITLAB_UNIT_PRIMITIVE] = GitLabUnitPrimitive.DUO_CHAT

    @authorize_with_unit_primitive_header()
    async def dummy_func(request):
        return "Success"

    result = await dummy_func(mock_request)
    assert result == "Success"
