# pylint: disable=dangerous-default-value,import-outside-toplevel
from unittest.mock import AsyncMock, Mock

import fastapi
import pytest
from starlette.datastructures import URL, QueryParams

from ai_gateway.config import ConfigModelLimits


@pytest.fixture(name="limits")
def limits_fixture():
    limits = Mock(spec=ConfigModelLimits)
    limits.for_model.return_value = {
        "concurrency": 100,
        "input_tokens": 50,
        "output_tokens": 25,
    }
    return limits


@pytest.fixture(name="request_factory")
def request_factory_fixture():
    def create(
        request_url: str = "http://0.0.0.0:5052/v1/proxy/test_service/valid_path",
        request_body: bytes = b'{"model": "model1"}',
        request_headers: dict = {"Content-Type": "application/json"},
    ):
        from gitlab_cloud_connector import CloudConnectorUser, UserClaims

        request = Mock(spec=fastapi.Request)
        request.url = URL(request_url)
        request.method = "POST"
        # A real request's query params always agree with its URL, and litellm
        # forwards them to the upstream, so they cannot be left empty here.
        request.query_params = QueryParams(request.url.query)
        mock_request_body = AsyncMock()
        mock_request_body.return_value = request_body
        request.body = mock_request_body
        request.headers = request_headers
        request.user = CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(gitlab_instance_uid="test-instance"),
        )
        return request

    return create
