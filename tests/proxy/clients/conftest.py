from unittest.mock import AsyncMock, Mock

import fastapi
import httpx
import pytest
from starlette.datastructures import URL

from ai_gateway.config import ConfigModelLimits


@pytest.fixture(name="async_client_factory")
def async_client_factory_fixture():
    def create(
        response_status_code: int = 200,
        response_headers: dict = {
            "Content-Type": "application/json",
            "date": "2024",
            "transfer-encoding": "chunked",
        },
        response_json: dict = {"response": "mocked"},
    ):
        client = Mock(spec=httpx.AsyncClient)
        response = httpx.Response(
            status_code=response_status_code,
            headers=response_headers,
            json=response_json,
            request=Mock(),
            content='{"response":"mocked"}',
        )
        client.send.return_value = response
        client.request.return_value = response
        return client

    return create


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
        request = Mock(spec=fastapi.Request)
        request.url = URL(request_url)
        request.method = "POST"
        request.query_params = {}
        mock_request_body = AsyncMock()
        mock_request_body.return_value = request_body
        request.body = mock_request_body
        request.headers = request_headers
        return request

    return create
