from unittest.mock import AsyncMock, Mock, patch

import fastapi
import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from starlette.datastructures import URL

from ai_gateway.config import ConfigModelLimits


@pytest.fixture(name="response_headers")
def response_headers_fixture() -> dict:
    return {
        "Content-Type": "application/json",
        "date": "2024",
        "transfer-encoding": "chunked",
    }


@pytest.fixture(name="async_client", autouse=True)
def async_client_fixture(response_headers: dict):
    client = Mock(spec=httpx.AsyncClient)
    response = httpx.Response(
        status_code=200,
        headers=response_headers,
        json={"response": "mocked"},
        request=Mock(),
        content='{"response":"mocked"}',
    )
    client.send.return_value = response
    client.request.return_value = response

    http_handler = Mock(spec=AsyncHTTPHandler, client=client)

    patcher = patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
        return_value=http_handler,
    )
    patcher.start()
    yield client
    patcher.stop()


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
        request.query_params = {}
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
