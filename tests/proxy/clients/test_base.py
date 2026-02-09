import json
from unittest.mock import ANY

import fastapi
import litellm
import pytest
from fastapi.responses import JSONResponse
from litellm.proxy._types import ProxyException
from starlette.datastructures import URL

from ai_gateway.instrumentators.model_requests import (
    ModelRequestInstrumentator,
    init_llm_operations,
)
from ai_gateway.proxy.clients.base import (
    BaseProxyClient,
    current_proxy_client,
    litellm_async_success_callback,
)
from lib.billing_events import BillingEvent


class TestProxyClient(BaseProxyClient):
    __test__ = False

    def _base_url(self):
        return "https://api.example.com"

    def _allowed_upstream_paths(self):
        return ["/valid_path"]

    def _allowed_headers_to_upstream(self):
        return ["Content-Type"]

    def _allowed_headers_to_downstream(self):
        return ["Content-Length"]

    def _upstream_service(self):
        return "test_service"

    def _allowed_upstream_models(self):
        return ["model1", "model2"]

    def _extract_model_name(self, upstream_path, json_body):
        return json_body.get("model")

    def _extract_stream_flag(self, upstream_path, json_body):
        return json_body.get("stream", False)

    def _update_headers_to_upstream(self, headers):
        headers.update({"X-Test-Header": "test"})


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(
    async_client,
    limits,
    internal_event_client,
    billing_event_client,
):
    """Fixture providing a TestProxyClient instance."""
    return TestProxyClient(limits, internal_event_client, billing_event_client)


@pytest.mark.asyncio
async def test_valid_proxy_request(
    proxy_client,
    request_factory,
):
    response = await proxy_client.proxy(request_factory())

    assert isinstance(response, fastapi.Response)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_obj", "expected_llm_operations"),
    [
        (
            {
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "test-model",
            },
            [
                {
                    "token_count": 15,
                    "model_id": "test-model",
                    "model_engine": "my-llm-provider",
                    "model_provider": "my-model-provider",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            ],
        ),
        (
            {
                "response": '{"model": "test-model", "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}}',
            },
            [
                {
                    "token_count": 10,
                    "model_id": "test-model",
                    "model_engine": "my-llm-provider",
                    "model_provider": "my-model-provider",
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                }
            ],
        ),
    ],
)
async def test_valid_proxy_request_billing_event_callback(
    request_factory,
    proxy_client,
    billing_event_client,
    completion_obj,
    expected_llm_operations,
):
    """Test that the litellm callback tracks billing events with correct parameters."""
    # Reset context var and signal start of usage data collection
    init_llm_operations()

    # Set up the proxy client context and watcher
    current_proxy_client.set(proxy_client)
    proxy_client.user = request_factory().user

    proxy_client.watcher = ModelRequestInstrumentator.WatchContainer(
        llm_provider="my-llm-provider",
        model_provider="my-model-provider",
        labels={"model_engine": "my-model-engine", "model_name": "my-model-name"},
        limits=None,
        streaming=True,
    )

    # Call the callback
    await litellm_async_success_callback(
        _kwargs={},
        completion_obj=completion_obj,
        _start_time=0,
        _end_time=1,
    )

    # Verify billing event was tracked with correct parameters
    billing_event_client.track_billing_event.assert_called_once_with(
        proxy_client.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.proxy.clients.base",
        unit_of_measure="request",
        quantity=1,
        metadata={
            "llm_operations": expected_llm_operations,
            "feature_qualified_name": "ai_gateway_proxy_use",
            "feature_ai_catalog_item": False,
        },
    )


def test_litellm_callback_registered():
    assert litellm_async_success_callback in litellm._async_success_callback


def test_current_proxy_client_context_var_set_on_init(
    async_client,
    limits,
    internal_event_client,
    billing_event_client,
):
    """Test that current_proxy_client context var is set when initializing a proxy client."""
    # Reset context var to None before test
    current_proxy_client.set(None)

    # Create a proxy client
    proxy_client = TestProxyClient(limits, internal_event_client, billing_event_client)

    # Verify the context var is set to the proxy client instance
    assert current_proxy_client.get() is proxy_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_url", "expected_error"),
    [
        ("http://0.0.0.0:5052/v1/proxy/test_service/valid_path", None),
        ("http://0.0.0.0:5052/v1/proxy/test_service/invalid_path", "404: Not found"),
        ("http://0.0.0.0:5052/v1/proxy/unknown_service/valid_path", "404: Not found"),
        (
            "http://0.0.0.0:5052/v1/proxy/test_service/invalid_path/test_service/valid_path",
            "404: Not found",
        ),
    ],
)
async def test_request_url(
    proxy_client,
    request_factory,
    request_url,
    expected_error,
):
    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_url=request_url))
    else:
        response = await proxy_client.proxy(request_factory(request_url=request_url))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_error"),
    [
        (b'{"model": "model1"}', None),
        (b"model is model1", "400: Invalid JSON"),
        (b"", "400: Invalid JSON"),
    ],
)
async def test_request_body(
    proxy_client,
    request_factory,
    request_body,
    expected_error,
):
    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_body=request_body))
    else:
        response = await proxy_client.proxy(request_factory(request_body=request_body))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_error"),
    [
        (b'{"model": "model1"}', None),
        (b'{"model": "expensive-model"}', "400: Unsupported model"),
        (b'{"different_key": "model1"}', "400: Unsupported model"),
        (b"{}", "400: Unsupported model"),
    ],
)
async def test_model_names(
    proxy_client,
    request_factory,
    request_body,
    expected_error,
):
    if expected_error:
        with pytest.raises(fastapi.HTTPException, match=expected_error):
            await proxy_client.proxy(request_factory(request_body=request_body))
    else:
        response = await proxy_client.proxy(request_factory(request_body=request_body))
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_headers", "expected_headers"),
    [
        (
            {"Content-Type": "application/json"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {"Content-Type": "application/json", "X-Gitlab-Instance-Id": "123"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {"Content-Type": "application/json", "X-Test-Header": "unknown"},
            {"Content-Type": "application/json", "X-Test-Header": "test"},
        ),
        (
            {},
            {"X-Test-Header": "test"},
        ),
    ],
)
async def test_upstream_headers(
    async_client,
    limits,
    request_factory,
    request_headers,
    expected_headers,
    internal_event_client,
    billing_event_client,
):
    proxy_client = TestProxyClient(limits, internal_event_client, billing_event_client)

    await proxy_client.proxy(request_factory(request_headers=request_headers))

    async_client.request.assert_called_once_with(
        method="POST",
        url=URL("https://api.example.com/valid_path"),
        headers=expected_headers,
        params={},
        json={"model": "model1"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "expected_streaming"),
    [
        (b'{"model": "model1"}', False),
        (b'{"model": "model1", "stream": true}', True),
    ],
)
async def test_streaming(
    async_client,
    limits,
    request_factory,
    request_body,
    expected_streaming,
    internal_event_client,
    billing_event_client,
):
    proxy_client = TestProxyClient(limits, internal_event_client, billing_event_client)

    response = await proxy_client.proxy(request_factory(request_body=request_body))

    if expected_streaming:
        async_client.send.assert_called_once_with(ANY, stream=expected_streaming)
        assert isinstance(response, fastapi.responses.StreamingResponse)
    else:
        async_client.request.assert_called_once
        assert isinstance(response, fastapi.Response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_headers", "expected_headers"),
    [
        (
            {},
            [("content-length", "21")],
        ),
        (
            {"Vendor-Trace-ID": "123"},
            [("content-length", "21")],
        ),
        (
            {"content-length": "200"},
            [("content-length", "200")],
        ),
    ],
)
async def test_downstream_headers(
    async_client,
    limits,
    request_factory,
    expected_headers,
    internal_event_client,
    billing_event_client,
):
    proxy_client = TestProxyClient(limits, internal_event_client, billing_event_client)

    response = await proxy_client.proxy(request_factory())

    assert response.headers.items() == expected_headers


@pytest.mark.asyncio
async def test_proxy_exception_code(
    async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_client,
):

    error_content = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 200076 tokens > 200000 maximum",
        },
    }

    http_exception = fastapi.HTTPException(
        status_code=400, detail=json.dumps(error_content)
    )

    async_client.request.side_effect = http_exception

    proxy_client = TestProxyClient(limits, internal_event_client, billing_event_client)
    response = await proxy_client.proxy(request_factory())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == error_content
