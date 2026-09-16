# pylint: disable=line-too-long
import json
from unittest.mock import Mock

import fastapi
import httpx
import litellm
import pytest
from fastapi.responses import JSONResponse

from ai_gateway.instrumentators.model_requests import (
    ModelRequestInstrumentator,
    init_llm_operations,
)
from ai_gateway.proxy.clients.base import (
    ProxyClient,
    ProxyModel,
    current_proxy_client,
    litellm_async_success_callback,
)
from lib.billing_events import BillingEvent
from lib.billing_events.service import ExecutionEnvironment


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(
    limits,
    internal_event_client,
    billing_event_service,
):
    """Fixture providing a ProxyClient instance backed by a real BillingEventService."""
    return ProxyClient(limits, internal_event_client, billing_event_service)


@pytest.fixture(name="stream")
def stream_fixture() -> bool:
    """Stream flag the test ProxyModel is built with; parametrize to override."""
    return False


@pytest.fixture(name="allowed_headers_to_downstream")
def allowed_headers_to_downstream_fixture() -> list[str]:
    """Response headers the test ProxyModel relays; parametrize to override."""
    return ["Content-Length"]


@pytest.fixture(name="test_proxy_model")
def test_proxy_model_fixture(stream: bool, allowed_headers_to_downstream: list[str]):
    """Fixture providing a test ProxyModel."""
    return ProxyModel(
        base_url="https://api.example.com",
        model_name="test-model",
        upstream_path="/valid_path",
        stream=stream,
        upstream_service="test_service",
        headers_to_upstream={"X-Test-Header": "test"},
        allowed_upstream_models=["test-model"],
        allowed_headers_to_upstream=["Content-Type"],
        allowed_headers_to_downstream=allowed_headers_to_downstream,
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_proxy_async_client")
async def test_valid_proxy_request(
    proxy_client,
    request_factory,
    test_proxy_model,
):
    response = await proxy_client.proxy(request_factory(), test_proxy_model)

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
                    "model_id": "my-model-name",
                    "model_engine": "my-model-provider",
                    "model_provider": "my-model-provider",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "agent_name": None,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "operation_type": "standard",
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
                    "model_id": "my-model-name",
                    "model_engine": "my-model-provider",
                    "model_provider": "my-model-provider",
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "agent_name": None,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "operation_type": "standard",
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

    # The service forwards to BillingEventsClient with positional event/category and
    # an enriched metadata dict (execution_environment, tool_names, orbit_called).
    billing_event_client.track_billing_event.assert_called_once_with(
        proxy_client.user,
        BillingEvent.AIGW_PROXY_USE,
        "ai_gateway.proxy.clients.base",
        unit_of_measure="request",
        quantity=1,
        metadata={
            "feature_qualified_name": "ai_gateway_proxy_use",
            "feature_ai_catalog_item": False,
            "execution_environment": ExecutionEnvironment.DAP.value,
            "llm_operations": expected_llm_operations,
            "tool_names": [],
            "orbit_called": False,
        },
    )


def test_litellm_callback_registered():
    assert litellm_async_success_callback in litellm._async_success_callback


def test_current_proxy_client_context_var_set_on_init(
    limits,
    internal_event_client,
    billing_event_service,
):
    """Test that current_proxy_client context var is set when initializing a proxy client."""
    # Reset context var to None before test
    current_proxy_client.set(None)

    # Create a proxy client
    proxy_client = ProxyClient(limits, internal_event_client, billing_event_service)

    # Verify the context var is set to the proxy client instance
    assert current_proxy_client.get() is proxy_client


@pytest.mark.asyncio
async def test_proxy_exception_code(
    mock_proxy_async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_service,
    test_proxy_model,
):
    """A ProxyException from litellm is returned as its parsed JSON message.

    Upstream error responses no longer arrive this way, so this covers litellm's own failures only.
    """
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

    # litellm dispatches every pass-through request through `send()`.
    mock_proxy_async_client.send.side_effect = http_exception

    proxy_client = ProxyClient(limits, internal_event_client, billing_event_service)
    response = await proxy_client.proxy(request_factory(), test_proxy_model)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == error_content


@pytest.mark.asyncio
async def test_proxy_exception_code_with_malformed_json_message(
    mock_proxy_async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_service,
    test_proxy_model,
):
    """A ProxyException message that is not JSON is wrapped in a message dict."""
    error_content = "type: invalid_request_error, message: prompt is too long: 200076 tokens > 200000 maximum"

    http_exception = fastapi.HTTPException(status_code=400, detail=error_content)

    # litellm dispatches every pass-through request through `send()`.
    mock_proxy_async_client.send.side_effect = http_exception

    proxy_client = ProxyClient(limits, internal_event_client, billing_event_service)
    response = await proxy_client.proxy(request_factory(), test_proxy_model)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == {"message": error_content}


_UPSTREAM_RATE_LIMIT_BODY = b'{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your organization\'s rate limit of 8,000,000 input tokens per minute"}}'


async def _read_body(response: fastapi.Response) -> bytes:
    """Read a proxied response body, whether it streams or arrives buffered."""
    if isinstance(response, fastapi.responses.StreamingResponse):
        return b"".join(
            [
                chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                async for chunk in response.body_iterator
            ]
        )

    return bytes(response.body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "request_body", "allowed_headers_to_downstream"),
    [
        pytest.param(
            True,
            b'{"model": "model1", "stream": true}',
            ["Content-Type"],
            id="streaming",
        ),
        pytest.param(
            False,
            b'{"model": "model1"}',
            ["Content-Type"],
            id="non-streaming",
        ),
    ],
)
async def test_upstream_error_body_relayed_unchanged(
    mock_proxy_async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_service,
    test_proxy_model,
    request_body,
):
    """An upstream error response reaches the client with its body untouched.

    A streaming upstream error used to be re-raised with the body read as
    bytes, so it arrived here as the string `b'{"type":"error",...}'`, failed
    to parse as JSON, and was served as `{"message": "b'...'"}` instead of the
    provider's own envelope. Both paths now relay the response as it came.
    """
    mock_proxy_async_client.send.return_value = httpx.Response(
        status_code=429,
        headers={"Content-Type": "application/json"},
        content=_UPSTREAM_RATE_LIMIT_BODY,
        request=Mock(),
    )

    proxy_client = ProxyClient(limits, internal_event_client, billing_event_service)
    response = await proxy_client.proxy(
        request_factory(request_body=request_body), test_proxy_model
    )

    assert response.status_code == 429
    assert response.headers["content-type"] == "application/json"
    assert await _read_body(response) == _UPSTREAM_RATE_LIMIT_BODY
