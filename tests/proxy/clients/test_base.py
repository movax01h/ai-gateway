import json

import fastapi
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


@pytest.fixture(name="proxy_client")
def proxy_client_fixture(
    limits,
    internal_event_client,
    billing_event_client,
):
    """Fixture providing a ProxyClient instance."""
    return ProxyClient(limits, internal_event_client, billing_event_client)


@pytest.fixture(name="test_proxy_model")
def test_proxy_model_fixture():
    """Fixture providing a test ProxyModel."""
    return ProxyModel(
        base_url="https://api.example.com",
        model_name="test-model",
        upstream_path="/valid_path",
        stream=False,
        upstream_service="test_service",
        headers_to_upstream={"X-Test-Header": "test"},
        allowed_upstream_models=["test-model"],
        allowed_headers_to_upstream=["Content-Type"],
        allowed_headers_to_downstream=["Content-Length"],
    )


@pytest.mark.asyncio
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
    limits,
    internal_event_client,
    billing_event_client,
):
    """Test that current_proxy_client context var is set when initializing a proxy client."""
    # Reset context var to None before test
    current_proxy_client.set(None)

    # Create a proxy client
    proxy_client = ProxyClient(limits, internal_event_client, billing_event_client)

    # Verify the context var is set to the proxy client instance
    assert current_proxy_client.get() is proxy_client


@pytest.mark.asyncio
async def test_proxy_exception_code(
    async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_client,
    test_proxy_model,
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

    proxy_client = ProxyClient(limits, internal_event_client, billing_event_client)
    response = await proxy_client.proxy(request_factory(), test_proxy_model)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == error_content


@pytest.mark.asyncio
async def test_proxy_exception_code_with_malformed_json_message(
    async_client,
    limits,
    request_factory,
    internal_event_client,
    billing_event_client,
    test_proxy_model,
):

    error_content = "type: invalid_request_error, message: prompt is too long: 200076 tokens > 200000 maximum"

    http_exception = fastapi.HTTPException(status_code=400, detail=error_content)

    async_client.request.side_effect = http_exception

    proxy_client = ProxyClient(limits, internal_event_client, billing_event_client)
    response = await proxy_client.proxy(request_factory(), test_proxy_model)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert json.loads(response.body) == {"message": error_content}
