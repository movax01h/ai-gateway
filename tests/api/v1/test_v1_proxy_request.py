import json

import pytest
from fastapi import Response

from ai_gateway.api.v1.proxy.request import track_billing_event
from ai_gateway.proxy.clients.token_usage import TokenUsage
from lib.billing_events import BillingEvent


@pytest.fixture
def setup_token_usage(mock_request):  # pylint: disable=redefined-outer-name
    def _setup(
        input_tokens=0,
        output_tokens=0,
        total_tokens=None,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        reasoning_tokens=0,
        model_name=None,
    ):
        mock_request.state.proxy_token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        if model_name:
            mock_request.state.proxy_model_name = model_name

    return _setup


@pytest.mark.asyncio
async def test_track_billing_event(mock_request, billing_event_client):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        return Response(content=b'{"message": "success"}', status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_token_usage(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100, output_tokens=50, model_name="claude-3-5-sonnet-20241022"
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_cache_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=30,
        model_name="claude-3-5-sonnet-20241022",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_exception(mock_request, billing_event_client):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        raise ValueError

    with pytest.raises(ValueError):
        await dummy_func(mock_request, billing_event_client=billing_event_client)

    billing_event_client.track_billing_event.assert_not_called()


@pytest.mark.asyncio
async def test_track_billing_event_with_non_200_status(
    mock_request, billing_event_client
):
    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=400)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    # Should not include metadata for non-200 responses
    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=None,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_format(
    mock_request, billing_event_client, setup_token_usage
):

    setup_token_usage(
        input_tokens=38,
        output_tokens=1024,
        total_tokens=1062,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_cached_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_input_tokens=30,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        response_body = {
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                },
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_input_tokens": 30,
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )


@pytest.mark.asyncio
async def test_track_billing_event_with_openai_reasoning_tokens(
    mock_request, billing_event_client, setup_token_usage
):
    setup_token_usage(
        input_tokens=38,
        output_tokens=1024,
        total_tokens=1062,
        reasoning_tokens=1024,
        model_name="gpt-5-2025-08-07",
    )

    @track_billing_event
    async def dummy_func(*_args, **_kwargs):
        # This is the exact response format from the user's example
        response_body = {
            "id": "chatcmpl-CdJZaJTA6UgqxTVQIWH3N8itNUq9Z",
            "object": "chat.completion",
            "model": "gpt-5-2025-08-07",
            "usage": {
                "prompt_tokens": 38,
                "completion_tokens": 1024,
                "total_tokens": 1062,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 1024,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
        return Response(content=json.dumps(response_body).encode(), status_code=200)

    await dummy_func(mock_request, billing_event_client=billing_event_client)

    expected_metadata = {
        "llm_operations": [
            {
                "model_id": "gpt-5-2025-08-07",
                "prompt_tokens": 38,
                "completion_tokens": 1024,  # Includes reasoning tokens
                "total_tokens": 1062,
                "reasoning_tokens": 1024,  # Reasoning tokens are also tracked separately
            }
        ],
        "feature_qualified_name": "ai_gateway_proxy_use",
        "feature_ai_catalog_item": False,
    }

    billing_event_client.track_billing_event.assert_called_once_with(
        mock_request.user,
        event=BillingEvent.AIGW_PROXY_USE,
        category="ai_gateway.api.v1.proxy.request",
        unit_of_measure="request",
        quantity=1,
        metadata=expected_metadata,
    )
