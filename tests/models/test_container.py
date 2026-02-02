from typing import cast
from unittest.mock import patch

import httpx
import pytest
from dependency_injector import containers, providers

from ai_gateway.models.container import (
    _init_anthropic_proxy_client,
    _init_vertex_ai_proxy_client,
    _init_vertex_grpc_client,
)
from ai_gateway.models.litellm import LiteLlmTextGenModel
from ai_gateway.proxy.clients.anthropic import AnthropicProxyClient
from ai_gateway.proxy.clients.openai import OpenAIProxyClient
from ai_gateway.proxy.clients.vertex_ai import VertexAIProxyClient


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {
                "endpoint": "test",
                "mock_model_responses": False,
                "custom_models_enabled": False,
            },
            True,
        ),
        (
            {
                "endpoint": "test",
                "mock_model_responses": False,
                "custom_models_enabled": True,
            },
            False,
        ),
        (
            {
                "endpoint": "test",
                "mock_model_responses": True,
                "custom_models_enabled": False,
            },
            False,
        ),
    ],
)
def test_init_vertex_grpc_client(args, expected_init):
    with patch(
        # "google.cloud.aiplatform.gapic.PredictionServiceAsyncClient"
        "ai_gateway.models.container.grpc_connect_vertex"
    ) as mock_grpc_client:
        _init_vertex_grpc_client(**args)

        if expected_init:
            mock_grpc_client.assert_called_once_with({"api_endpoint": args["endpoint"]})
        else:
            mock_grpc_client.assert_not_called()


def test_anthropic_proxy_client():
    with patch("httpx.AsyncClient") as mock_httpx_client:
        _init_anthropic_proxy_client()

    mock_httpx_client.assert_called_once_with(
        timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0),
    )


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        (
            {
                "model_keys": {"fireworks_api_key": "test_fireworks_key"},
                "model_endpoints": {
                    "fireworks_current_region_endpoint": {
                        "endpoint": "https://test.fireworks.ai/"
                    }
                },
            },
            True,
        ),
        ({}, False),
    ],
)
def _init_async_fireworks_client(args, expected_init):
    with patch("AsyncOpenAI") as mock_openai_client:
        _init_async_fireworks_client(**args)

        if expected_init:
            mock_openai_client.assert_called_once_with(api_key="test_fireworks_key")
        else:
            mock_openai_client.assert_not_called()


def test_vertex_ai_proxy_client():
    with patch("httpx.AsyncClient") as mock_httpx_client:
        _init_vertex_ai_proxy_client()

    mock_httpx_client.assert_called_once_with(
        timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0),
    )


@pytest.mark.asyncio
async def test_container(mock_ai_gateway_container: containers.DeclarativeContainer):
    models = cast(providers.Container, mock_ai_gateway_container.pkg_models)

    assert isinstance(models.anthropic_proxy_client(), AnthropicProxyClient)
    assert isinstance(models.vertex_ai_proxy_client(), VertexAIProxyClient)
    assert isinstance(models.openai_proxy_client(), OpenAIProxyClient)
    assert isinstance(models.litellm(name="gpt"), LiteLlmTextGenModel)
