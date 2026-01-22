from typing import cast
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from dependency_injector import containers, providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import validated
from ai_gateway.config import Config
from ai_gateway.models import ModelAPIError
from ai_gateway.prompts.container import ContainerPrompts
from ai_gateway.searches import Searcher
from ai_gateway.searches.container import ContainerSearches


@pytest.fixture(name="config_values")
def config_values_fixture():
    # test using a valid looking fireworks config so we can stub out the actual
    # call rather than the `from_model_name` classmethod
    return {
        "model_keys": {"fireworks_api_key": "fw_api_key"},
        "model_endpoints": {
            "fireworks_regional_endpoints": {
                "us-central1": {
                    "qwen2p5-coder-7b": {
                        "endpoint": "https://fireworks.endpoint.com/v1",
                        "identifier": "accounts/fireworks/models/qwen2p5-coder-7b#accounts/deployment/deadbeef",
                    },
                },
            }
        },
        "vertex_search": {"fallback_datastore_version": "18.0"},
    }


@pytest.fixture(name="fastapi_server_app")
def fastapi_server_app_fixture(mock_config: Config) -> FastAPI:
    return create_fast_api_server(mock_config)


@pytest.fixture(name="client")
def client_fixture(
    fastapi_server_app: FastAPI,
    mock_ai_gateway_container: containers.Container,  # pylint: disable=unused-argument
) -> TestClient:
    return TestClient(fastapi_server_app)


@pytest.fixture(name="mock_validate_default_models")
def mock_validate_default_models_fixture(
    mock_ai_gateway_container: containers.Container,
):
    with patch.object(
        cast(ContainerPrompts, mock_ai_gateway_container.pkg_prompts).prompt_registry(),
        "validate_default_models",
        return_value=True,
    ) as mock:
        yield mock


@pytest.fixture(name="mock_searcher")
def mock_searcher_fixture(
    mock_ai_gateway_container: containers.Container,
):
    mock_searcher = AsyncMock(spec=Searcher)
    mock_searcher.search_with_retry = AsyncMock(return_value=[])
    mock_searcher.search = AsyncMock(return_value=[])

    search_container = cast(ContainerSearches, mock_ai_gateway_container.searches)
    mock_provider = providers.Factory(lambda: mock_searcher)

    with search_container.search_provider.override(mock_provider):
        yield mock_searcher


# Avoid the global state of checks leaking between tests
@pytest.fixture(autouse=True)
def reset_validated():
    validated.clear()
    yield


def test_healthz(client: TestClient):
    response = client.get("/monitoring/healthz")
    assert response.status_code == 200


def test_ready(
    client: TestClient,
    mock_validate_default_models: Mock,
    mock_llm_text: Mock,
    mock_searcher: Mock,
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        response = client.get("/monitoring/ready")

    assert response.status_code == 200

    mock_validate_default_models.assert_called_once()
    assert mock_llm_text.mock_calls == [call("def hello_world():", None, False)]
    mock_searcher.search.assert_called_once()


def test_ready_failure(
    client: TestClient,
    mock_validate_default_models: Mock,
    mock_llm_text: Mock,  # pylint: disable=unused-argument
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        mock_validate_default_models.side_effect = ModelAPIError("test error")
        response = client.get("/monitoring/ready")

    assert response.status_code == 503

    mock_validate_default_models.assert_called_once()


@pytest.mark.usefixtures(
    "mock_generations",
    "mock_llm_text",
    "mock_config",
    "mock_validate_default_models",
    "mock_searcher",
)
def test_ready_cloud_connector_failure_from_library(client: TestClient):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=False):
        response = client.get("/monitoring/ready")

    assert response.status_code == 503


def test_ready_doc_search_failure(
    client: TestClient,
    mock_validate_default_models: Mock,  # pylint: disable=unused-argument
    mock_llm_text: Mock,  # pylint: disable=unused-argument
    mock_searcher: Mock,
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        mock_searcher.search.side_effect = Exception("Unknown error")
        response = client.get("/monitoring/ready")

    assert response.status_code == 503

    mock_searcher.search.assert_called_once_with(
        query="can I upload images to GitLab repo?",
        page_size=1,
        gl_version="18.0",  # Uses fallback_datastore_version from config
    )


class TestCustomModelEnabled:
    # This overrides conftest.py `config_values` which is referenced by `mock_config`
    @pytest.fixture(name="config_values")
    def config_values_fixture(self):
        yield {
            "custom_models": {"enabled": "true"},
            "model_keys": {"fireworks_api_key": "fw_api_key"},
            "model_endpoints": {
                "fireworks_regional_endpoints": {
                    "us-central1": {
                        "qwen2p5-coder-7b": {
                            "endpoint": "https://fireworks.endpoint.com/v1",
                            "identifier": "accounts/fireworks/models/qwen2p5-coder-7b#accounts/deployment/deadbeef",
                        },
                    },
                }
            },
        }

    @pytest.mark.usefixtures(
        "mock_generations",
        "mock_llm_text",
        "mock_config",
        "mock_validate_default_models",
        "mock_searcher",
    )
    def test_ready_custom_models_enabled_skips_cloud_connector(
        self,
        client: TestClient,
    ):
        # Even if cloud_connector_ready would return False, the endpoint should still work
        with patch(
            "ai_gateway.api.monitoring.cloud_connector_ready", return_value=False
        ):
            response = client.get("/monitoring/ready")

        assert response.status_code == 200
