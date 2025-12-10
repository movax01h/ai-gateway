from typing import cast
from unittest.mock import Mock, call, patch

import pytest
from dependency_injector import containers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import validated
from ai_gateway.config import Config
from ai_gateway.models import ModelAPIError
from ai_gateway.prompts.container import ContainerPrompts


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
    ) as mock:
        yield mock


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
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        response = client.get("/monitoring/ready")

    assert response.status_code == 200

    mock_validate_default_models.assert_called_once()
    assert mock_llm_text.mock_calls == [call("def hello_world():", None, False)]


def test_ready_failure(
    client: TestClient,
    mock_validate_default_models: Mock,
    mock_llm_text: Mock,  # pylint: disable=unused-argument
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        mock_validate_default_models.side_effect = ModelAPIError("test error")
        response = client.get("/monitoring/ready")

    mock_validate_default_models.assert_called_once()
    assert response.status_code == 503


@pytest.mark.usefixtures(
    "mock_generations", "mock_llm_text", "mock_config", "mock_validate_default_models"
)
def test_ready_cloud_connector_failure_from_library(client: TestClient):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=False):
        response = client.get("/monitoring/ready")

    assert response.status_code == 503


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
