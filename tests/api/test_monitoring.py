from unittest.mock import Mock, call, patch

import pytest
from dependency_injector import containers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import validated
from ai_gateway.config import Config
from ai_gateway.models import ModelAPIError


@pytest.fixture
def config_values():
    # test using a valid looking fireworks config so we can stub out the actual
    # call rather than the `from_model_name` classmethod
    return {
        "model_keys": {"fireworks_api_key": "fw_api_key"},
        "model_endpoints": {
            "fireworks_regional_endpoints": {
                "us-central1": {
                    "endpoint": "https://fireworks.endpoint.com/v1",
                    "identifier": "accounts/fireworks/models/qwen2p5-coder-7b#accounts/deployment/deadbeef",
                },
            }
        },
    }


@pytest.fixture
def fastapi_server_app(mock_config: Config) -> FastAPI:
    return create_fast_api_server(mock_config)


@pytest.fixture
def client(
    fastapi_server_app: FastAPI, mock_container: containers.Container
) -> TestClient:
    return TestClient(fastapi_server_app)


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
    mock_generations: Mock,
    mock_completions_legacy: Mock,
    mock_llm_text: Mock,
):
    response = client.get("/monitoring/ready")
    response = client.get("/monitoring/ready")

    assert response.status_code == 200
    # assert we only called each model once
    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    assert mock_generations.mock_calls == [
        call.execute(
            prefix="",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]

    assert mock_llm_text.mock_calls == [call("def hello_world():", None, False)]

    # Assert the attributes of the mock code generations object
    assert mock_generations.return_value.model.name == "claude-3-haiku-20240307"


def model_failure(*args, **kwargs):
    raise ModelAPIError("Vertex unreachable")


def test_ready_vertex_failure(
    client: TestClient,
    mock_generations: Mock,
    mock_completions_legacy: Mock,
    mock_llm_text: Mock,
):
    mock_generations.side_effect = model_failure
    mock_completions_legacy.side_effect = model_failure

    response = client.get("/monitoring/ready")

    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]
    # Don't try anthropic if vertex is not available, no need to spend
    # the money if the service is not going to be ready
    assert not mock_generations.mock_calls
    assert response.status_code == 503


def test_ready_anthropic_failure(
    client: TestClient,
    mock_generations: Mock,
    mock_completions_legacy: Mock,
    mock_llm_text: Mock,
):
    mock_generations.side_effect = model_failure

    response = client.get("/monitoring/ready")

    assert mock_completions_legacy.mock_calls == [
        call.execute(
            prefix="def hello_world():",
            suffix="",
            file_name="monitoring.py",
            editor_lang="python",
        )
    ]

    assert mock_generations.mock_calls == [
        call.execute(
            prefix="",
            file_name="monitoring.py",
            editor_lang="python",
            model_provider="anthropic",
        )
    ]

    # Assert the attributes of the mock code generations object
    assert mock_generations.return_value.model.name == "claude-3-haiku-20240307"

    assert response.status_code == 503


def test_ready_fireworks_failure(
    client: TestClient,
    mock_generations: Mock,
    mock_completions_legacy: Mock,
    mock_llm_text: Mock,
):
    mock_llm_text.side_effect = model_failure
    response = client.get("/monitoring/ready")

    assert response.status_code == 503


def test_ready_cloud_connector_failure_from_library(
    client: TestClient,
    mock_generations: Mock,
    mock_completions_legacy: Mock,
    mock_llm_text: Mock,
    mock_config: Config,
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=False):
        response = client.get("/monitoring/ready")

    assert response.status_code == 503


class TestCustomModelEnabled:
    # This overrides conftest.py `config_values` which is referenced by `mock_config`
    @pytest.fixture
    def config_values(self):
        yield {
            "custom_models": {"enabled": "true"},
            "model_keys": {"fireworks_api_key": "fw_api_key"},
            "model_endpoints": {
                "fireworks_regional_endpoints": {
                    "us-central1": {
                        "endpoint": "https://fireworks.endpoint.com/v1",
                        "identifier": "accounts/fireworks/models/qwen2p5-coder-7b#accounts/deployment/deadbeef",
                    },
                }
            },
        }

    def test_ready_custom_models_enabled_skips_cloud_connector(
        self,
        client: TestClient,
        mock_generations: Mock,
        mock_completions_legacy: Mock,
        mock_llm_text: Mock,
        mock_config: Config,
    ):
        # Even if cloud_connector_ready would return False, the endpoint should still work
        with patch(
            "ai_gateway.api.monitoring.cloud_connector_ready", return_value=False
        ):
            response = client.get("/monitoring/ready")

        assert response.status_code == 200
