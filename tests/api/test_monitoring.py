from typing import cast
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from dependency_injector import containers, providers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tree_sitter_languages import get_parser

from ai_gateway.api import create_fast_api_server
from ai_gateway.api.monitoring import execute_default_code_completions, validated
from ai_gateway.config import Config
from ai_gateway.models import ModelAPIError
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.prompts.container import ContainerPrompts
from ai_gateway.searches import Searcher
from ai_gateway.searches.container import ContainerSearches


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {
        "vertex_search": {"fallback_datastore_version": "18.0"},
    }


@pytest.fixture(name="fastapi_server_app")
def fastapi_server_app_fixture(mock_config: Config) -> FastAPI:
    return create_fast_api_server(mock_config)


@pytest.fixture(name="client")
def client_fixture(
    fastapi_server_app: FastAPI,
    # Fixture-on-fixture: requested for side-effect overrides only.
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


@pytest.fixture(name="mock_execute_default_code_completions", autouse=True)
def mock_execute_default_code_completions_fixture():
    with patch(
        "ai_gateway.api.monitoring.execute_default_code_completions",
        new_callable=AsyncMock,
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
    mock_searcher: Mock,
    mock_execute_default_code_completions: AsyncMock,
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        response = client.get("/monitoring/ready")
        # Second call shouldn't run the default code completion again
        client.get("/monitoring/ready")

    assert response.status_code == 200

    assert mock_validate_default_models.call_count == 2
    assert mock_searcher.search.call_count == 2
    mock_execute_default_code_completions.assert_awaited_once()


@pytest.mark.usefixtures("mock_validate_default_models", "mock_searcher")
def test_ready_default_code_completion_failure(
    client: TestClient,
    mock_execute_default_code_completions: AsyncMock,
):
    mock_execute_default_code_completions.side_effect = ModelAPIError("test error")

    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        response = client.get("/monitoring/ready")
        # A failed check isn't cached, so it runs again
        client.get("/monitoring/ready")

    assert response.status_code == 503
    assert mock_execute_default_code_completions.await_count == 2


def test_ready_failure(
    client: TestClient,
    mock_validate_default_models: Mock,
):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=True):
        mock_validate_default_models.side_effect = ModelAPIError("test error")
        response = client.get("/monitoring/ready")

    assert response.status_code == 503

    mock_validate_default_models.assert_called_once()


@pytest.mark.usefixtures(
    "mock_generations",
    "mock_config",
    "mock_validate_default_models",
    "mock_searcher",
)
def test_ready_cloud_connector_failure_from_library(client: TestClient):
    with patch("ai_gateway.api.monitoring.cloud_connector_ready", return_value=False):
        response = client.get("/monitoring/ready")

    assert response.status_code == 503


@pytest.mark.usefixtures("mock_validate_default_models")
def test_ready_doc_search_failure(
    client: TestClient,
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
        }

    @pytest.mark.usefixtures(
        "mock_generations",
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


@pytest.mark.asyncio
async def test_execute_default_code_completions():
    prompt_registry = Mock()
    code_completions = Mock(execute=AsyncMock())
    completions_agent_factory = Mock(return_value=code_completions)
    config = Mock()
    config.mock_model_responses.return_value = False
    config.model_keys.return_value = {}
    config.fireworks_api_base_url.return_value = None
    config.feature_flags.excl_post_process.return_value = []
    config.feature_flags.fireworks_score_threshold.return_value = {}

    await execute_default_code_completions(
        prompt_registry, completions_agent_factory, config
    )

    prompt_registry.get_on_behalf.assert_not_called()
    assert {
        (get_call.args, get_call.kwargs["model_metadata"].name)
        for get_call in prompt_registry.get.call_args_list
    } == {
        (("code_suggestions/completions", None), "codestral_2508_fireworks"),
        (("code_suggestions/completions", None), "codestral_2508_vertex"),
    }
    assert (
        code_completions.execute.await_args_list
        == [
            call(
                prefix="def hello_world():",
                suffix="",
                file_name="monitoring.py",
                editor_lang="python",
                context_max_percent=0.3,
            )
        ]
        * 2
    )


class TestDefaultCodeCompletions:
    """Run the default code completions check through the real completions path, stubbing only model generation."""

    @pytest.fixture(name="config_values")
    def config_values_fixture(self):
        return {
            "fireworks_api_base_url": "https://api.fireworks.ai/inference/v1",
            "model_keys": {"fireworks_provider_api_key": "fw_api_key"},
        }

    # Overrides the module-level autouse fixture so the check runs for real
    @pytest.fixture(name="mock_execute_default_code_completions", autouse=True)
    def mock_execute_default_code_completions_fixture(self):
        yield

    @pytest.fixture(name="client")
    def client_fixture(
        self,
        fastapi_server_app: FastAPI,
        # Fixture-on-fixture: requested for side-effect overrides only.
        mock_ai_gateway_container: containers.Container,  # pylint: disable=unused-argument
    ) -> TestClient:
        return TestClient(fastapi_server_app, raise_server_exceptions=False)

    @pytest.fixture(name="mock_generate", autouse=True)
    def mock_generate_fixture(self, completion: str):
        with patch.object(
            AgentModel,
            "generate",
            new_callable=AsyncMock,
            return_value=TextGenModelOutput(text=completion, score=10**5),
        ) as mock:
            yield mock

    @pytest.fixture(name="get_parser")
    def get_parser_fixture(self):
        with patch(
            "ai_gateway.code_suggestions.prompts.parsers.treesitter.get_parser",
            wraps=get_parser,
        ) as mock:
            yield mock

    @pytest.mark.parametrize("completion", ["", "\n    return 'hello'"])
    @pytest.mark.usefixtures("mock_validate_default_models", "mock_searcher")
    def test_ready(
        self, client: TestClient, mock_generate: AsyncMock, get_parser: Mock
    ):
        with patch(
            "ai_gateway.api.monitoring.cloud_connector_ready", return_value=True
        ):
            response = client.get("/monitoring/ready")

        assert response.status_code == 200
        assert mock_generate.await_count == 2
        get_parser.assert_called()

    @pytest.mark.parametrize("completion", ["", "\n    return 'hello'"])
    @pytest.mark.usefixtures("mock_validate_default_models", "mock_searcher")
    def test_ready_parser_failure(self, client: TestClient, get_parser: Mock):
        get_parser.side_effect = PermissionError("tree-sitter cache")

        with patch(
            "ai_gateway.api.monitoring.cloud_connector_ready", return_value=True
        ):
            response = client.get("/monitoring/ready")
            # A failed check isn't cached, so the parser runs again
            get_parser.reset_mock()
            second_response = client.get("/monitoring/ready")

        assert response.status_code == 500
        assert second_response.status_code == 500
        get_parser.assert_called()
