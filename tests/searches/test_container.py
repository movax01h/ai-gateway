from typing import Type, cast
from unittest.mock import patch

import pytest
from dependency_injector import containers, providers

from ai_gateway.searches.container import _init_vertex_search_service_client
from ai_gateway.searches.search import Searcher, VertexAISearch
from ai_gateway.searches.sqlite_search import SqliteSearch


@pytest.fixture(name="config_values")
def config_values_fixture(custom_models_enabled: bool):
    return {"custom_models": {"enabled": custom_models_enabled}}


@pytest.mark.parametrize(
    ("custom_models_enabled", "search_provider_class"),
    [(True, SqliteSearch), (False, VertexAISearch)],
)
def test_container(
    mock_ai_gateway_container: containers.DeclarativeContainer,
    search_provider_class: Type[Searcher],
):
    searches = cast(providers.Container, mock_ai_gateway_container.searches)
    assert isinstance(searches.search_provider(), search_provider_class)


@pytest.mark.parametrize(
    ("args", "expected_init"),
    [
        ({"mock_model_responses": False, "custom_models_enabled": False}, True),
        ({"mock_model_responses": False, "custom_models_enabled": True}, False),
        ({"mock_model_responses": True, "custom_models_enabled": False}, False),
    ],
)
def test_init_vertex_search_service_client(args, expected_init):
    with patch(
        "google.cloud.discoveryengine.SearchServiceAsyncClient"
    ) as mock_search_client:
        _init_vertex_search_service_client(**args)

        if expected_init:
            mock_search_client.assert_called_once()
        else:
            mock_search_client.assert_not_called()
