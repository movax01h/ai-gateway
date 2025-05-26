from typing import Type, cast

import pytest
from dependency_injector import containers, providers

from ai_gateway.searches.search import Searcher, VertexAISearch
from ai_gateway.searches.sqlite_search import SqliteSearch


@pytest.fixture
def config_values(custom_models_enabled: bool):
    return {"custom_models": {"enabled": custom_models_enabled}}


@pytest.mark.parametrize(
    ("custom_models_enabled", "search_provider_class"),
    [(True, SqliteSearch), (False, VertexAISearch)],
)
def test_container(
    mock_container: containers.DeclarativeContainer,
    search_provider_class: Type[Searcher],
):
    searches = cast(providers.Container, mock_container.searches)

    assert isinstance(searches.search_provider(), search_provider_class)
