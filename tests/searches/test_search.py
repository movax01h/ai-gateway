from unittest.mock import AsyncMock, call, patch

import pytest
from fastapi import HTTPException, status
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from ai_gateway.searches.search import DataStoreNotFound, VertexAISearch
from ai_gateway.searches.typing import SearchResult


@pytest.fixture(name="mock_vertex_search_struct_data")
def mock_vertex_search_struct_data_fixture():
    return {
        "content": "GitLab's mission is to make software development easier and more efficient.",
        "metadata": {
            "source": "GitLab Docs",
            "version": "17.0.0",
            "source_url": "https://docs.gitlab.com/ee/foo",
        },
    }


@pytest.fixture(name="mock_vertex_search_response")
def mock_vertex_search_response_fixture(mock_vertex_search_struct_data):
    response_dict = {
        "results": [
            {
                "document": {
                    "id": "1",
                    "struct_data": ParseDict(mock_vertex_search_struct_data, Struct()),
                },
            }
        ],
    }

    return discoveryengine.SearchResponse(**response_dict)


@pytest.fixture(name="mock_vertex_search_request")
def mock_vertex_search_request_fixture():
    with patch("ai_gateway.searches.container.discoveryengine.SearchRequest") as mock:
        yield mock


@pytest.fixture(name="mock_search_service_client")
def mock_search_service_client_fixture(mock_vertex_search_response):
    with patch(
        "ai_gateway.searches.container.discoveryengine.SearchServiceAsyncClient"
    ) as mock:
        mock.search = AsyncMock(return_value=mock_vertex_search_response)
        mock.serving_config_path.return_value = "path/to/service_config"
        yield mock


@pytest.fixture(name="vertex_ai_search_factory")
def vertex_ai_search_factory_fixture():
    def create(
        client: discoveryengine.SearchServiceAsyncClient,
        project: str = "test-project",
        fallback_datastore_version: str = "17.0.0",
    ) -> VertexAISearch:
        return VertexAISearch(
            client=client,
            project=project,
            fallback_datastore_version=fallback_datastore_version,
        )

    return create


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_success_first_attempt(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
        await vertex_search.search_with_retry(query, gl_version)

        mock_search.assert_called_once_with(query, "17.1.0")


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_success_second_attempt(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
        mock_search.side_effect = [
            DataStoreNotFound("Data store not found", input="17.1.0"),
            AsyncMock(),
        ]

        await vertex_search.search_with_retry(query, gl_version)

        mock_search.assert_has_calls(
            [call(query, "17.1.0"), call(query, "17.1.0", gl_version="17.0.0")]
        )


@pytest.mark.asyncio
async def test_vertex_ai_search_with_retry_failed_all_attempts(
    mock_search_service_client, vertex_ai_search_factory
):
    query = "test query"
    gl_version = "17.1.0"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, fallback_datastore_version="17.0.0"
    )

    with pytest.raises(HTTPException):
        with patch.object(VertexAISearch, "search", return_value=None) as mock_search:
            mock_search.side_effect = [
                HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                ),
                HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data store not found.",
                ),
            ]

            await vertex_search.search_with_retry(query, gl_version)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version, expected_data_store_id",
    [
        ("1.2.3", "gitlab-docs-1-2"),
        ("10.11.12-pre", "gitlab-docs-10-11"),
    ],
)
@pytest.mark.usefixtures("mock_vertex_search_response")
async def test_vertex_ai_search(
    mock_search_service_client,
    mock_vertex_search_request,
    mock_vertex_search_struct_data,
    gl_version,
    expected_data_store_id,
    vertex_ai_search_factory,
):
    project = "test-project"
    query = "test query"

    vertex_search = vertex_ai_search_factory(
        client=mock_search_service_client, project=project
    )

    result = await vertex_search.search(query, gl_version)
    expected = [
        {
            "id": "1",
            "content": mock_vertex_search_struct_data["content"],
            "metadata": mock_vertex_search_struct_data["metadata"],
        }
    ]
    assert [r.model_dump() for r in result] == expected

    mock_search_service_client.serving_config_path.assert_called_once_with(
        project=project,
        location="global",
        data_store=expected_data_store_id,
        serving_config="default_config",
    )
    mock_search_service_client.search.assert_called_once_with(
        mock_vertex_search_request.return_value
    )


@pytest.mark.asyncio
async def test_invalid_version(mock_search_service_client, vertex_ai_search_factory):
    query = "test query"
    gl_version = "invalid version"
    vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

    with pytest.raises(DataStoreNotFound):
        await vertex_search.search(query, gl_version)

    mock_search_service_client.serving_config_path.assert_not_called()
    mock_search_service_client.search.assert_not_called()


@pytest.mark.asyncio
async def test_datastore_not_found(
    mock_search_service_client,
    vertex_ai_search_factory,
):
    query = "test query"
    gl_version = "15.0.0"

    mock_search_service_client.search.side_effect = NotFound("not found")

    vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

    with pytest.raises(DataStoreNotFound):
        await vertex_search.search(query, gl_version)


class TestVertexAIDumpResults:
    """Test the dump_results override in VertexAISearch."""

    def test_dump_results_groups_by_md5(
        self, mock_search_service_client, vertex_ai_search_factory
    ):
        """Test that results are grouped by md5sum."""
        vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

        results = [
            SearchResult(
                id="1",
                content="Snippet 1",
                metadata={
                    "md5sum": "abc123",
                    "source_url": "https://docs.gitlab.com/ee/foo",
                    "title": "Feature Foo",
                },
            ),
            SearchResult(
                id="2",
                content="Snippet 2",
                metadata={
                    "md5sum": "abc123",
                    "source_url": "https://docs.gitlab.com/ee/foo",
                    "title": "Feature Foo",
                },
            ),
        ]

        dumped = vertex_search.dump_results(results)

        assert len(dumped) == 1
        assert dumped[0]["source_url"] == "https://docs.gitlab.com/ee/foo"
        assert dumped[0]["source_title"] == "Feature Foo"
        assert len(dumped[0]["relevant_snippets"]) == 2
        assert "Snippet 1" in dumped[0]["relevant_snippets"]
        assert "Snippet 2" in dumped[0]["relevant_snippets"]

    def test_dump_results_handles_multiple_pages(
        self, mock_search_service_client, vertex_ai_search_factory
    ):
        """Test handling of results from multiple pages."""
        vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

        results = [
            SearchResult(
                id="1",
                content="Content from page 1",
                metadata={
                    "md5sum": "page1",
                    "source_url": "https://docs.gitlab.com/ee/page1",
                    "title": "Page 1",
                },
            ),
            SearchResult(
                id="2",
                content="Content from page 2",
                metadata={
                    "md5sum": "page2",
                    "source_url": "https://docs.gitlab.com/ee/page2",
                    "title": "Page 2",
                },
            ),
        ]

        dumped = vertex_search.dump_results(results)

        assert len(dumped) == 2
        assert dumped[0]["source_title"] == "Page 1"
        assert dumped[1]["source_title"] == "Page 2"

    def test_dump_results_empty_list(
        self, mock_search_service_client, vertex_ai_search_factory
    ):
        """Test handling empty results."""
        vertex_search = vertex_ai_search_factory(client=mock_search_service_client)

        dumped = vertex_search.dump_results([])

        assert dumped == []
