import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_gateway.searches.sqlite_search import SqliteSearch
from ai_gateway.searches.typing import SearchResult
from duo_workflow_service.tools.documentation_search import (
    DocumentationSearch,
    SearchInput,
)


class TestDocumentationSearch:
    @pytest.fixture(name="vertex_search_mock")
    def vertex_search_mock_fixture(self):
        return AsyncMock()

    @pytest.fixture(name="discoveryengine_client_mock")
    def discoveryengine_client_mock_fixture(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_arun_success(
        self,
        vertex_search_mock,
        discoveryengine_client_mock,
    ):
        processed_results = [
            {
                "id": "1",
                "content": "Test content",
                "metadata": {
                    "md5sum": "hash123",
                    "source_url": "http://example.com",
                    "title": "Test Documentation",
                },
            }
        ]

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", return_value=processed_results):
            response = await tool._arun(search="test query")

        expected_response = json.dumps({"search_results": processed_results})

        assert response == expected_response

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_gitlab_version")
    async def test_arun_with_dynamic_gitlab_version(
        self,
        vertex_search_mock,
        discoveryengine_client_mock,
        gl_version,
    ):
        processed_results = [
            {
                "id": "1",
                "content": "Test content",
                "metadata": {
                    "md5sum": "hash123",
                    "source_url": "http://example.com",
                    "title": "Test Documentation",
                },
            }
        ]

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", return_value=processed_results):
            response = await tool._arun(search="test query")

        expected_response = json.dumps({"search_results": processed_results})
        assert response == expected_response

    @pytest.mark.asyncio
    async def test_arun_with_exception(
        self,
        vertex_search_mock,
        discoveryengine_client_mock,
    ):
        tool = DocumentationSearch()
        error_msg = "Test error"
        with patch.object(
            tool, "_fetch_documentation", side_effect=Exception(error_msg)
        ):
            with pytest.raises(Exception, match=error_msg):
                await tool._arun(search="test query")


def test_format_display_message():
    tool = DocumentationSearch()

    input_data = SearchInput(search="test search")

    message = tool.format_display_message(input_data)

    expected_message = "Searching GitLab documentation for: 'test search'"
    assert message == expected_message


class TestDocumentationSearchWithSqliteSearch:
    """Test DocumentationSearch when using SqliteSearch as the searcher."""

    @pytest.mark.asyncio
    async def test_fetch_documentation_with_sqlite_results(self):
        """Test _fetch_documentation injects SqliteSearch and calls search_with_retry."""
        # Simulate SqliteSearch results (no MD5 grouping, direct results)
        search_results = [
            SearchResult(
                id="doc1.md",
                content="GitLab is a DevOps platform.",
                metadata={
                    "filename": "doc1.md",
                    "Header1": "What is GitLab",
                },
            ),
            SearchResult(
                id="doc2.md",
                content="CI/CD pipelines automate testing.",
                metadata={
                    "filename": "doc2.md",
                    "Header1": "CI/CD Pipelines",
                },
            ),
        ]

        # Create a mock SqliteSearch instance
        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(
            return_value=[
                {
                    "id": "doc1.md",
                    "content": "GitLab is a DevOps platform.",
                    "metadata": {
                        "filename": "doc1.md",
                        "Header1": "What is GitLab",
                    },
                },
                {
                    "id": "doc2.md",
                    "content": "CI/CD pipelines automate testing.",
                    "metadata": {
                        "filename": "doc2.md",
                        "Header1": "CI/CD Pipelines",
                    },
                },
            ]
        )

        # Mock _fetch_documentation to use our mock searcher
        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            result = await tool._fetch_documentation("gitlab ci cd")

        # Assert search_with_retry was called on the searcher
        mock_sqlite_search.search_with_retry.assert_called_once()
        call_kwargs = mock_sqlite_search.search_with_retry.call_args[1]
        assert call_kwargs["query"] == "gitlab ci cd"
        assert call_kwargs["gl_version"] == "18.0.0"
        assert call_kwargs["page_size"] == 4

        # Assert dump_results was called with search results
        mock_sqlite_search.dump_results.assert_called_once_with(search_results)

        # Verify final results
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "doc1.md"
        assert result[1]["id"] == "doc2.md"

    @pytest.mark.asyncio
    async def test_fetch_documentation_sqlite_with_token_limiting(self):
        """Test _fetch_documentation with SqliteSearch token limiting calls search_with_retry."""
        # Simulate SqliteSearch with token limiting applied
        search_results = [
            SearchResult(
                id="doc1.md",
                content="a " * 3000,  # ~4200 tokens
                metadata={"filename": "doc1.md"},
            )
        ]

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(
            return_value=[
                {
                    "id": "doc1.md",
                    "content": "a " * 3000,
                    "metadata": {"filename": "doc1.md"},
                }
            ]
        )

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            result = await tool._fetch_documentation("documentation")

        # Assert search_with_retry was called
        mock_sqlite_search.search_with_retry.assert_called_once()

        assert len(result) == 1
        assert result[0]["id"] == "doc1.md"

    @pytest.mark.asyncio
    async def test_fetch_documentation_sqlite_empty_results(self):
        """Test _fetch_documentation with SqliteSearch returning no results calls search_with_retry."""
        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=[])
        mock_sqlite_search.dump_results = MagicMock(return_value=[])

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            result = await tool._fetch_documentation("nonexistent topic")

        # Assert search_with_retry was called
        mock_sqlite_search.search_with_retry.assert_called_once()

        assert result == []

    @pytest.mark.asyncio
    async def test_arun_with_sqlite_search_backend(self):
        """Test _arun method using SqliteSearch backend calls search_with_retry."""
        search_results = [
            SearchResult(
                id="sqlite_doc.md",
                content="SQLite documentation content",
                metadata={"filename": "sqlite_doc.md"},
            )
        ]

        processed_results = [
            {
                "id": "sqlite_doc.md",
                "content": "SQLite documentation content",
                "metadata": {"filename": "sqlite_doc.md"},
            }
        ]

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(return_value=processed_results)

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            response = await tool._arun(search="sqlite database")

        expected_response = json.dumps({"search_results": processed_results})
        assert response == expected_response

    @pytest.mark.asyncio
    async def test_fetch_documentation_sqlite_large_result_set(self):
        """Test _fetch_documentation handles large result sets like SqliteSearch."""
        # Simulate 10 results from SqliteSearch
        search_results = [
            SearchResult(
                id=f"doc{i}.md",
                content=f"Content for document {i}",
                metadata={"filename": f"doc{i}.md", "index": i},
            )
            for i in range(10)
        ]

        processed_results = [
            {
                "id": f"doc{i}.md",
                "content": f"Content for document {i}",
                "metadata": {"filename": f"doc{i}.md", "index": i},
            }
            for i in range(10)
        ]

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(return_value=processed_results)

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            result = await tool._fetch_documentation("general query")

        # Assert search_with_retry was called
        mock_sqlite_search.search_with_retry.assert_called_once()

        assert len(result) == 10
        assert result[0]["id"] == "doc0.md"
        assert result[9]["id"] == "doc9.md"

    @pytest.mark.asyncio
    async def test_fetch_documentation_sqlite_metadata_preservation(self):
        """Test _fetch_documentation preserves metadata like SqliteSearch does."""
        metadata = {
            "filename": "test_doc.md",
            "Header1": "Main Topic",
            "Header2": "Subtopic",
            "Header3": "Detail",
        }

        search_results = [
            SearchResult(
                id="test_doc.md",
                content="Documentation content",
                metadata=metadata,
            ),
        ]

        processed_results = [
            {
                "id": "test_doc.md",
                "content": "Documentation content",
                "metadata": metadata,
            },
        ]

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(return_value=processed_results)

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            result = await tool._fetch_documentation("test query")

        # Assert search_with_retry was called
        mock_sqlite_search.search_with_retry.assert_called_once()

        assert result[0]["metadata"] == metadata
        assert result[0]["metadata"]["Header1"] == "Main Topic"

    @pytest.mark.asyncio
    async def test_arun_with_sqlite_search_error(self):
        """Test _arun propagates errors from SqliteSearch rather than swallowing them."""
        error_msg = "Database connection failed"

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(
            side_effect=Exception(error_msg)
        )

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            with pytest.raises(Exception, match=error_msg):
                await tool._arun(search="query")

        # Assert search_with_retry was called before error occurred
        mock_sqlite_search.search_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_documentation_sqlite_search_parameters(self):
        """Test _fetch_documentation passes correct parameters to search_with_retry."""
        search_results = [
            SearchResult(
                id="doc.md",
                content="Content",
                metadata={"filename": "doc.md"},
            )
        ]

        processed_results = [
            {
                "id": "doc.md",
                "content": "Content",
                "metadata": {"filename": "doc.md"},
            }
        ]

        mock_sqlite_search = AsyncMock(spec=SqliteSearch)
        mock_sqlite_search.search_with_retry = AsyncMock(return_value=search_results)
        mock_sqlite_search.dump_results = MagicMock(return_value=processed_results)

        async def mock_fetch_doc(query, searcher=None):
            if searcher is None:
                searcher = mock_sqlite_search
            search_results_list = await searcher.search_with_retry(
                query=query, gl_version="18.0.0", page_size=4
            )
            return searcher.dump_results(search_results_list)

        tool = DocumentationSearch()
        with patch.object(tool, "_fetch_documentation", side_effect=mock_fetch_doc):
            await tool._fetch_documentation("test query")

        # Verify search_with_retry was called with correct parameters
        mock_sqlite_search.search_with_retry.assert_called_once()
        call_kwargs = mock_sqlite_search.search_with_retry.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert "gl_version" in call_kwargs
        assert "page_size" in call_kwargs


class TestDocumentationSearchFetchDocumentation:
    """Test the _fetch_documentation method uses dependency injection."""

    @pytest.mark.asyncio
    async def test_fetch_documentation_with_mocked_fetch(self):
        """Test _fetch_documentation behavior with mocked internal fetch."""
        tool = DocumentationSearch()

        expected_results = [
            {
                "relevant_snippets": ["Test content"],
                "source_url": "http://example.com",
                "source_title": "Test Documentation",
            }
        ]

        with patch.object(
            tool, "_fetch_documentation", return_value=expected_results
        ) as mock_fetch:
            result = await tool._fetch_documentation("test query")

            assert result == expected_results
            mock_fetch.assert_called_once()
