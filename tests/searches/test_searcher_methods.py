# pylint: disable=all  # Test file for Searcher base class methods
from unittest import mock

from structlog.testing import capture_logs

from ai_gateway.searches.search import Searcher
from ai_gateway.searches.typing import SearchResult


class TestSearcherLogSearchResults:
    """Test the log_search_results method in the Searcher base class."""

    @mock.patch("ai_gateway.structured_logging.can_log_request_data", return_value=True)
    def test_log_search_results_logs_query_and_version(self, mock_can_log):
        """Test that log_search_results logs query and gl_version."""
        searcher = Searcher()
        results = [
            SearchResult(
                id="1",
                content="Test content 1",
                metadata={"source": "test"},
            ),
            SearchResult(
                id="2",
                content="Test content 2",
                metadata={"source": "test"},
            ),
        ]

        with capture_logs() as cap_logs:
            searcher.log_search_results(
                query="test query",
                page_size=10,
                gl_version="17.0.0",
                results=results,
            )

        assert len(cap_logs) == 1
        log_entry = cap_logs[0]
        assert log_entry["event"] == "Search completed"
        assert log_entry["query"] == "test query"
        assert log_entry["gl_version"] == "17.0.0"
        assert log_entry["page_size"] == 10
        assert log_entry["filtered_results"] == 2
        assert log_entry["class"] == "Searcher"

    @mock.patch("ai_gateway.structured_logging.can_log_request_data", return_value=True)
    def test_log_search_results_includes_token_count(self, mock_can_log):
        """Test that token count is logged when provided."""
        searcher = Searcher()
        results = [
            SearchResult(
                id="1",
                content="Test content",
                metadata={"source": "test"},
            ),
        ]
        token_count = 150

        with capture_logs() as cap_logs:
            searcher.log_search_results(
                query="test query",
                page_size=4,
                gl_version="17.0.0",
                results=results,
                token_count=token_count,
            )

        assert len(cap_logs) == 1
        log_entry = cap_logs[0]
        assert log_entry["total_tokens"] == 150
        assert log_entry["filtered_results"] == 1

    @mock.patch("ai_gateway.structured_logging.can_log_request_data", return_value=True)
    def test_log_search_results_empty_results(self, mock_can_log):
        """Test logging empty search results."""
        searcher = Searcher()
        results = []

        with capture_logs() as cap_logs:
            searcher.log_search_results(
                query="test query",
                page_size=10,
                gl_version="17.0.0",
                results=results,
            )

        assert len(cap_logs) == 1
        log_entry = cap_logs[0]
        assert log_entry["event"] == "Search completed"
        assert log_entry["filtered_results"] == 0
        assert log_entry["results_metadata"] == []

    @mock.patch("ai_gateway.structured_logging.can_log_request_data", return_value=True)
    def test_log_search_results_preserves_metadata(self, mock_can_log):
        """Test that results metadata is logged correctly."""
        searcher = Searcher()
        metadata = {
            "source": "GitLab Docs",
            "source_url": "https://docs.gitlab.com/ee/foo",
        }
        results = [
            SearchResult(
                id="1",
                content="Test content",
                metadata=metadata,
            ),
        ]

        with capture_logs() as cap_logs:
            searcher.log_search_results(
                query="test query",
                page_size=4,
                gl_version="17.0.0",
                results=results,
            )

        assert len(cap_logs) == 1
        log_entry = cap_logs[0]
        assert log_entry["results_metadata"] == [metadata]


class TestSearcherDumpResults:
    """Test the dump_results method in the Searcher base class."""

    def test_dump_results_single_result(self):
        """Test dumping a single search result."""
        searcher = Searcher()
        results = [
            SearchResult(
                id="1",
                content="Test content",
                metadata={"source": "test", "title": "Test"},
            ),
        ]

        dumped = searcher.dump_results(results)

        assert len(dumped) == 1
        assert dumped[0]["id"] == "1"
        assert dumped[0]["content"] == "Test content"
        assert dumped[0]["metadata"]["source"] == "test"

    def test_dump_results_multiple_results(self):
        """Test dumping multiple search results."""
        searcher = Searcher()
        results = [
            SearchResult(
                id="1",
                content="Content 1",
                metadata={"source": "test"},
            ),
            SearchResult(
                id="2",
                content="Content 2",
                metadata={"source": "test"},
            ),
        ]

        dumped = searcher.dump_results(results)

        assert len(dumped) == 2
        assert dumped[0]["id"] == "1"
        assert dumped[1]["id"] == "2"

    def test_dump_results_empty_results(self):
        """Test dumping empty search results."""
        searcher = Searcher()
        results = []

        dumped = searcher.dump_results(results)

        assert dumped == []

    def test_dump_results_preserves_metadata(self):
        """Test that dumping preserves all metadata."""
        searcher = Searcher()
        metadata = {
            "source": "GitLab Docs",
            "source_url": "https://docs.gitlab.com/ee/foo",
            "title": "Test Title",
            "version": "17.0.0",
        }
        results = [
            SearchResult(
                id="1",
                content="Test content",
                metadata=metadata,
            ),
        ]

        dumped = searcher.dump_results(results)

        assert dumped[0]["metadata"] == metadata
