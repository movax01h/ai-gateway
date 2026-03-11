import json
import os.path
from unittest.mock import patch

import pytest

from ai_gateway.searches.sqlite_search import SqliteSearch


@pytest.fixture(name="mock_sqlite_search_struct_data")
def mock_sqlite_search_struct_data_fixture():
    return {
        "id": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        "metadata": {
            "Header1": "Git Large File Storage (LFS)",
            "Header2": "Add a file with Git LFS",
            "Header3": "Add a file type to Git LFS",
            "filename": "tmp/gitlab-master-doc/doc/topics/git/lfs/index.md",
        },
    }


@pytest.fixture(name="mock_sqlite_search_response")
def mock_sqlite_search_response_fixture():
    return [
        json.dumps(
            {
                "Header1": "Tutorial: Set up issue boards for team hand-off",
                "filename": "tmp/gitlab-master-doc/doc/foo/index.md",
            }
        ),
        "GitLab's mission is to make software development easier and more efficient.",
    ]


@pytest.fixture(name="sqlite_search_factory")
def sqlite_search_factory_fixture():
    def create() -> SqliteSearch:
        return SqliteSearch()

    return create


@pytest.fixture(name="mock_os_path_to_db")
def mock_os_path_to_db_fixture():
    current_dir = os.path.dirname(__file__)
    local_docs_example_path = current_dir.replace(
        "searches", "_assets/tpl/tools/searches/local_docs_example.db"
    )
    with patch("posixpath.join", return_value=local_docs_example_path) as mock:
        yield mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version",
    [
        "1.2.3",
        "10.11.12-pre",
    ],
)
@pytest.mark.usefixtures("mock_os_path_to_db")
async def test_sqlite_search(
    mock_sqlite_search_struct_data,
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    sqlite_search = sqlite_search_factory()

    result = await sqlite_search.search(query, gl_version, page_size)
    assert result[0].id == mock_sqlite_search_struct_data["id"]
    assert result[0].metadata == mock_sqlite_search_struct_data["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gl_version",
    [
        ("1.2.3"),
        ("10.11.12-pre"),
    ],
)
async def test_sqlite_search_with_no_db(
    gl_version,
    sqlite_search_factory,
):
    query = "What is lfs?"
    page_size = 4

    with patch("os.path.isfile", return_value=False):
        sqlite_search = sqlite_search_factory()

        result = await sqlite_search.search(query, gl_version, page_size)
        assert result == []


class TestSqliteSearchTokenLimiting:
    """Test token limiting functionality in SqliteSearch."""

    def test_estimate_token_count_empty(self, sqlite_search_factory):
        """Test token count for empty text."""
        sqlite_search = sqlite_search_factory()

        token_count = sqlite_search.estimate_token_count("")

        assert token_count == 0

    def test_limit_search_results_within_limit(self, sqlite_search_factory):
        """Test limiting results when within token limit."""
        sqlite_search = sqlite_search_factory()

        response = [
            {
                "id": "1",
                "content": "short content",
                "metadata": {"filename": "file1.md"},
            },
        ]

        results, token_count = sqlite_search.limit_search_results(
            response, max_tokens=1000
        )

        assert len(results) == 1
        assert results[0].id == "1"
        assert token_count > 0

    def test_limit_search_results_exceeds_limit(self, sqlite_search_factory):
        """Test limiting results when exceeding token limit."""
        sqlite_search = sqlite_search_factory()

        response = [
            {
                "id": "1",
                "content": "a " * 3000,
                "metadata": {"filename": "file1.md"},
            },
            {
                "id": "2",
                "content": "b " * 2000,
                "metadata": {"filename": "file2.md"},
            },
        ]

        results, token_count = sqlite_search.limit_search_results(
            response, max_tokens=5500
        )

        assert len(results) == 1
        assert results[0].id == "1"
        assert token_count == int(3000 * 1.4)

    def test_limit_search_results_empty_response(self, sqlite_search_factory):
        """Test limiting empty response."""
        sqlite_search = sqlite_search_factory()

        results, token_count = sqlite_search.limit_search_results([], max_tokens=8000)

        assert results == []
        assert token_count == 0

    def test_limit_search_results_multiple_results(self, sqlite_search_factory):
        """Test limiting multiple results that fit within token limit."""
        sqlite_search = sqlite_search_factory()

        response = [
            {
                "id": "1",
                "content": "content " * 100,
                "metadata": {"filename": "file1.md"},
            },
            {
                "id": "2",
                "content": "content " * 100,
                "metadata": {"filename": "file2.md"},
            },
            {
                "id": "3",
                "content": "content " * 100,
                "metadata": {"filename": "file3.md"},
            },
        ]

        results, token_count = sqlite_search.limit_search_results(
            response, max_tokens=5000
        )

        assert len(results) == 3
        assert results[0].id == "1"
        assert results[1].id == "2"
        assert results[2].id == "3"
        assert token_count <= 5000
