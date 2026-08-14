# pylint: disable=file-naming-for-tests
import pytest

from duo_workflow_service.graphql_queries import TOOLS_QUERIES_DIR, load_graphql_query


def test_loads_query_file_verbatim():
    query = load_graphql_query(str(TOOLS_QUERIES_DIR / "set_reviewers.graphql"))

    assert "mergeRequestSetReviewers" in query


def test_missing_file_names_the_path_it_looked_for(tmp_path):
    # The queries are resolved from a packaged directory, so a typo or a .graphql
    # file left out of the build surfaces here. Name the path so it is obvious which.
    missing = tmp_path / "does_not_exist.graphql"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_graphql_query(str(missing))

    assert str(missing) in str(exc_info.value)
