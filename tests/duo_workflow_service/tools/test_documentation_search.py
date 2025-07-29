import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.interceptors.gitlab_version_interceptor import gitlab_version
from duo_workflow_service.tools.documentation_search import (
    DocumentationSearch,
    SearchInput,
    _get_env_var,
)


class TestDocumentationSearch:
    @pytest.fixture(name="vertex_search_mock")
    def vertex_search_mock_fixture(self):
        return AsyncMock()

    @pytest.fixture(name="discoveryengine_client_mock")
    def discoveryengine_client_mock_fixture(self):
        return AsyncMock()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.tools.documentation_search.discoveryengine.SearchServiceAsyncClient"
    )
    @patch("duo_workflow_service.tools.documentation_search.VertexAISearch")
    @patch.dict(
        os.environ,
        {
            "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-project",
            "AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION": "17.0.0",
        },
    )
    async def test_arun_success(
        self,
        mock_vertex_search_class,
        mock_client_class,
        vertex_search_mock,
        discoveryengine_client_mock,
    ):
        mock_client_class.return_value = discoveryengine_client_mock
        mock_vertex_search_class.return_value = vertex_search_mock

        search_results = [{"id": 1, "title": "Test Documentation"}]
        vertex_search_mock.search_with_retry.return_value = search_results

        tool = DocumentationSearch()

        response = await tool._arun(search="test query")

        expected_response = json.dumps({"search_results": search_results})
        assert response == expected_response

        # Verify VertexAISearch was instantiated correctly
        mock_vertex_search_class.assert_called_once_with(
            client=discoveryengine_client_mock,
            project="test-project",
            fallback_datastore_version="17.0.0",
        )

        # Verify search was called with correct parameters (default version)
        vertex_search_mock.search_with_retry.assert_called_once_with(
            query="test query",
            gl_version="18.0.0",
            page_size=4,
        )

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.tools.documentation_search.discoveryengine.SearchServiceAsyncClient"
    )
    @patch("duo_workflow_service.tools.documentation_search.VertexAISearch")
    @patch.dict(
        os.environ,
        {
            "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-project",
            "AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION": "17.0.0",
        },
    )
    async def test_arun_with_dynamic_gitlab_version(
        self,
        mock_vertex_search_class,
        mock_client_class,
        vertex_search_mock,
        discoveryengine_client_mock,
    ):
        mock_client_class.return_value = discoveryengine_client_mock
        mock_vertex_search_class.return_value = vertex_search_mock

        search_results = [{"id": 1, "title": "Test Documentation"}]
        vertex_search_mock.search_with_retry.return_value = search_results

        # Set GitLab version in context
        gitlab_version.set("17.5.2")

        tool = DocumentationSearch()

        response = await tool._arun(search="test query")

        expected_response = json.dumps({"search_results": search_results})
        assert response == expected_response

        vertex_search_mock.search_with_retry.assert_called_once_with(
            query="test query",
            gl_version="17.5.2",
            page_size=4,
        )

        gitlab_version.set(None)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.tools.documentation_search.discoveryengine.SearchServiceAsyncClient"
    )
    @patch("duo_workflow_service.tools.documentation_search.VertexAISearch")
    @patch.dict(os.environ, {}, clear=True)
    async def test_arun_with_exception(
        self,
        mock_vertex_search_class,
        mock_client_class,
    ):
        tool = DocumentationSearch()

        response = await tool._arun(search="test query")

        expected_response = json.dumps(
            {
                "error": "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT environment variable is not set"
            }
        )
        assert response == expected_response


class TestGetEnvVar:
    @patch.dict(os.environ, {"TEST_VAR": "test_value"})
    def test_get_env_var_success(self):
        result = _get_env_var("TEST_VAR")
        assert result == "test_value"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_var_missing_raises_runtime_error(self):
        with pytest.raises(
            RuntimeError,
            match="AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT environment variable is not set",
        ):
            _get_env_var("AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT")

    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_var_missing_generic_var(self):
        with pytest.raises(
            RuntimeError, match="MISSING_VAR environment variable is not set"
        ):
            _get_env_var("MISSING_VAR")


def test_format_display_message():
    tool = DocumentationSearch()

    input_data = SearchInput(search="test search")

    message = tool.format_display_message(input_data)

    expected_message = "Searching GitLab documentation for: 'test search'"
    assert message == expected_message
