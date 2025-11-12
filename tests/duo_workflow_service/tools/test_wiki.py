import json
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.wiki import GetWikiPage, WikiResourceInput


@pytest.fixture(name="wiki_page_data")
def wiki_page_data_fixture():
    """Fixture for common wiki page data."""
    return {
        "content": "# Home\n\nWelcome to the wiki!",
        "format": "markdown",
        "slug": "home",
        "title": "Home",
        "wiki_page_meta_id": 59,
        "encoding": "UTF-8",
    }


@pytest.fixture(name="wiki_notes_data")
def wiki_notes_data_fixture():
    """Fixture for wiki page notes/comments data."""
    return [
        {
            "id": 1,
            "body": "Great documentation!",
            "created_at": "2024-01-01T12:00:00Z",
            "author": {"id": 1, "name": "Test User"},
        },
        {
            "id": 2,
            "body": "Thanks for the update",
            "created_at": "2024-01-02T12:00:00Z",
            "author": {"id": 2, "name": "Another User"},
        },
    ]


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    return mock


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


class TestGetWikiPage:
    """Test suite for GetWikiPage tool."""

    @pytest.mark.asyncio
    async def test_get_project_wiki_page_success(
        self, gitlab_client_mock, metadata, wiki_page_data, wiki_notes_data
    ):
        """Test successfully fetching a project wiki page with notes."""
        # Mock both API calls
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_page_data,
        )
        notes_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_notes_data,
        )

        gitlab_client_mock.aget = AsyncMock(side_effect=[wiki_response, notes_response])

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            project_id="namespace/project",
            slug="home",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert "notes" in result
        assert result["wiki_page"] == wiki_page_data
        assert result["notes"] == wiki_notes_data

        # Verify API calls were made correctly
        assert gitlab_client_mock.aget.call_count == 2
        first_call = gitlab_client_mock.aget.call_args_list[0]
        second_call = gitlab_client_mock.aget.call_args_list[1]

        assert (
            first_call.kwargs["path"]
            == f"/api/v4/projects/{quote('namespace/project', safe='')}/wikis/home"
        )
        assert (
            second_call.kwargs["path"]
            == f"/api/v4/projects/{quote('namespace/project', safe='')}/wiki_pages/59/notes"
        )

    @pytest.mark.asyncio
    async def test_get_group_wiki_page_success(
        self, gitlab_client_mock, metadata, wiki_page_data, wiki_notes_data
    ):
        """Test successfully fetching a group wiki page with notes."""
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_page_data,
        )
        notes_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_notes_data,
        )

        gitlab_client_mock.aget = AsyncMock(side_effect=[wiki_response, notes_response])

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            group_id="my-group",
            slug="home",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert "notes" in result

        # Verify API calls were made to group endpoints
        assert gitlab_client_mock.aget.call_count == 2
        first_call = gitlab_client_mock.aget.call_args_list[0]
        second_call = gitlab_client_mock.aget.call_args_list[1]

        assert first_call.kwargs["path"] == "/api/v4/groups/my-group/wikis/home"
        assert (
            second_call.kwargs["path"] == "/api/v4/groups/my-group/wiki_pages/59/notes"
        )

    @pytest.mark.asyncio
    async def test_get_wiki_page_with_url_encoded_slug(
        self, gitlab_client_mock, metadata, wiki_page_data, wiki_notes_data
    ):
        """Test fetching a wiki page with URL-encoded slug (nested page)."""
        wiki_page_data_nested = {**wiki_page_data, "slug": "dir/page_name"}

        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_page_data_nested,
        )
        notes_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_notes_data,
        )

        gitlab_client_mock.aget = AsyncMock(side_effect=[wiki_response, notes_response])

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            project_id="namespace/project",
            slug="dir/page_name",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert result["wiki_page"]["slug"] == "dir/page_name"

        # Verify slug was URL-encoded in API call
        first_call = gitlab_client_mock.aget.call_args_list[0]
        assert (
            first_call.kwargs["path"]
            == f"/api/v4/projects/{quote('namespace/project', safe='')}/wikis/{quote('dir/page_name', safe='')}"
        )

    @pytest.mark.asyncio
    async def test_get_wiki_page_notes_error_returns_partial_result(
        self, gitlab_client_mock, metadata, wiki_page_data
    ):
        """Test that if notes fetch fails, wiki page is still returned."""
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_page_data,
        )
        notes_response = GitLabHttpResponse(
            status_code=404,
            body={"error": "Not found"},
        )

        gitlab_client_mock.aget = AsyncMock(side_effect=[wiki_response, notes_response])

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            project_id="namespace/project",
            slug="home",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert "notes_error" in result
        assert result["wiki_page"] == wiki_page_data
        assert result["notes_error"] == "Failed to fetch notes: HTTP 404"
        assert "notes" not in result

    @pytest.mark.asyncio
    async def test_get_wiki_page_not_found(self, gitlab_client_mock, metadata):
        """Test handling of wiki page not found (404)."""
        wiki_response = GitLabHttpResponse(
            status_code=404,
            body={"error": "Not found"},
        )

        gitlab_client_mock.aget = AsyncMock(return_value=wiki_response)

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_id="namespace/project",
                slug="nonexistent",
            )

        assert str(exc_info.value) == (
            "Failed to fetch wiki page: HTTP 404. "
            "Verify that the project_id 'namespace/project' and slug 'nonexistent' are correct."
        )

        # Notes should not be fetched if wiki page fetch fails
        assert gitlab_client_mock.aget.call_count == 1

    @pytest.mark.asyncio
    async def test_get_wiki_page_notes_exception_returns_partial_result(
        self, gitlab_client_mock, metadata, wiki_page_data
    ):
        """Test that if notes fetch raises exception, wiki page is still returned."""
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_page_data,
        )

        gitlab_client_mock.aget = AsyncMock(
            side_effect=[wiki_response, Exception("Network error")]
        )

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            project_id="namespace/project",
            slug="home",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert "notes_error" in result
        assert result["wiki_page"] == wiki_page_data
        assert result["notes_error"] == "Failed to fetch notes: Network error"

    @pytest.mark.asyncio
    async def test_get_wiki_page_exception(self, gitlab_client_mock, metadata):
        """Test handling of exceptions during wiki page fetch."""
        gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Connection error"))

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        with pytest.raises(Exception) as exc_info:
            await tool._arun(
                project_id="namespace/project",
                slug="home",
            )

        assert "Connection error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_wiki_page_string_response_parsed_successfully(
        self, gitlab_client_mock, metadata, wiki_page_data, wiki_notes_data
    ):
        """Test that string responses are parsed correctly as JSON."""
        # Return the wiki page as a JSON string instead of a dict
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body=json.dumps(wiki_page_data),
        )
        notes_response = GitLabHttpResponse(
            status_code=200,
            body=wiki_notes_data,
        )

        gitlab_client_mock.aget = AsyncMock(side_effect=[wiki_response, notes_response])

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        response = await tool._arun(
            project_id="namespace/project",
            slug="home",
        )

        result = json.loads(response)
        assert "wiki_page" in result
        assert "notes" in result
        # The wiki_page should still be the string (not parsed again)
        assert result["wiki_page"] == json.dumps(wiki_page_data)
        assert result["notes"] == wiki_notes_data

    @pytest.mark.asyncio
    async def test_get_wiki_page_invalid_json_response(
        self, gitlab_client_mock, metadata
    ):
        """Test that invalid JSON responses raise a ToolException."""
        # Return an invalid JSON string
        wiki_response = GitLabHttpResponse(
            status_code=200,
            body="Not valid JSON {{{",
        )

        gitlab_client_mock.aget = AsyncMock(return_value=wiki_response)

        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_id="namespace/project",
                slug="home",
            )

        error_message = str(exc_info.value)
        assert error_message.startswith("Failed to parse wiki page response as JSON:")

    @pytest.mark.asyncio
    async def test_validation_error_no_resource_id(
        self, gitlab_client_mock, metadata
    ):  # pylint: disable=unused-argument
        """Test validation error when neither project_id nor group_id is provided."""
        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(slug="home")

        assert (
            str(exc_info.value) == "Either 'project_id' or 'group_id' must be provided"
        )

    @pytest.mark.asyncio
    async def test_validation_error_both_project_and_group_id(
        self, gitlab_client_mock, metadata
    ):  # pylint: disable=unused-argument
        """Test validation error when both project_id and group_id are provided."""
        tool = GetWikiPage(description="Get wiki page", metadata=metadata)

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_id="namespace/project",
                group_id="my-group",
                slug="home",
            )

        assert (
            str(exc_info.value)
            == "Only one of 'project_id' or 'group_id' should be provided, not both"
        )

    @pytest.mark.asyncio
    async def test_format_display_message_with_project_id(self, metadata):
        """Test format_display_message with project_id."""
        tool = GetWikiPage(description="Get wiki page", metadata=metadata)
        args = WikiResourceInput(project_id="namespace/project", slug="home")

        message = tool.format_display_message(args)
        assert message == "Read wiki page 'home' in project namespace/project"

    @pytest.mark.asyncio
    async def test_format_display_message_with_group_id(self, metadata):
        """Test format_display_message with group_id."""
        tool = GetWikiPage(description="Get wiki page", metadata=metadata)
        args = WikiResourceInput(group_id="my-group", slug="documentation")

        message = tool.format_display_message(args)
        assert message == "Read wiki page 'documentation' in group my-group"
