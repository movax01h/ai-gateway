import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.ascp.list_components import (
    DEFAULT_PAGE_SIZE,
    ListAscpComponents,
    ListAscpComponentsInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    mock.graphql = AsyncMock()
    return mock


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


def _make_component(component_id: int, title: str, sub_directory: str) -> dict:
    return {
        "id": f"gid://gitlab/Ascp::Component/{component_id}",
        "title": title,
        "description": None,
        "subDirectory": sub_directory,
        "expectedUserBehavior": None,
        "scan": {
            "id": "gid://gitlab/Ascp::Scan/1",
            "scanSequence": 1,
            "scanType": "FULL",
        },
        "securityContext": None,
        "createdAt": "2025-02-19T10:00:00.000Z",
        "updatedAt": "2025-02-19T10:00:00.000Z",
    }


@pytest.mark.asyncio
async def test_ascp_list_components_success(
    gitlab_client_mock,
    metadata,
):
    components = [
        _make_component(1, "Authentication Service", "services/auth"),
        _make_component(2, "Payment Service", "services/payment"),
    ]
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "project": {
                "ascpComponents": {
                    "nodes": components,
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        },
    )

    tool = ListAscpComponents(metadata=metadata)

    response = await tool._arun(project_path="namespace/project")

    response_json = json.loads(response)
    assert "components" in response_json
    assert "page_info" in response_json
    assert len(response_json["components"]) == 2
    assert response_json["components"][0]["title"] == "Authentication Service"
    assert response_json["page_info"]["has_next_page"] is False

    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "listAscpComponents" in call_args[0]
    assert call_args[1]["projectPath"] == "namespace/project"
    assert call_args[1]["first"] == DEFAULT_PAGE_SIZE
    assert "title" not in call_args[1]
    assert "subDirectory" not in call_args[1]


@pytest.mark.asyncio
async def test_ascp_list_components_pagination_cursor(
    gitlab_client_mock,
    metadata,
):
    """Cursor returned in page_info.end_cursor can be passed as after for next page."""
    components = [_make_component(1, "Auth Service", "services/auth")]
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "project": {
                "ascpComponents": {
                    "nodes": components,
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor123"},
                }
            }
        },
    )

    tool = ListAscpComponents(metadata=metadata)

    response = await tool._arun(project_path="namespace/project", first=10)

    response_json = json.loads(response)
    assert response_json["page_info"]["has_next_page"] is True
    assert response_json["page_info"]["end_cursor"] == "cursor123"

    # Caller passes end_cursor as after to fetch next page
    await tool._arun(project_path="namespace/project", first=10, after="cursor123")
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["after"] == "cursor123"
    assert call_args[1]["first"] == 10


@pytest.mark.asyncio
async def test_ascp_list_components_with_filters(
    gitlab_client_mock,
    metadata,
):
    """Title and sub_directory filters are passed through to GraphQL."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "project": {
                "ascpComponents": {
                    "nodes": [_make_component(1, "Auth Service", "services/auth")],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        },
    )

    tool = ListAscpComponents(metadata=metadata)

    await tool._arun(
        project_path="namespace/project",
        title="Auth",
        sub_directory="services/auth",
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["title"] == "Auth"
    assert call_args[1]["subDirectory"] == "services/auth"


@pytest.mark.asyncio
async def test_ascp_list_components_empty_result(
    gitlab_client_mock,
    metadata,
):
    """Empty nodes list returns empty components array."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "project": {
                "ascpComponents": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        },
    )

    tool = ListAscpComponents(metadata=metadata)

    response = await tool._arun(project_path="namespace/project")

    response_json = json.loads(response)
    assert response_json["components"] == []


@pytest.mark.asyncio
async def test_ascp_list_components_project_not_found(
    gitlab_client_mock,
    metadata,
):
    """When project is null, tool raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"project": None},
    )

    tool = ListAscpComponents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_path="namespace/nonexistent")

    assert "not found or not accessible" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_list_components_graphql_top_level_errors(
    gitlab_client_mock,
    metadata,
):
    """Top-level GraphQL errors (e.g. auth failures) raise ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "errors": [
                {"message": "Unauthorized"},
                {"message": "Feature not enabled"},
            ]
        },
    )

    tool = ListAscpComponents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_path="namespace/project")

    error_msg = str(exc_info.value)
    assert "Unauthorized" in error_msg
    assert "Feature not enabled" in error_msg


@pytest.mark.asyncio
async def test_ascp_list_components_exception(
    gitlab_client_mock,
    metadata,
):
    """Network errors propagate directly."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=ConnectionError("Network failure"),
    )

    tool = ListAscpComponents(metadata=metadata)

    # Exceptions propagate directly
    with pytest.raises(ConnectionError, match="Network failure"):
        await tool._arun(project_path="namespace/project")


@pytest.mark.asyncio
async def test_ascp_list_components_malformed_response(
    gitlab_client_mock,
    metadata,
):
    """Non-dict response raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(return_value=None)

    tool = ListAscpComponents(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_path="namespace/project")

    assert "no response or invalid format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_list_components_full_request_and_response(
    gitlab_client_mock,
    metadata,
):
    """Full round-trip: send all query variables, receive full response shape."""
    project_path = "my-group/my-project"
    title = "Auth"
    sub_directory = "services/auth"
    first = 10
    after = "cursor123"

    full_response = {
        "project": {
            "ascpComponents": {
                "nodes": [
                    {
                        "id": "gid://gitlab/Ascp::Component/1",
                        "title": "Authentication Service",
                        "description": "Handles auth",
                        "subDirectory": "services/auth",
                        "expectedUserBehavior": None,
                        "scan": {
                            "id": "gid://gitlab/Ascp::Scan/1",
                            "scanSequence": 1,
                            "scanType": "FULL",
                        },
                        "securityContext": {
                            "id": "gid://gitlab/Ascp::SecurityContext/1",
                            "summary": "sec",
                        },
                        "createdAt": "2025-02-19T10:00:00.000Z",
                        "updatedAt": "2025-02-19T10:00:00.000Z",
                    }
                ],
                "pageInfo": {"hasNextPage": True, "endCursor": "next_cursor_xyz"},
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=full_response)

    tool = ListAscpComponents(metadata=metadata)
    response = await tool._arun(
        project_path=project_path,
        title=title,
        sub_directory=sub_directory,
        first=first,
        after=after,
    )

    gitlab_client_mock.graphql.assert_called_once()
    variables = gitlab_client_mock.graphql.call_args[0][1]
    assert variables["projectPath"] == project_path
    assert variables["title"] == title
    assert variables["subDirectory"] == sub_directory
    assert variables["first"] == first
    assert variables["after"] == after

    response_json = json.loads(response)
    expected_nodes = full_response["project"]["ascpComponents"]["nodes"]
    expected_page_info = full_response["project"]["ascpComponents"]["pageInfo"]
    assert response_json["components"] == expected_nodes
    assert (
        response_json["page_info"]["has_next_page"] == expected_page_info["hasNextPage"]
    )
    assert response_json["page_info"]["end_cursor"] == expected_page_info["endCursor"]


def test_ascp_list_components_format_display_message():
    """Test format_display_message returns expected string."""
    tool = ListAscpComponents(metadata={})
    input_data = ListAscpComponentsInput(project_path="my-group/my-project")
    expected_message = "List ASCP components for my-group/my-project"
    assert tool.format_display_message(input_data) == expected_message
