import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.ascp.create_component import (
    CreateAscpComponent,
    CreateAscpComponentInput,
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


@pytest.fixture(name="created_component_data_fixture")
def created_component_data_fixture_func():
    """Fixture for created ASCP component data."""
    return {
        "id": "gid://gitlab/Ascp::Component/1",
        "title": "Authentication Service",
        "description": None,
        "subDirectory": "services/auth",
        "expectedUserBehavior": None,
        "scan": {"id": "gid://gitlab/Ascp::Scan/1"},
        "createdAt": "2025-02-19T10:00:00.000Z",
        "updatedAt": "2025-02-19T10:00:00.000Z",
    }


@pytest.mark.asyncio
async def test_ascp_create_component_success(
    gitlab_client_mock,
    metadata,
    created_component_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpComponentCreate": {
                "component": created_component_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpComponent(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        title="Authentication Service",
        sub_directory="services/auth",
        scan_id="gid://gitlab/Ascp::Scan/1",
    )

    response_json = json.loads(response)
    assert "component" in response_json
    assert response_json["component"] == created_component_data_fixture
    assert response_json["component"]["title"] == "Authentication Service"
    assert response_json["component"]["subDirectory"] == "services/auth"

    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "ascpComponentCreate" in call_args[0]
    assert call_args[1]["input"]["projectPath"] == "namespace/project"
    assert call_args[1]["input"]["title"] == "Authentication Service"
    assert call_args[1]["input"]["subDirectory"] == "services/auth"
    assert call_args[1]["input"]["scanId"] == "gid://gitlab/Ascp::Scan/1"
    assert "description" not in call_args[1]["input"]
    assert "expectedUserBehavior" not in call_args[1]["input"]


@pytest.mark.asyncio
async def test_ascp_create_component_with_optional_fields(
    gitlab_client_mock,
    metadata,
    created_component_data_fixture,
):
    component_data = {
        **created_component_data_fixture,
        "description": "Handles user authentication",
        "expectedUserBehavior": "Users log in via OAuth",
    }
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpComponentCreate": {
                "component": component_data,
                "errors": [],
            },
        },
    )

    tool = CreateAscpComponent(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        title="Authentication Service",
        sub_directory="services/auth",
        scan_id="gid://gitlab/Ascp::Scan/1",
        description="Handles user authentication",
        expected_user_behavior="Users log in via OAuth",
    )

    response_json = json.loads(response)
    assert response_json["component"]["description"] == "Handles user authentication"
    assert (
        response_json["component"]["expectedUserBehavior"] == "Users log in via OAuth"
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["input"]["description"] == "Handles user authentication"
    assert call_args[1]["input"]["expectedUserBehavior"] == "Users log in via OAuth"


@pytest.mark.asyncio
async def test_ascp_create_component_graphql_top_level_errors(
    gitlab_client_mock,
    metadata,
):
    """Top-level GraphQL errors (e.g. auth failures) raise ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"errors": [{"message": "Unauthorized"}]},
    )

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )

    assert "Unauthorized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_component_response_without_key(
    gitlab_client_mock,
    metadata,
):
    """When response has no ascpComponentCreate key and no top-level errors, raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(return_value={})

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )

    assert "Failed to create ASCP component" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_component_mutation_errors(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpComponentCreate": {
                "component": None,
                "errors": ["Scan not found"],
            },
        },
    )

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/999",
        )

    assert "Scan not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_component_multiple_errors(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns multiple errors, they are joined in ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpComponentCreate": {
                "component": None,
                "errors": ["Error one", "Error two"],
            },
        },
    )

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )

    error_msg = str(exc_info.value)
    assert "Error one" in error_msg
    assert "Error two" in error_msg


@pytest.mark.asyncio
async def test_ascp_create_component_exception(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=ConnectionError("Network failure"),
    )

    tool = CreateAscpComponent(metadata=metadata)

    # Exceptions propagate directly
    with pytest.raises(ConnectionError, match="Network failure"):
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )


@pytest.mark.asyncio
async def test_ascp_create_component_malformed_response(
    gitlab_client_mock,
    metadata,
):
    """Non-dict response (e.g. None) raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(return_value=None)

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )

    assert "no response or invalid format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_component_missing_component_id(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns component without id, tool raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpComponentCreate": {
                "component": {"title": "Auth Service"},
                "errors": [],
            },
        },
    )

    tool = CreateAscpComponent(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            title="Auth Service",
            sub_directory="services/auth",
            scan_id="gid://gitlab/Ascp::Scan/1",
        )

    assert "Failed to create ASCP component" in str(exc_info.value)


def test_ascp_create_component_format_display_message():
    """Test format_display_message returns expected string."""
    tool = CreateAscpComponent(metadata={})
    input_data = CreateAscpComponentInput(
        project_path="my-group/my-project",
        title="Authentication Service",
        sub_directory="services/auth",
        scan_id="gid://gitlab/Ascp::Scan/1",
    )
    expected_message = (
        "Create ASCP component 'Authentication Service' in my-group/my-project"
    )
    assert tool.format_display_message(input_data) == expected_message
