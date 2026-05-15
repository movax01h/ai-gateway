import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.ascp.create_security_context import (
    CreateAscpSecurityContext,
    CreateAscpSecurityContextInput,
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


@pytest.fixture(name="created_security_context_data_fixture")
def created_security_context_data_fixture_func():
    """Fixture for created ASCP security context data."""
    return {
        "id": "gid://gitlab/Ascp::SecurityContext/1",
        "summary": "Security context for authentication service",
        "authenticationModel": None,
        "authorizationModel": None,
        "dataSensitivity": None,
        "scan": {"id": "gid://gitlab/Ascp::Scan/1"},
        "securityGuidelines": {
            "nodes": [
                {
                    "id": "gid://gitlab/Ascp::SecurityGuideline/1",
                    "name": "No direct DB access",
                    "operation": "READ",
                    "legitimateUse": None,
                    "securityBoundary": None,
                    "businessContext": None,
                    "severityIfViolated": "HIGH",
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_ascp_create_security_context_success(
    gitlab_client_mock,
    metadata,
    created_security_context_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": created_security_context_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        component_id="gid://gitlab/Ascp::Component/1",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[
            {
                "name": "No direct DB access",
                "operation": "READ",
                "severity_if_violated": "HIGH",
            }
        ],
        summary="Security context for authentication service",
    )

    response_json = json.loads(response)
    assert "security_context" in response_json
    assert response_json["security_context"] == created_security_context_data_fixture
    assert (
        response_json["security_context"]["id"]
        == "gid://gitlab/Ascp::SecurityContext/1"
    )

    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "ascpSecurityContextCreate" in call_args[0]
    assert call_args[1]["input"]["projectPath"] == "namespace/project"
    assert call_args[1]["input"]["componentId"] == "gid://gitlab/Ascp::Component/1"
    assert call_args[1]["input"]["scanId"] == "gid://gitlab/Ascp::Scan/1"
    assert (
        call_args[1]["input"]["summary"]
        == "Security context for authentication service"
    )


@pytest.mark.asyncio
async def test_ascp_create_security_context_camelcase_conversion(
    gitlab_client_mock,
    metadata,
    created_security_context_data_fixture,
):
    """Verify snake_case guideline fields are converted to camelCase in the GraphQL call."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": created_security_context_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    await tool._arun(
        project_path="namespace/project",
        component_id="gid://gitlab/Ascp::Component/1",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[
            {
                "name": "No direct DB access",
                "operation": "READ",
                "legitimate_use": "Only ORM queries",
                "security_boundary": "Must use service layer",
                "business_context": "Prevents SQL injection",
                "severity_if_violated": "CRITICAL",
            }
        ],
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    guideline = call_args[1]["input"]["guidelines"][0]
    assert guideline["name"] == "No direct DB access"
    assert guideline["operation"] == "READ"
    assert guideline["legitimateUse"] == "Only ORM queries"
    assert guideline["securityBoundary"] == "Must use service layer"
    assert guideline["businessContext"] == "Prevents SQL injection"
    assert guideline["severityIfViolated"] == "CRITICAL"
    # snake_case keys must NOT appear
    assert "legitimate_use" not in guideline
    assert "security_boundary" not in guideline
    assert "business_context" not in guideline
    assert "severity_if_violated" not in guideline


@pytest.mark.asyncio
async def test_ascp_create_security_context_multiple_guidelines(
    gitlab_client_mock,
    metadata,
    created_security_context_data_fixture,
):
    """All guidelines in the list are converted and sent to the API."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": created_security_context_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    await tool._arun(
        project_path="namespace/project",
        component_id="gid://gitlab/Ascp::Component/1",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[
            {
                "name": "No direct DB access",
                "operation": "READ",
                "severity_if_violated": "HIGH",
            },
            {
                "name": "Rate limit all writes",
                "operation": "WRITE",
                "severity_if_violated": "CRITICAL",
            },
            {"name": "Audit deletes", "operation": "DELETE"},
        ],
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    guidelines = call_args[1]["input"]["guidelines"]
    assert len(guidelines) == 3
    assert guidelines[0]["name"] == "No direct DB access"
    assert guidelines[0]["severityIfViolated"] == "HIGH"
    assert guidelines[1]["name"] == "Rate limit all writes"
    assert guidelines[1]["severityIfViolated"] == "CRITICAL"
    assert guidelines[2]["name"] == "Audit deletes"
    # default severity_if_violated="MEDIUM" is always sent
    assert guidelines[2]["severityIfViolated"] == "MEDIUM"


@pytest.mark.asyncio
async def test_ascp_create_security_context_default_severity(
    gitlab_client_mock,
    metadata,
    created_security_context_data_fixture,
):
    """When severity_if_violated is omitted, the default MEDIUM is sent in the payload."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": created_security_context_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    await tool._arun(
        project_path="namespace/project",
        component_id="gid://gitlab/Ascp::Component/1",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[{"name": "No direct DB access", "operation": "READ"}],
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    guideline = call_args[1]["input"]["guidelines"][0]
    assert guideline["severityIfViolated"] == "MEDIUM"
    assert "legitimate_use" not in guideline
    assert "security_boundary" not in guideline
    assert "business_context" not in guideline


@pytest.mark.asyncio
async def test_ascp_create_security_context_with_optional_fields(
    gitlab_client_mock,
    metadata,
    created_security_context_data_fixture,
):
    security_context_data = {
        **created_security_context_data_fixture,
        "authenticationModel": "OAuth2",
        "authorizationModel": "RBAC",
        "dataSensitivity": "HIGH",
    }
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": security_context_data,
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        component_id="gid://gitlab/Ascp::Component/1",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[{"name": "No direct DB access", "operation": "READ"}],
        authentication_model="OAuth2",
        authorization_model="RBAC",
        data_sensitivity="HIGH",
    )

    response_json = json.loads(response)
    assert response_json["security_context"]["authenticationModel"] == "OAuth2"
    assert response_json["security_context"]["authorizationModel"] == "RBAC"
    assert response_json["security_context"]["dataSensitivity"] == "HIGH"

    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["input"]["authenticationModel"] == "OAuth2"
    assert call_args[1]["input"]["authorizationModel"] == "RBAC"
    assert call_args[1]["input"]["dataSensitivity"] == "HIGH"


@pytest.mark.asyncio
async def test_ascp_create_security_context_graphql_top_level_errors(
    gitlab_client_mock,
    metadata,
):
    """Top-level GraphQL errors (e.g. auth failures) raise ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"errors": [{"message": "Unauthorized"}]},
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    assert "Unauthorized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_security_context_response_without_key(
    gitlab_client_mock,
    metadata,
):
    """When response has no ascpSecurityContextCreate key and no top-level errors, raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(return_value={})

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    assert "Failed to create ASCP security context" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_security_context_mutation_errors(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": None,
                "errors": ["Component not found"],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/999",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    assert "Component not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_security_context_multiple_errors(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns multiple errors, they are joined in ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": None,
                "errors": ["Error one", "Error two"],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    error_msg = str(exc_info.value)
    assert "Error one" in error_msg
    assert "Error two" in error_msg


@pytest.mark.asyncio
async def test_ascp_create_security_context_exception(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=ConnectionError("Network failure"),
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    # Exceptions propagate directly
    with pytest.raises(ConnectionError, match="Network failure"):
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )


@pytest.mark.asyncio
async def test_ascp_create_security_context_malformed_response(
    gitlab_client_mock,
    metadata,
):
    """Non-dict response (e.g. None) raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(return_value=None)

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    assert "no response or invalid format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ascp_create_security_context_missing_id(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns security_context without id, tool raises ToolException."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpSecurityContextCreate": {
                "securityContext": {"summary": "partial"},
                "errors": [],
            },
        },
    )

    tool = CreateAscpSecurityContext(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_path="namespace/project",
            component_id="gid://gitlab/Ascp::Component/1",
            scan_id="gid://gitlab/Ascp::Scan/1",
            guidelines=[{"name": "Test", "operation": "READ"}],
        )

    assert "Failed to create ASCP security context" in str(exc_info.value)


def test_ascp_create_security_context_format_display_message():
    """Test format_display_message returns expected string."""
    tool = CreateAscpSecurityContext(metadata={})
    input_data = CreateAscpSecurityContextInput(
        project_path="my-group/my-project",
        component_id="gid://gitlab/Ascp::Component/42",
        scan_id="gid://gitlab/Ascp::Scan/1",
        guidelines=[{"name": "Test", "operation": "READ"}],
    )
    expected_message = (
        "Create ASCP security context for component gid://gitlab/Ascp::Component/42"
    )
    assert tool.format_display_message(input_data) == expected_message
