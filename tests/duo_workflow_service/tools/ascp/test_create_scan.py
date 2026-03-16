# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.ascp.create_scan import (
    CreateAscpScan,
    CreateAscpScanInput,
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


@pytest.fixture(name="created_scan_data_fixture")
def created_scan_data_fixture_func():
    """Fixture for created ASCP scan data."""
    return {
        "id": "gid://gitlab/Ascp::Scan/1",
        "scanSequence": 1,
        "scanType": "FULL",
        "commitSha": "abc123def456789",
        "baseCommitSha": None,
        "baseScan": None,  # FULL scans have no base scan
        "createdAt": "2025-02-19T10:00:00.000Z",
        "updatedAt": "2025-02-19T10:00:00.000Z",
    }


@pytest.mark.asyncio
async def test_ascp_create_scan_success(
    gitlab_client_mock,
    metadata,
    created_scan_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": created_scan_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123def456789",
        scan_type="FULL",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert response_json["errors"] == []
    assert response_json["response"]["scan"] == created_scan_data_fixture
    assert response_json["response"]["scan"]["scanType"] == "FULL"
    assert response_json["response"]["scan"]["commitSha"] == "abc123def456789"

    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "ascpScanCreate" in call_args[0]
    assert call_args[1]["input"]["projectPath"] == "namespace/project"
    assert call_args[1]["input"]["commitSha"] == "abc123def456789"
    assert call_args[1]["input"]["scanType"] == "FULL"


@pytest.mark.asyncio
async def test_ascp_create_scan_incremental_with_base(
    gitlab_client_mock,
    metadata,
    created_scan_data_fixture,
):
    scan_data = {
        **created_scan_data_fixture,
        "scanType": "INCREMENTAL",
        "baseScan": {"id": "gid://gitlab/Ascp::Scan/0"},
    }
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": scan_data,
                "errors": [],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="group/project",
        commit_sha="def456",
        scan_type="INCREMENTAL",
        base_scan_id="gid://gitlab/Ascp::Scan/0",
        base_commit_sha="base789",
    )

    response_json = json.loads(response)
    assert response_json["errors"] == []
    assert response_json["response"]["scan"]["scanType"] == "INCREMENTAL"
    assert (
        response_json["response"]["scan"]["baseScan"]["id"]
        == "gid://gitlab/Ascp::Scan/0"
    )

    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["input"]["scanType"] == "INCREMENTAL"
    assert call_args[1]["input"]["baseScanId"] == "gid://gitlab/Ascp::Scan/0"
    assert call_args[1]["input"]["baseCommitSha"] == "base789"


@pytest.mark.asyncio
async def test_ascp_create_scan_response_without_ascp_scan_create(
    gitlab_client_mock,
    metadata,
):
    """When response has no ascpScanCreate (e.g. top-level errors only), tool returns generic error."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"errors": [{"message": "Unauthorized"}]},
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert response_json["errors"][0] == "Failed to create ASCP scan."
    assert response_json["response"]["raw_response"] == {}


@pytest.mark.asyncio
async def test_ascp_create_scan_mutation_errors(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": None,
                "errors": ["Invalid commit SHA"],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="invalid",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert "Invalid commit SHA" in response_json["errors"][0]


@pytest.mark.asyncio
async def test_ascp_create_scan_mutation_multiple_errors(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns multiple errors, all appear in the tool response."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": None,
                "errors": ["Error one", "Error two"],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert response_json["errors"] == ["Error one", "Error two"]


@pytest.mark.asyncio
async def test_ascp_create_scan_exception(
    gitlab_client_mock,
    metadata,
):
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=ConnectionError("Network failure"),
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert len(response_json["errors"]) == 1
    assert "ascp_create_scan" in response_json["errors"][0]
    assert "ConnectionError" in response_json["errors"][0]
    assert "Network failure" in response_json["errors"][0]


@pytest.mark.asyncio
async def test_ascp_create_scan_default_scan_type(
    gitlab_client_mock,
    metadata,
    created_scan_data_fixture,
):
    """Without scan_type, variables must include scanType: 'FULL' and call succeeds."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": created_scan_data_fixture,
                "errors": [],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123def456789",
    )

    response_json = json.loads(response)
    assert response_json["errors"] == []
    assert "response" in response_json
    assert response_json["response"]["scan"] is not None
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert call_args[1]["input"]["scanType"] == "FULL"


@pytest.mark.asyncio
async def test_ascp_create_scan_malformed_response(
    gitlab_client_mock,
    metadata,
):
    """Non-dict response (e.g. None) must return JSON with error list."""
    gitlab_client_mock.graphql = AsyncMock(return_value=None)

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert any(
        "no response or invalid format" in msg for msg in response_json["errors"]
    )


@pytest.mark.asyncio
async def test_ascp_create_scan_missing_scan_id(
    gitlab_client_mock,
    metadata,
):
    """When mutation returns scan without id, tool returns error."""
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "ascpScanCreate": {
                "scan": {"scanSequence": 1},
                "errors": [],
            },
        },
    )

    tool = CreateAscpScan(metadata=metadata)

    response = await tool._arun(
        project_path="namespace/project",
        commit_sha="abc123",
    )

    response_json = json.loads(response)
    assert "errors" in response_json
    assert "response" in response_json
    assert isinstance(response_json["errors"], list)
    assert "Failed to create ASCP scan" in response_json["errors"][0]


def test_ascp_create_scan_format_display_message():
    """Test format_display_message returns expected string."""
    tool = CreateAscpScan(metadata={})
    input_data = CreateAscpScanInput(
        project_path="my-group/my-project",
        commit_sha="abc123def456789",
    )
    expected_message = "Create ASCP scan for my-group/my-project at abc123def456789"
    assert tool.format_display_message(input_data) == expected_message
