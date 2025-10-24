import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.findings.get_security_finding_details import (
    GetSecurityFindingDetails,
    GetSecurityFindingDetailsInput,
)

# editorconfig-checker-disable
GET_SECURITY_FINDINGS_JSON = """
{
  "data": {
    "project": {
      "id": "gid://gitlab/Project/26",
      "webUrl": "http://gdk.test:3000/gitlab-duo/myproject",
      "nameWithNamespace": "GitLab Duo / Myproject",
      "pipeline": {
        "id": "gid://gitlab/Ci::Pipeline/3886",
        "iid": "273",
        "sha": "b791334f7a9e72ba8796002c1ef7573d50c17676",
        "ref": "security/sast-fix-773-173",
        "status": "SUCCESS",
        "createdAt": "2025-09-24T17:52:52Z",
        "securityReportFinding": {
          "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
          "title": "Improper neutralization of special elements used in an OS command ('OS Command Injection')",
          "state": "DETECTED",
          "severity": "HIGH",
          "reportType": "SAST",
          "falsePositive": false,
          "aiResolutionEnabled": true,
          "aiResolutionAvailable": true,
          "vulnerability": null,
          "dismissedAt": null
        }
      }
    }
  }
}
"""
# editorconfig-checker-enable


@pytest.fixture
def gitlab_client_mock():
    """Fixture for a mocked GitLab client."""
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    """Fixture for tool metadata containing the mocked client."""
    return {"gitlab_client": gitlab_client_mock}


@pytest.fixture
def security_finding_response_data():
    """Fixture for a successful security finding GraphQL response."""
    return json.loads(GET_SECURITY_FINDINGS_JSON)


@pytest.mark.asyncio
class TestGetSecurityFindingDetails:
    """Tests for the GetSecurityFindingDetails tool."""

    async def test_arun_with_pipeline_id_success(
        self, gitlab_client_mock, metadata, security_finding_response_data
    ):
        """Test successful finding retrieval when pipeline_id is provided."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=security_finding_response_data
        )
        tool = GetSecurityFindingDetails(metadata=metadata)
        input_data = {
            "project_full_path": "gitlab-duo/myproject",
            "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
            "pipeline_id": "273",
        }
        response_str = await tool.arun(input_data)
        response = json.loads(response_str)

        assert "error" not in response
        assert "finding" in response
        assert response["finding"]["uuid"] == input_data["uuid"]
        assert response["pipeline_context"]["id"] == "gid://gitlab/Ci::Pipeline/3886"
        assert (
            response["project_context"]["nameWithNamespace"] == "GitLab Duo / Myproject"
        )
        assert response["metadata"]["is_promoted"] is False
        assert response["metadata"]["is_dismissed"] is False
        assert response["metadata"]["ai_resolution_available"] is True

        gitlab_client_mock.apost.assert_called_once()
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["projectFullPath"] == "gitlab-duo/myproject"
        assert call_body["variables"]["pipelineId"] == "gid://gitlab/Ci::Pipeline/273"
        assert call_body["variables"]["findingUuid"] == input_data["uuid"]

    async def test_arun_with_pipeline_id_finding_not_found(
        self, gitlab_client_mock, metadata
    ):
        """Test case where the finding is not found in the specified pipeline."""
        mock_response = {
            "data": {"project": {"pipeline": {"securityReportFinding": None}}}
        }
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "uuid": "not-found-uuid",
                "pipeline_id": "273",
            }
        )
        error_response = json.loads(response)
        assert "error" in error_response
        assert (
            "Security finding not found in the specified pipeline"
            in error_response["error"]
        )
        assert error_response["uuid"] == "not-found-uuid"
        assert error_response["pipeline_id"] == "gid://gitlab/Ci::Pipeline/273"

    async def test_arun_with_pipeline_id_pipeline_not_found(
        self, gitlab_client_mock, metadata
    ):
        """Test case where the specified pipeline is not found."""
        mock_response = {"data": {"project": {"pipeline": None}}}
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "uuid": "some-uuid",
                "pipeline_id": "999",
            }
        )
        error_response = json.loads(response)
        assert "error" in error_response
        assert "Pipeline not found" in error_response["error"]

    async def test_arun_exception(self, gitlab_client_mock, metadata):
        """Test handling of a generic exception during API call."""
        gitlab_client_mock.apost.side_effect = Exception("Network Error")
        tool = GetSecurityFindingDetails(metadata=metadata)

        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "some-uuid",
                    "pipeline_id": "123",
                }
            )

        assert (
            "An unexpected error occurred while fetching the security finding: Network Error"
            in str(exc_info.value)
        )

    async def test_arun_with_int_pipeline_id(
        self, gitlab_client_mock, metadata, security_finding_response_data
    ):
        """Test successful finding retrieval when pipeline_id is provided as int."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=security_finding_response_data
        )
        tool = GetSecurityFindingDetails(metadata=metadata)
        input_data = {
            "project_full_path": "gitlab-duo/myproject",
            "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
            "pipeline_id": 273,
        }
        response_str = await tool.arun(input_data)
        response = json.loads(response_str)

        assert "error" not in response
        assert "finding" in response

        gitlab_client_mock.apost.assert_called_once()
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["pipelineId"] == "gid://gitlab/Ci::Pipeline/273"

    async def test_arun_with_gid_pipeline_id(
        self, gitlab_client_mock, metadata, security_finding_response_data
    ):
        """Test successful finding retrieval when pipeline_id is provided as GID."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=security_finding_response_data
        )
        tool = GetSecurityFindingDetails(metadata=metadata)
        input_data = {
            "project_full_path": "gitlab-duo/myproject",
            "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
            "pipeline_id": "gid://gitlab/Ci::Pipeline/273",
        }
        response_str = await tool.arun(input_data)
        response = json.loads(response_str)

        assert "error" not in response
        assert "finding" in response

        gitlab_client_mock.apost.assert_called_once()
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["pipelineId"] == "gid://gitlab/Ci::Pipeline/273"

    async def test_arun_graphql_errors(self, gitlab_client_mock, metadata):
        """Test handling of GraphQL errors in response."""
        mock_response = {
            "errors": [{"message": "Field 'securityReportFinding' doesn't exist"}]
        }
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "uuid": "some-uuid",
                "pipeline_id": "123",
            }
        )
        error_response = json.loads(response)
        assert "error" in error_response
        assert error_response["error"] == "GraphQL query failed"
        assert "errors" in error_response

    async def test_arun_project_not_found(self, gitlab_client_mock, metadata):
        """Test case where the project is not found."""
        mock_response = {"data": {"project": None}}
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "gitlab-duo/nonexistent",
                "uuid": "some-uuid",
                "pipeline_id": "123",
            }
        )
        error_response = json.loads(response)
        assert "error" in error_response
        assert "Project not found or access denied" in error_response["error"]


class TestFormatDisplayMessage:
    """Tests for the format_display_message method."""

    def test_format_display_message_with_int_pipeline_id(self):
        """Test display message when pipeline_id is provided as int."""
        tool = GetSecurityFindingDetails(metadata={})
        args = GetSecurityFindingDetailsInput(
            project_full_path="group/project",
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            pipeline_id=12345,
        )
        message = tool.format_display_message(args)
        assert (
            message
            == "Get details for security finding 1e9a2bf7... from pipeline 12345"
        )

    def test_format_display_message_with_string_pipeline_id(self):
        """Test display message when pipeline_id is provided as string."""
        tool = GetSecurityFindingDetails(metadata={})
        args = GetSecurityFindingDetailsInput(
            project_full_path="group/project",
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            pipeline_id="12345",
        )
        message = tool.format_display_message(args)
        assert (
            message
            == "Get details for security finding 1e9a2bf7... from pipeline 12345"
        )

    def test_format_display_message_with_gid_pipeline_id(self):
        """Test display message when pipeline_id is provided as GID."""
        tool = GetSecurityFindingDetails(metadata={})
        args = GetSecurityFindingDetailsInput(
            project_full_path="group/project",
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            pipeline_id="gid://gitlab/Ci::Pipeline/12345",
        )
        message = tool.format_display_message(args)
        assert (
            message
            == "Get details for security finding 1e9a2bf7... from pipeline 12345"
        )
