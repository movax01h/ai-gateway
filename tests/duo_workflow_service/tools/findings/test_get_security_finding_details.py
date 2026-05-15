import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
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
      "pipelines": {
        "nodes": [
          {
            "id": "gid://gitlab/Ci::Pipeline/3886",
            "iid": "273",
            "sha": "b791334f7a9e72ba8796002c1ef7573d50c17676",
            "ref": "security/sast-fix-773-173",
            "status": "SUCCESS",
            "createdAt": "2025-09-24T17:52:52Z",
            "securityReportFinding": {
              "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
              "title": "Improper neutralization of special elements used in an OS command",
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
        ]
      }
    }
  }
}
"""
# editorconfig-checker-enable


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    """Fixture for a mocked GitLab client."""
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    """Fixture for tool metadata containing the mocked client."""
    return {"gitlab_client": gitlab_client_mock}


@pytest.fixture(name="security_finding_response_data")
def security_finding_response_data_fixture():
    """Fixture for a successful security finding GraphQL response."""
    return json.loads(GET_SECURITY_FINDINGS_JSON)


@pytest.mark.asyncio
class TestGetSecurityFindingDetails:
    """Tests for the GetSecurityFindingDetails tool."""

    async def test_arun_success(
        self, gitlab_client_mock, metadata, security_finding_response_data
    ):
        """Test successful retrieval when ref is provided."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=security_finding_response_data
        )
        tool = GetSecurityFindingDetails(metadata=metadata)
        response_str = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
                "ref": "security/sast-fix-773-173",
            }
        )
        response = json.loads(response_str)

        assert "error" not in response
        assert response["finding"]["uuid"] == "1e9a2bf7-0450-5894-8db5-895c98e39deb"
        assert response["finding"]["severity"] == "HIGH"
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
        assert call_body["variables"]["ref"] == "security/sast-fix-773-173"
        assert (
            call_body["variables"]["findingUuid"]
            == "1e9a2bf7-0450-5894-8db5-895c98e39deb"
        )

    async def test_arun_finding_not_found(self, gitlab_client_mock, metadata):
        """Test case where the UUID is not present in the pipeline."""
        mock_response = {
            "data": {
                "project": {
                    "id": "gid://gitlab/Project/26",
                    "webUrl": "http://gdk.test",
                    "nameWithNamespace": "GitLab Duo / Myproject",
                    "pipelines": {
                        "nodes": [
                            {
                                "id": "gid://gitlab/Ci::Pipeline/3886",
                                "iid": "273",
                                "sha": None,
                                "ref": "security/sast-fix-773-173",
                                "status": "SUCCESS",
                                "createdAt": None,
                                "securityReportFinding": None,
                            }
                        ]
                    },
                }
            }
        }
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "not-found-uuid",
                    "ref": "security/sast-fix-773-173",
                }
            )
        assert "Security finding not found" in str(exc_info.value)
        assert "not-found-uuid" in str(exc_info.value)
        assert "security/sast-fix-773-173" in str(exc_info.value)

    async def test_arun_no_pipeline_for_ref(self, gitlab_client_mock, metadata):
        """Test case where no pipeline is found for the given ref."""
        mock_response = {
            "data": {
                "project": {
                    "id": "gid://gitlab/Project/26",
                    "pipelines": {"nodes": []},
                }
            }
        }
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "some-uuid",
                    "ref": "nonexistent-branch",
                }
            )
        assert "No pipeline found for ref" in str(exc_info.value)

    async def test_arun_project_not_found(self, gitlab_client_mock, metadata):
        """Test case where the project is not found."""
        gitlab_client_mock.apost = AsyncMock(return_value={"data": {"project": None}})
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/nonexistent",
                    "uuid": "some-uuid",
                    "ref": "main",
                }
            )
        assert "Project not found or access denied" in str(exc_info.value)

    async def test_arun_graphql_errors(self, gitlab_client_mock, metadata):
        """Test handling of GraphQL errors in response."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "errors": [{"message": "Field 'securityReportFinding' doesn't exist"}]
            }
        )
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "some-uuid",
                    "ref": "main",
                }
            )
        assert "GraphQL query failed" in str(exc_info.value)

    async def test_arun_exception(self, gitlab_client_mock, metadata):
        """Test handling of a generic exception during API call."""
        gitlab_client_mock.apost.side_effect = Exception("Network Error")
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "some-uuid",
                    "ref": "main",
                }
            )
        assert (
            "An unexpected error occurred while fetching the security finding: Network Error"
            in str(exc_info.value)
        )

    async def test_arun_with_gitlab_http_response(
        self, gitlab_client_mock, metadata, security_finding_response_data
    ):
        """Test that the tool correctly handles GitLabHttpResponse objects."""
        http_response = GitLabHttpResponse(
            status_code=200,
            body=security_finding_response_data,
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.apost = AsyncMock(return_value=http_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        response_str = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
                "ref": "security/sast-fix-773-173",
            }
        )
        response = json.loads(response_str)
        assert "error" not in response
        assert response["finding"]["uuid"] == "1e9a2bf7-0450-5894-8db5-895c98e39deb"

    async def test_arun_with_gitlab_http_response_errors(
        self, gitlab_client_mock, metadata
    ):
        """Test handling of GraphQL errors in GitLabHttpResponse."""
        mock_response_data = {
            "errors": [{"message": "Field 'securityReportFinding' doesn't exist"}]
        }
        http_response = GitLabHttpResponse(
            status_code=200,
            body=mock_response_data,
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.apost = AsyncMock(return_value=http_response)
        tool = GetSecurityFindingDetails(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "uuid": "some-uuid",
                    "ref": "main",
                }
            )
        assert "GraphQL query failed" in str(exc_info.value)


class TestFormatDisplayMessage:
    """Tests for the format_display_message method."""

    def test_format_display_message(self):
        tool = GetSecurityFindingDetails(metadata={})
        args = GetSecurityFindingDetailsInput(
            project_full_path="group/project",
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            ref="my-feature-branch",
        )
        assert (
            tool.format_display_message(args)
            == "Get details for security finding 1e9a2bf7... on ref 'my-feature-branch'"
        )
