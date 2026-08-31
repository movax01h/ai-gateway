import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.findings.list_security_findings import (
    ListSecurityFindings,
    ListSecurityFindingsInput,
    SecurityFindingReportType,
    SecurityFindingSeverity,
    SecurityFindingState,
)
from duo_workflow_service.tools.findings.queries import LIST_SECURITY_FINDINGS_QUERY
from tests.duo_workflow_service.tools.query_test_helpers import (
    extract_location_fragment,
)

# editorconfig-checker-disable
PIPELINE_FINDINGS_JSON = """
{
  "data": {
    "project": {
      "id": "gid://gitlab/Project/26",
      "pipelines": {
        "nodes": [
          {
            "id": "gid://gitlab/Ci::Pipeline/3886",
            "iid": "273",
            "sha": "b791334f7a9e72ba8796002c1ef7573d50c17676",
            "ref": "security/sast-fix-773-173",
            "status": "SUCCESS",
            "securityReportFindings": {
              "nodes": [
                {
                  "uuid": "1e9a2bf7-0450-5894-8db5-895c98e39deb",
                  "title": "OS Command Injection",
                  "severity": "HIGH",
                  "state": "DETECTED",
                  "reportType": "SAST",
                  "falsePositive": false,
                  "aiResolutionAvailable": true,
                  "location": { "file": "pkg/admin/admin.go" },
                  "vulnerability": null
                },
                {
                  "uuid": "6ce00f15-dc81-5d6b-8482-cf24b4dd91e6",
                  "title": "Path Traversal",
                  "severity": "HIGH",
                  "state": "DISMISSED",
                  "reportType": "SAST",
                  "falsePositive": true,
                  "aiResolutionAvailable": true,
                  "location": { "file": "pkg/image/imageUploader.go" },
                  "vulnerability": { "id": "gid://gitlab/Vulnerability/123" }
                },
                {
                  "uuid": "e25ae55f-4239-5929-a777-4e429dfc4acd",
                  "title": "Improper handling of highly compressed data",
                  "severity": "MEDIUM",
                  "state": "DETECTED",
                  "reportType": "SAST",
                  "falsePositive": false,
                  "aiResolutionAvailable": false,
                  "location": { "file": "pkg/image/imageUploader.go" },
                  "vulnerability": null
                }
              ],
              "pageInfo": {
                "hasNextPage": false,
                "endCursor": "OA"
              }
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


@pytest.fixture(name="pipeline_findings_response_data")
def pipeline_findings_response_data_fixture():
    """Fixture for a successful pipeline findings GraphQL response."""
    return json.loads(PIPELINE_FINDINGS_JSON)


@pytest.mark.asyncio
class TestListSecurityFindings:
    """Tests for the ListSecurityFindings tool."""

    async def test_arun_basic_success(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test a successful basic call to list findings."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        response_str = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
            }
        )
        response = json.loads(response_str)

        assert "error" not in response
        assert len(response["findings"]) == 3
        assert response["pipeline"]["iid"] == "273"

        summary = response["summary"]
        assert summary["total"] == 3
        assert summary["by_severity"] == {"HIGH": 2, "MEDIUM": 1}
        assert summary["by_report_type"] == {"SAST": 3}
        assert summary["by_state"] == {"DETECTED": 2, "DISMISSED": 1}
        assert summary["ai_resolvable"] == 2
        assert summary["promoted_to_vulnerability"] == 1
        assert summary["false_positives"] == 1
        assert summary["sast_files_affected"] == 2

        gitlab_client_mock.apost.assert_called_once()
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["fullPath"] == "gitlab-duo/myproject"
        assert call_body["variables"]["ref"] == "security/sast-fix-773-173"

    async def test_arun_with_filters(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that filters are correctly passed to the GraphQL query."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "severity": [SecurityFindingSeverity.HIGH],
                "report_type": [SecurityFindingReportType.SAST],
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        variables = call_body["variables"]
        assert variables["severity"] == ["high"]
        assert variables["reportType"] == ["sast"]
        query = call_body["query"]
        assert "location {" in query
        assert "... on VulnerabilityLocationSast {" in query
        assert "... on VulnerabilityLocationSecretDetection {" in query
        assert "... on VulnerabilityLocationDependencyScanning {" in query
        assert "... on VulnerabilityLocationContainerScanning {" in query

    async def test_arun_include_dismissed_false(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test the logic for `include_dismissed=False` sets the state filter."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "include_dismissed": False,
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        variables = call_body["variables"]
        assert "state" in variables
        assert set(variables["state"]) == {"DETECTED", "CONFIRMED", "RESOLVED"}

    async def test_arun_filters_sent_lowercase_from_mixed_case_strings(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that mixed-case string filters are accepted and sent lowercase."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "severity": [" Critical ", SecurityFindingSeverity.HIGH],
                "report_type": ["sast", "Dependency_Scanning"],
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        variables = call_body["variables"]
        assert variables["severity"] == ["critical", "high"]
        assert variables["reportType"] == ["sast", "dependency_scanning"]

    async def test_arun_duplicate_filter_values_deduped(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that duplicate filter values are deduplicated before being sent.

        Duplicate severities would duplicate every matching result row server-side.
        """
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "severity": ["HIGH", "high", SecurityFindingSeverity.HIGH, "CRITICAL"],
                "report_type": ["sast", "SAST"],
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        variables = call_body["variables"]
        assert variables["severity"] == ["high", "critical"]
        assert variables["reportType"] == ["sast"]

    async def test_arun_sarif_report_type(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that the SARIF report type is accepted and sent lowercase."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "report_type": [SecurityFindingReportType.SARIF],
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["reportType"] == ["sarif"]

    async def test_arun_state_sent_uppercase(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that state values stay UPPERCASE (GraphQL VulnerabilityState enum)."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "state": ["detected", SecurityFindingState.DISMISSED],
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["state"] == ["DETECTED", "DISMISSED"]

    async def test_arun_default_state_includes_dismissed(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that the include_dismissed=True default sends all four states.

        Rails only includes dismissed findings when a state filter is passed explicitly, so the tool must send one by
        default.
        """
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert set(call_body["variables"]["state"]) == {
            "DETECTED",
            "CONFIRMED",
            "RESOLVED",
            "DISMISSED",
        }

    async def test_arun_surfaces_dependency_package_data(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that dependency package data from DS/CS locations reaches the tool output."""
        nodes = pipeline_findings_response_data["data"]["project"]["pipelines"][
            "nodes"
        ][0]["securityReportFindings"]["nodes"]
        nodes[0]["reportType"] = "DEPENDENCY_SCANNING"
        nodes[0]["location"] = {
            "file": "go.sum",
            "dependency": {"version": "1.2.3", "package": {"name": "left-pad"}},
        }
        nodes[1]["reportType"] = "CONTAINER_SCANNING"
        nodes[1]["location"] = {
            "image": "registry.example.com/app:latest",
            "dependency": {"version": "1.1.1t", "package": {"name": "openssl"}},
        }
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        response = json.loads(
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "ref": "security/sast-fix-773-173",
                }
            )
        )
        ds_location = response["findings"][0]["location"]
        assert ds_location["dependency"]["package"]["name"] == "left-pad"
        assert ds_location["dependency"]["version"] == "1.2.3"
        cs_location = response["findings"][1]["location"]
        assert cs_location["dependency"]["package"]["name"] == "openssl"
        assert cs_location["dependency"]["version"] == "1.1.1t"

    async def test_arun_metadata_reports_state_filter_defaulted(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that metadata says whether the state filter was defaulted or explicit."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)

        response = json.loads(
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "ref": "security/sast-fix-773-173",
                }
            )
        )
        assert response["metadata"]["filters_applied"]["state_filter_defaulted"] is True

        response = json.loads(
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "ref": "security/sast-fix-773-173",
                    "state": [SecurityFindingState.DETECTED],
                }
            )
        )
        assert (
            response["metadata"]["filters_applied"]["state_filter_defaulted"] is False
        )

    async def test_arun_include_dismissed_null_behaves_as_default(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that an explicit include_dismissed=None falls back to the documented True default."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "include_dismissed": None,
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert set(call_body["variables"]["state"]) == {
            "DETECTED",
            "CONFIRMED",
            "RESOLVED",
            "DISMISSED",
        }

    async def test_arun_explicit_state_overrides_include_dismissed(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that an explicit state filter is used as-is."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "state": [SecurityFindingState.DETECTED],
                "include_dismissed": True,
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["state"] == ["DETECTED"]

    async def test_arun_explicit_dismissed_state_wins_over_include_dismissed_false(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that an explicit state filter takes precedence over include_dismissed."""
        gitlab_client_mock.apost = AsyncMock(
            return_value=pipeline_findings_response_data
        )
        tool = ListSecurityFindings(metadata=metadata)
        await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "state": [SecurityFindingState.DISMISSED],
                "include_dismissed": False,
            }
        )
        call_body = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert call_body["variables"]["state"] == ["DISMISSED"]

    @pytest.mark.parametrize(
        "filter_kwargs, invalid_value",
        [
            ({"severity": ["moderate"]}, "moderate"),
            ({"report_type": ["GENERIC"]}, "GENERIC"),
            ({"report_type": ["bogus"]}, "bogus"),
            ({"state": ["open"]}, "open"),
        ],
    )
    async def test_arun_invalid_filter_values(
        self, gitlab_client_mock, metadata, filter_kwargs, invalid_value
    ):
        """Test that invalid filter values raise ToolException without an HTTP call."""
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "ref": "security/sast-fix-773-173",
                    **filter_kwargs,
                }
            )
        message = str(exc_info.value)
        assert message.startswith("Invalid")
        assert invalid_value in message
        assert "Valid values" in message
        gitlab_client_mock.apost.assert_not_called()

    async def test_arun_pagination(self, gitlab_client_mock, metadata):
        """Test that the tool correctly handles pagination."""
        page1_data = json.loads(PIPELINE_FINDINGS_JSON)
        page1_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["nodes"] = page1_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["nodes"][:1]
        page1_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["pageInfo"] = {
            "hasNextPage": True,
            "endCursor": "cursor123",
        }

        page2_data = json.loads(PIPELINE_FINDINGS_JSON)
        page2_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["nodes"] = page2_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["nodes"][1:]
        page2_data["data"]["project"]["pipelines"]["nodes"][0][
            "securityReportFindings"
        ]["pageInfo"] = {
            "hasNextPage": False,
            "endCursor": None,
        }

        gitlab_client_mock.apost = AsyncMock(side_effect=[page1_data, page2_data])
        tool = ListSecurityFindings(metadata=metadata)
        response_str = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
                "fetch_all_pages": True,
            }
        )
        response = json.loads(response_str)
        assert len(response["findings"]) == 3
        assert gitlab_client_mock.apost.call_count == 2
        second_call_body = json.loads(
            gitlab_client_mock.apost.call_args_list[1][1]["body"]
        )
        assert second_call_body["variables"]["after"] == "cursor123"

    async def test_arun_project_not_found(self, gitlab_client_mock, metadata):
        """Test error handling when the project is not found raises ToolException."""
        gitlab_client_mock.apost = AsyncMock(return_value={"data": {"project": None}})
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_full_path="non/existent",
                ref="some-branch",
            )
        assert "Project not found or access denied" in str(exc_info.value)

    async def test_arun_no_pipeline_for_ref(self, gitlab_client_mock, metadata):
        """Test error handling when no pipeline is found for the given ref."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "project": {
                        "id": "gid://gitlab/Project/26",
                        "pipelines": {"nodes": []},
                    }
                }
            }
        )
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_full_path="gitlab-duo/myproject",
                ref="nonexistent-branch",
            )
        assert "No pipeline found for ref" in str(exc_info.value)

    async def test_arun_graphql_errors(self, gitlab_client_mock, metadata):
        """Test handling of GraphQL errors in response raises ToolException."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "errors": [{"message": "Field 'securityReportFindings' doesn't exist"}]
            }
        )
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                project_full_path="gitlab-duo/myproject",
                ref="security/sast-fix-773-173",
            )
        assert "GraphQL errors" in str(exc_info.value)

    async def test_arun_exception(self, gitlab_client_mock, metadata):
        """Test handling of generic exceptions."""
        gitlab_client_mock.apost.side_effect = Exception("Network Error")
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "ref": "security/sast-fix-773-173",
                }
            )
        assert "Failed to list security findings: Network Error" in str(exc_info.value)

    async def test_arun_with_gitlab_http_response(
        self, gitlab_client_mock, metadata, pipeline_findings_response_data
    ):
        """Test that the tool correctly handles GitLabHttpResponse objects."""
        http_response = GitLabHttpResponse(
            status_code=200,
            body=pipeline_findings_response_data,
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.apost = AsyncMock(return_value=http_response)
        tool = ListSecurityFindings(metadata=metadata)
        response_str = await tool.arun(
            {
                "project_full_path": "gitlab-duo/myproject",
                "ref": "security/sast-fix-773-173",
            }
        )
        response = json.loads(response_str)
        assert "error" not in response
        assert len(response["findings"]) == 3
        assert response["pipeline"]["iid"] == "273"

    async def test_arun_with_gitlab_http_response_errors(
        self, gitlab_client_mock, metadata
    ):
        """Test handling of GraphQL errors in GitLabHttpResponse raises ToolException."""
        http_response = GitLabHttpResponse(
            status_code=200,
            body={
                "errors": [{"message": "Field 'securityReportFindings' doesn't exist"}]
            },
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.apost = AsyncMock(return_value=http_response)
        tool = ListSecurityFindings(metadata=metadata)
        with pytest.raises(ToolException, match="GraphQL errors"):
            await tool.arun(
                {
                    "project_full_path": "gitlab-duo/myproject",
                    "pipeline_id": "gid://gitlab/Ci::Pipeline/3886",
                    "ref": "security/sast-fix-773-173",
                }
            )

    async def test_format_display_message(self):
        """Test the user-friendly display message formatting."""
        tool = ListSecurityFindings(metadata={})

        args_no_filters = ListSecurityFindingsInput(
            project_full_path="group/project",
            ref="main",
        )
        assert (
            tool.format_display_message(args_no_filters)
            == "List security findings for ref 'main' in group/project"
        )

        args_with_filters = ListSecurityFindingsInput(
            project_full_path="group/project",
            ref="main",
            severity=[SecurityFindingSeverity.CRITICAL, SecurityFindingSeverity.HIGH],
            report_type=[SecurityFindingReportType.SAST],
            state=[SecurityFindingState.DETECTED],
        )
        msg = tool.format_display_message(args_with_filters)
        assert "severity: CRITICAL, HIGH" in msg
        assert "type: SAST" in msg
        assert "state: DETECTED" in msg


@pytest.mark.parametrize(
    "location_type",
    [
        "VulnerabilityLocationDependencyScanning",
        "VulnerabilityLocationContainerScanning",
    ],
)
def test_query_location_fragment_includes_dependency(location_type):
    """The DS and CS location fragments should select dependency package data."""
    fragment = extract_location_fragment(LIST_SECURITY_FINDINGS_QUERY, location_type)
    assert "dependency { version package { name } }" in fragment


def test_query_dependency_scanning_fragment_includes_blob_path():
    """The DS location fragment should select blobPath."""
    fragment = extract_location_fragment(
        LIST_SECURITY_FINDINGS_QUERY, "VulnerabilityLocationDependencyScanning"
    )
    assert "blobPath" in fragment
