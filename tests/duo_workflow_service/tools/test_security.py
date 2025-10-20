import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from packaging.version import InvalidVersion
from pydantic import ValidationError

from duo_workflow_service.tools.security import (
    ConfirmVulnerability,
    ConfirmVulnerabilityInput,
    CreateVulnerabilityIssue,
    CreateVulnerabilityIssueInput,
    DismissVulnerability,
    DismissVulnerabilityInput,
    LinkVulnerabilityToIssue,
    LinkVulnerabilityToIssueInput,
    LinkVulnerabilityToMergeRequest,
    LinkVulnerabilityToMergeRequestInput,
    ListVulnerabilities,
    ListVulnerabilitiesInput,
    RevertToDetectedVulnerability,
    RevertToDetectedVulnerabilityInput,
    VulnerabilityReportType,
    VulnerabilitySeverity,
)


@pytest.fixture(name="vulnerability_data")
def vulnerability_data_fixture():
    """Fixture for common vulnerability data."""
    return [
        {
            "id": "gid://gitlab/Vulnerability/1",
            "title": "SQL Injection",
            "reportType": "SAST",
            "severity": "CRITICAL",
            "state": "DETECTED",
            "location": {
                "file": "app/controllers/users_controller.rb",
                "startLine": 42,
            },
        },
        {
            "id": "gid://gitlab/Vulnerability/2",
            "title": "Outdated Dependency",
            "reportType": "DEPENDENCY_SCANNING",
            "severity": "HIGH",
            "state": "CONFIRMED",
            "location": {
                "file": "Gemfile.lock",
                "dependency": {
                    "package": {"name": "rails"},
                    "version": "5.2.0",
                },
            },
        },
        {
            "id": "gid://gitlab/Vulnerability/3",
            "title": "Container Vulnerability",
            "reportType": "CONTAINER_SCANNING",
            "severity": "MEDIUM",
            "state": "RESOLVED",
            "location": {
                "image": "alpine:3.14",
                "operatingSystem": "alpine",
                "dependency": {
                    "package": {"name": "openssl"},
                    "version": "1.1.1k",
                },
            },
        },
        {
            "id": "gid://gitlab/Vulnerability/4",
            "title": "Hardcoded Secret",
            "reportType": "SECRET_DETECTION",
            "severity": "HIGH",
            "state": "DISMISSED",
            "location": {
                "file": "config/database.yml",
                "startLine": 15,
            },
        },
    ]


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


class TestListVulnerabilitiesInput:
    def test_valid_input(self):
        """Test valid input creation."""
        input_data = ListVulnerabilitiesInput(
            project_full_path="namespace/project",
            severity=[VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL],
            report_type=[VulnerabilityReportType.SAST],
            per_page=50,
        )
        assert input_data.project_full_path == "namespace/project"
        assert len(input_data.severity) == 2

    def test_project_path_validation(self):
        """Test project path validation."""
        valid_paths = [
            "namespace/project",
            "group/subgroup/project",
            "my-group_123/project.name",
        ]
        for path in valid_paths:
            input_data = ListVulnerabilitiesInput(project_full_path=path)
            assert input_data.project_full_path == path

        invalid_paths = [
            "../../../etc/passwd",
            "/absolute/path",
            "path/to/project/",
            "ab",
        ]
        for path in invalid_paths:
            with pytest.raises(ValidationError):
                ListVulnerabilitiesInput(project_full_path=path)

    def test_default_values(self):
        """Test that default values are properly set."""
        input_data = ListVulnerabilitiesInput(project_full_path="namespace/project")

        assert input_data.severity is None
        assert input_data.report_type is None
        assert input_data.per_page == 100
        assert input_data.page == 1
        assert input_data.fetch_all_pages is True

    def test_pagination_validation(self):
        """Test pagination parameter validation."""
        input_data = ListVulnerabilitiesInput(
            project_full_path="namespace/project",
            per_page=50,
            page=2,
        )
        assert input_data.per_page == 50
        assert input_data.page == 2

        with pytest.raises(ValidationError):
            ListVulnerabilitiesInput(
                project_full_path="namespace/project",
                per_page=101,
            )

        with pytest.raises(ValidationError):
            ListVulnerabilitiesInput(
                project_full_path="namespace/project",
                per_page=0,
            )

        with pytest.raises(ValidationError):
            ListVulnerabilitiesInput(
                project_full_path="namespace/project",
                page=0,
            )


@pytest.mark.asyncio
class TestListVulnerabilities:
    async def test_basic_listing(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test basic vulnerability listing without filters."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "project": {
                        "vulnerabilities": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": vulnerability_data,
                        }
                    }
                }
            }
        )

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "namespace/project"})
        result = json.loads(response)

        assert "vulnerabilities" in result
        assert "summary" in result
        assert len(result["vulnerabilities"]) == 4
        assert result["summary"]["total"] == 4
        assert result["summary"]["by_severity"]["CRITICAL"] == 1
        assert result["summary"]["by_severity"]["HIGH"] == 2
        assert result["summary"]["by_severity"]["MEDIUM"] == 1
        assert result["summary"]["by_state"]["DETECTED"] == 1
        assert result["summary"]["by_state"]["CONFIRMED"] == 1
        assert result["summary"]["by_state"]["RESOLVED"] == 1
        assert result["summary"]["by_state"]["DISMISSED"] == 1

    async def test_severity_filtering(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test filtering by multiple severity levels."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "project": {
                        "vulnerabilities": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": vulnerability_data,
                        }
                    }
                }
            }
        )

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "severity": [
                    VulnerabilitySeverity.CRITICAL,
                    VulnerabilitySeverity.HIGH,
                ],
            }
        )

        call_args = gitlab_client_mock.apost.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["variables"]["severity"] == ["CRITICAL", "HIGH"]

    async def test_report_type_filtering(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test filtering by multiple report types."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "project": {
                        "vulnerabilities": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": vulnerability_data,
                        }
                    }
                }
            }
        )

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "report_type": [
                    VulnerabilityReportType.SAST,
                    VulnerabilityReportType.DAST,
                ],
            }
        )

        call_args = gitlab_client_mock.apost.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["variables"]["reportType"] == ["SAST", "DAST"]

    async def test_pagination(self, gitlab_client_mock, metadata, vulnerability_data):
        """Test pagination with multiple pages."""
        first_page_response = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
                        "nodes": vulnerability_data[:2],
                    }
                }
            }
        }

        second_page_response = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": vulnerability_data[2:],
                    }
                }
            }
        }

        gitlab_client_mock.apost = AsyncMock(
            side_effect=[first_page_response, second_page_response]
        )

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "per_page": 2,
                "fetch_all_pages": True,
            }
        )
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 4
        assert gitlab_client_mock.apost.call_count == 2

        second_call_args = gitlab_client_mock.apost.call_args_list[1]
        body = json.loads(second_call_args.kwargs["body"])
        assert body["variables"]["after"] == "cursor1"

    async def test_pagination_without_cursor(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test pagination when endCursor is missing despite hasNextPage being True."""
        response_with_missing_cursor = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "pageInfo": {"hasNextPage": True, "endCursor": None},
                        "nodes": vulnerability_data[:2],
                    }
                }
            }
        }

        gitlab_client_mock.apost = AsyncMock(return_value=response_with_missing_cursor)

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "fetch_all_pages": True,
            }
        )
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 2
        assert gitlab_client_mock.apost.call_count == 1

    async def test_missing_page_info(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test handling when pageInfo is missing from response."""
        response_without_pageinfo = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "nodes": vulnerability_data[:2],
                    }
                }
            }
        }

        gitlab_client_mock.apost = AsyncMock(return_value=response_without_pageinfo)

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "fetch_all_pages": True,
            }
        )
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 2
        assert gitlab_client_mock.apost.call_count == 1

    async def test_project_not_found(self, gitlab_client_mock, metadata):
        """Test handling when project is not found."""
        gitlab_client_mock.apost = AsyncMock(return_value={"data": {"project": None}})

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "nonexistent/project"})
        result = json.loads(response)

        assert "error" in result
        assert "Project not found or access denied" in result["error"]
        assert result["project_path"] == "nonexistent/project"

    async def test_api_error(self, gitlab_client_mock, metadata):
        """Test handling of API errors."""
        gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network error"))

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "namespace/project"})
        result = json.loads(response)

        assert "error" in result
        assert "An error occurred while listing vulnerabilities" in result["error"]
        assert result["error_type"] == "Exception"

    async def test_invalid_response_structure(self, gitlab_client_mock, metadata):
        """Test handling of invalid API response structure."""
        gitlab_client_mock.apost = AsyncMock(return_value={"invalid": "response"})

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "namespace/project"})
        result = json.loads(response)

        assert "error" in result

    async def test_null_nodes_in_response(self, gitlab_client_mock, metadata):
        """Test handling when nodes is null in response."""
        response_with_null_nodes = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": None,
                    }
                }
            }
        }

        gitlab_client_mock.apost = AsyncMock(return_value=response_with_null_nodes)

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "namespace/project"})
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 0
        assert result["summary"]["total"] == 0

    async def test_empty_results(self, gitlab_client_mock, metadata):
        """Test handling of empty vulnerability list."""
        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "project": {
                        "vulnerabilities": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [],
                        }
                    }
                }
            }
        )

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun({"project_full_path": "namespace/project"})
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 0
        assert result["summary"]["total"] == 0
        assert result["summary"]["by_severity"] == {}
        assert result["summary"]["by_state"] == {}
        assert result["summary"]["by_report_type"] == {}

    async def test_empty_cursor_string(
        self, gitlab_client_mock, metadata, vulnerability_data
    ):
        """Test pagination when cursor is an empty string."""
        first_response = {
            "data": {
                "project": {
                    "vulnerabilities": {
                        "pageInfo": {"hasNextPage": True, "endCursor": ""},
                        "nodes": vulnerability_data[:2],
                    }
                }
            }
        }

        gitlab_client_mock.apost = AsyncMock(return_value=first_response)

        tool = ListVulnerabilities(metadata=metadata)
        response = await tool.arun(
            {
                "project_full_path": "namespace/project",
                "fetch_all_pages": True,
            }
        )
        result = json.loads(response)

        assert len(result["vulnerabilities"]) == 2
        assert gitlab_client_mock.apost.call_count == 1


class TestDisplayMessageFormatting:
    def test_format_display_message(self):
        """Test display message formatting with various filters."""
        tool = ListVulnerabilities(metadata={})

        args = ListVulnerabilitiesInput(project_full_path="namespace/project")
        message = tool.format_display_message(args)
        assert message == "List vulnerabilities in project namespace/project"

        args = ListVulnerabilitiesInput(
            project_full_path="namespace/project",
            severity=[VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL],
        )
        message = tool.format_display_message(args)
        assert "severity: HIGH, CRITICAL" in message

        args = ListVulnerabilitiesInput(
            project_full_path="namespace/project",
            severity=[VulnerabilitySeverity.HIGH],
            report_type=[VulnerabilityReportType.SAST],
        )
        message = tool.format_display_message(args)

        assert "severity: HIGH" in message
        assert "report type: SAST" in message


class TestEnumCoverage:
    def test_all_vulnerability_severities(self):
        """Test all VulnerabilitySeverity enum values."""
        severities = list(VulnerabilitySeverity)
        assert len(severities) == 6
        assert VulnerabilitySeverity.CRITICAL in severities
        assert VulnerabilitySeverity.HIGH in severities
        assert VulnerabilitySeverity.MEDIUM in severities
        assert VulnerabilitySeverity.LOW in severities
        assert VulnerabilitySeverity.INFO in severities
        assert VulnerabilitySeverity.UNKNOWN in severities

    def test_all_vulnerability_report_types(self):
        """Test all VulnerabilityReportType enum values."""
        report_types = list(VulnerabilityReportType)
        assert len(report_types) == 10
        assert VulnerabilityReportType.SAST in report_types
        assert VulnerabilityReportType.DEPENDENCY_SCANNING in report_types
        assert VulnerabilityReportType.CONTAINER_SCANNING in report_types


# Dismiss Vulnerability Tests
@pytest.mark.asyncio
async def test_dismiss_vulnerability(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityDismiss": {
                    "errors": [],
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "description": "Test vulnerability",
                        "state": "DISMISSED",
                        "dismissedAt": "2023-01-01T00:00:00Z",
                        "dismissalReason": "FALSE_POSITIVE",
                    },
                }
            }
        }
    )

    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Security review deemed this a false positive",
        "dismissal_reason": "FALSE_POSITIVE",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerability": {
                "id": "gid://gitlab/Vulnerability/123",
                "description": "Test vulnerability",
                "state": "DISMISSED",
                "dismissedAt": "2023-01-01T00:00:00Z",
                "dismissalReason": "FALSE_POSITIVE",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    body = json.loads(call_args.kwargs["body"])
    assert body["variables"]["vulnerabilityId"] == "gid://gitlab/Vulnerability/123"
    assert (
        body["variables"]["comment"] == "Security review deemed this a false positive"
    )
    assert body["variables"]["dismissalReason"] == "FALSE_POSITIVE"


@pytest.mark.asyncio
async def test_dismiss_vulnerability_with_numeric_id(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityDismiss": {
                    "errors": [],
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "description": "Test vulnerability",
                        "state": "DISMISSED",
                        "dismissedAt": "2023-01-01T00:00:00Z",
                        "dismissalReason": "ACCEPTABLE_RISK",
                    },
                }
            }
        }
    )

    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "123",  # Numeric ID without GraphQL prefix
        "comment": "Acceptable risk after review",
        "dismissal_reason": "ACCEPTABLE_RISK",
    }

    response = await tool.arun(input_data)

    # Should automatically add the GraphQL prefix
    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    body = json.loads(call_args.kwargs["body"])
    assert body["variables"]["vulnerabilityId"] == "gid://gitlab/Vulnerability/123"


@pytest.mark.asyncio
async def test_dismiss_vulnerability_invalid_dismissal_reason(
    gitlab_client_mock, metadata
):
    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Test comment",
        "dismissal_reason": "INVALID_REASON",
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid dismissal reason" in error_response["error"]
    assert "INVALID_REASON" in error_response["error"]

    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_dismiss_vulnerability_comment_too_long(gitlab_client_mock, metadata):
    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "x" * 50001,  # 50,001 characters (should fail)
        "dismissal_reason": "FALSE_POSITIVE",
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Comment must be 50,000 characters or less" in error_response["error"]

    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_dismiss_vulnerability_with_api_errors(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityDismiss": {
                    "errors": ["Vulnerability not found", "Access denied"],
                    "vulnerability": None,
                }
            }
        }
    )

    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Test comment",
        "dismissal_reason": "FALSE_POSITIVE",
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Vulnerability not found; Access denied" in error_response["error"]


@pytest.mark.parametrize(
    "dismissal_reason",
    [
        "ACCEPTABLE_RISK",
        "FALSE_POSITIVE",
        "MITIGATING_CONTROL",
        "USED_IN_TESTS",
        "NOT_APPLICABLE",
    ],
)
@pytest.mark.asyncio
async def test_dismiss_vulnerability_valid_dismissal_reasons(
    gitlab_client_mock, metadata, dismissal_reason
):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityDismiss": {
                    "errors": [],
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "description": "Test vulnerability",
                        "state": "DISMISSED",
                        "dismissedAt": "2023-01-01T00:00:00Z",
                        "dismissalReason": dismissal_reason,
                    },
                }
            }
        }
    )

    tool = DismissVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Test comment",
        "dismissal_reason": dismissal_reason,
    }

    response = await tool.arun(input_data)

    # Should not return an error
    response_data = json.loads(response)
    assert "error" not in response_data
    assert "vulnerability" in response_data

    gitlab_client_mock.apost.assert_called_once()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            DismissVulnerabilityInput(
                vulnerability_id="gid://gitlab/Vulnerability/123",
                comment="Test comment",
                dismissal_reason="FALSE_POSITIVE",
            ),
            "Dismiss vulnerability gid://gitlab/Vulnerability/123",
        ),
        (
            DismissVulnerabilityInput(
                vulnerability_id="456",
                comment="Another test comment",
                dismissal_reason="ACCEPTABLE_RISK",
            ),
            "Dismiss vulnerability 456",
        ),
    ],
)
def test_dismiss_vulnerability_format_display_message(input_data, expected_message):
    tool = DismissVulnerability(metadata={})
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_link_vulnerability_to_issue(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityIssueLinkCreate": {
                    "errors": [],
                    "issueLinks": [
                        {
                            "id": "gid://gitlab/VulnerabilityIssueLink/1",
                            "issue": {
                                "id": "gid://gitlab/Issue/1",
                                "title": "Security Issue #1",
                                "name": "Security Issue #1",
                            },
                            "linkType": "RELATED",
                        }
                    ],
                }
            }
        }
    )

    tool = LinkVulnerabilityToIssue(metadata=metadata)

    input_data = {
        "issue_id": "gid://gitlab/Issue/1",
        "vulnerability_ids": [
            "gid://gitlab/Vulnerability/23",
            "gid://gitlab/Vulnerability/10",
        ],
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "issueLinks": [
                {
                    "id": "gid://gitlab/VulnerabilityIssueLink/1",
                    "issue": {
                        "id": "gid://gitlab/Issue/1",
                        "title": "Security Issue #1",
                        "name": "Security Issue #1",
                    },
                    "linkType": "RELATED",
                }
            ]
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_mutation = """
        mutation($vulnerabilityIds: [VulnerabilityID!]!, $issueId: IssueID!) {
          vulnerabilityIssueLinkCreate(input: { issueId: $issueId, vulnerabilityIds: $vulnerabilityIds }) {
            issueLinks {
              id
              issue {
                id,
                title,
                name
              }
              linkType
            }
            errors
          }
        }
        """
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_mutation,
                "variables": {
                    "issueId": "gid://gitlab/Issue/1",
                    "vulnerabilityIds": [
                        "gid://gitlab/Vulnerability/23",
                        "gid://gitlab/Vulnerability/10",
                    ],
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_link_vulnerability_to_issue_with_numeric_ids(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityIssueLinkCreate": {
                    "errors": [],
                    "issueLinks": [
                        {
                            "id": "gid://gitlab/VulnerabilityIssueLink/1",
                            "issue": {
                                "id": "gid://gitlab/Issue/1",
                                "title": "Security Issue #1",
                                "name": "Security Issue #1",
                            },
                            "linkType": "RELATED",
                        }
                    ],
                }
            }
        }
    )

    tool = LinkVulnerabilityToIssue(metadata=metadata)

    input_data = {
        "issue_id": "1",
        "vulnerability_ids": ["23", "gid://gitlab/Vulnerability/10"],
    }

    response = await tool.arun(input_data)

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    body = json.loads(call_args[1]["body"])
    assert body["variables"]["issueId"] == "gid://gitlab/Issue/1"
    assert body["variables"]["vulnerabilityIds"] == [
        "gid://gitlab/Vulnerability/23",
        "gid://gitlab/Vulnerability/10",
    ]

    response_data = json.loads(response)
    assert "error" not in response_data
    assert "issueLinks" in response_data


@pytest.mark.asyncio
async def test_link_vulnerability_to_issue_with_api_errors(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityIssueLinkCreate": {
                    "errors": ["Issue not found", "Vulnerability not found"],
                    "issueLinks": None,
                }
            }
        }
    )

    tool = LinkVulnerabilityToIssue(metadata=metadata)

    input_data = {
        "issue_id": "gid://gitlab/Issue/999",
        "vulnerability_ids": ["gid://gitlab/Vulnerability/999"],
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Issue not found; Vulnerability not found" in error_response["error"]


@pytest.mark.asyncio
async def test_link_vulnerability_to_issue_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API Error"))

    tool = LinkVulnerabilityToIssue(metadata=metadata)

    input_data = {
        "issue_id": "1",
        "vulnerability_ids": ["23"],
    }

    with pytest.raises(Exception) as exc_info:
        await tool.arun(input_data)

    assert "API Error" in str(exc_info.value)


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            LinkVulnerabilityToIssueInput(
                issue_id="gid://gitlab/Issue/1",
                vulnerability_ids=[
                    "gid://gitlab/Vulnerability/23",
                    "gid://gitlab/Vulnerability/10",
                ],
            ),
            "Link issue to vulnerability ['gid://gitlab/Vulnerability/23', 'gid://gitlab/Vulnerability/10']",
        ),
        (
            LinkVulnerabilityToIssueInput(issue_id="1", vulnerability_ids=["23"]),
            "Link issue to vulnerability ['23']",
        ),
    ],
)
def test_link_vulnerability_to_issue_format_display_message(
    input_data, expected_message
):
    tool = LinkVulnerabilityToIssue(metadata={})
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_confirm_vulnerability(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityConfirm": {
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "state": "CONFIRMED",
                        "title": "Test Vulnerability",
                        "severity": "HIGH",
                        "reportType": "SAST",
                    },
                    "errors": [],
                }
            }
        }
    )

    tool = ConfirmVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Verified as a real security issue",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerability": {
                "id": "gid://gitlab/Vulnerability/123",
                "state": "CONFIRMED",
                "title": "Test Vulnerability",
                "severity": "HIGH",
                "reportType": "SAST",
            },
            "success": True,
            "message": "Vulnerability confirmed successfully",
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_mutation = """
mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
    vulnerabilityConfirm(input: { id: $vulnerabilityId, comment: $comment }) {
    vulnerability {
        id
        state
        title
        severity
        reportType
    }
    errors
    }
}
"""
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_mutation,
                "variables": {
                    "vulnerabilityId": "gid://gitlab/Vulnerability/123",
                    "comment": "Verified as a real security issue",
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_confirm_vulnerability_without_comment(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityConfirm": {
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "state": "CONFIRMED",
                        "title": "Test Vulnerability",
                        "severity": "HIGH",
                        "reportType": "SAST",
                    },
                    "errors": [],
                }
            }
        }
    )

    tool = ConfirmVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerability": {
                "id": "gid://gitlab/Vulnerability/123",
                "state": "CONFIRMED",
                "title": "Test Vulnerability",
                "severity": "HIGH",
                "reportType": "SAST",
            },
            "success": True,
            "message": "Vulnerability confirmed successfully",
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_mutation = """
mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
    vulnerabilityConfirm(input: { id: $vulnerabilityId, comment: $comment }) {
    vulnerability {
        id
        state
        title
        severity
        reportType
    }
    errors
    }
}
"""
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_mutation,
                "variables": {
                    "vulnerabilityId": "gid://gitlab/Vulnerability/123",
                    "comment": None,
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_confirm_vulnerability_with_numeric_id(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityConfirm": {
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "state": "CONFIRMED",
                        "title": "Test Vulnerability",
                        "severity": "HIGH",
                        "reportType": "SAST",
                    },
                    "errors": [],
                }
            }
        }
    )

    tool = ConfirmVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "123",  # Numeric ID
        "comment": "Verified as a real security issue",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerability": {
                "id": "gid://gitlab/Vulnerability/123",
                "state": "CONFIRMED",
                "title": "Test Vulnerability",
                "severity": "HIGH",
                "reportType": "SAST",
            },
            "success": True,
            "message": "Vulnerability confirmed successfully",
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_mutation = """
mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
    vulnerabilityConfirm(input: { id: $vulnerabilityId, comment: $comment }) {
    vulnerability {
        id
        state
        title
        severity
        reportType
    }
    errors
    }
}
"""
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_mutation,
                "variables": {
                    "vulnerabilityId": "gid://gitlab/Vulnerability/123",
                    "comment": "Verified as a real security issue",
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_confirm_vulnerability_with_graphql_errors(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityConfirm": {
                    "vulnerability": None,
                    "errors": ["Vulnerability not found"],
                }
            }
        }
    )

    tool = ConfirmVulnerability(metadata=metadata)

    response = await tool.arun({"vulnerability_id": "gid://gitlab/Vulnerability/999"})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "GraphQL errors: ['Vulnerability not found']" in error_response["error"]


@pytest.mark.asyncio
async def test_confirm_vulnerability_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API Error"))

    tool = ConfirmVulnerability(metadata=metadata)

    response = await tool.arun({"vulnerability_id": "gid://gitlab/Vulnerability/123"})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "API Error" in error_response["error"]


@pytest.mark.asyncio
async def test_confirm_vulnerability_with_long_comment_error(
    gitlab_client_mock, metadata
):
    """Test case 4: with comment with more than 50000 characters, the agent should throw error about too long comment"""
    tool = ConfirmVulnerability(metadata=metadata)

    # Create a comment with more than 50,000 characters
    long_comment = "a" * 50001

    input_data = {
        "vulnerability_id": "123",
        "comment": long_comment,
    }

    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Comment must be 50,000 characters or less" in error_response["error"]

    # Verify that no API call was made due to validation error
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ConfirmVulnerabilityInput(vulnerability_id="123"),
            "Confirm vulnerability 123",
        ),
        (
            ConfirmVulnerabilityInput(
                vulnerability_id="gid://gitlab/Vulnerability/456"
            ),
            "Confirm vulnerability gid://gitlab/Vulnerability/456",
        ),
    ],
)
def test_confirm_vulnerability_format_display_message(input_data, expected_message):
    tool = ConfirmVulnerability(metadata={})
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_revert_to_detected_vulnerability(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilityRevertToDetected": {
                    "errors": [],
                    "vulnerability": {
                        "id": "gid://gitlab/Vulnerability/123",
                        "title": "SQL Injection",
                        "state": "DETECTED",
                        "severity": "HIGH",
                    },
                }
            }
        }
    )

    tool = RevertToDetectedVulnerability(metadata=metadata)

    input_data = {
        "vulnerability_id": "gid://gitlab/Vulnerability/123",
        "comment": "Reverting for re-assessment",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerability": {
                "id": "gid://gitlab/Vulnerability/123",
                "title": "SQL Injection",
                "state": "DETECTED",
                "severity": "HIGH",
            },
            "status": "reverted_to_detected",
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_mutation = """
        mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
          vulnerabilityRevertToDetected(input: {
            id: $vulnerabilityId
            comment: $comment
          }) {
            errors
            vulnerability {
              id
              title
              state
              severity
            }
          }
        }
        """
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_mutation,
                "variables": {
                    "vulnerabilityId": "gid://gitlab/Vulnerability/123",
                    "comment": "Reverting for re-assessment",
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_revert_to_detected_vulnerability_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API Error"))

    tool = RevertToDetectedVulnerability(metadata=metadata)

    response = await tool.arun({"vulnerability_id": "gid://gitlab/Vulnerability/123"})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "API Error" in error_response["error"]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            RevertToDetectedVulnerabilityInput(
                vulnerability_id="gid://gitlab/Vulnerability/123"
            ),
            "Revert vulnerability gid://gitlab/Vulnerability/123 to detected state",
        ),
        (
            RevertToDetectedVulnerabilityInput(
                vulnerability_id="gid://gitlab/Vulnerability/456",
                comment="Reverting for re-assessment after code changes",
            ),
            "Revert vulnerability gid://gitlab/Vulnerability/456 to detected state - Reason: Reverting for re-assessment after code changes",
        ),
    ],
)
def test_revert_to_detected_vulnerability_format_display_message(
    input_data, expected_message
):
    tool = RevertToDetectedVulnerability(metadata={})
    assert tool.format_display_message(input_data) == expected_message


@pytest.fixture
def vulnerability_ids():
    return [
        "gid://gitlab/Vulnerability/542",
        "gid://gitlab/Vulnerability/543",
    ]


@pytest.fixture
def input_data(vulnerability_ids):
    return {
        "project_full_path": "gitlab-duo/test",
        "vulnerability_ids": vulnerability_ids,
    }


@pytest.fixture
def successful_project_response():
    return {"data": {"project": {"id": "gid://gitlab/Project/1000000"}}}


@pytest.fixture
def successful_issue_response():
    return {
        "data": {
            "vulnerabilitiesCreateIssue": {
                "issue": {
                    "id": "gid://gitlab/Issue/641",
                    "title": "Investigate vulnerabilities",
                    "name": "Investigate vulnerabilities",
                },
                "errors": [],
            }
        }
    }


@pytest.fixture
def successful_mock_sequence(successful_project_response, successful_issue_response):
    return [successful_project_response, successful_issue_response]


@pytest.fixture
def expected_response():
    return json.dumps(
        {
            "issue": {
                "id": "gid://gitlab/Issue/641",
                "title": "Investigate vulnerabilities",
                "name": "Investigate vulnerabilities",
            }
        }
    )


@pytest.mark.asyncio
async def test_create_vulnerability_issue(
    gitlab_client_mock,
    metadata,
    input_data,
    successful_mock_sequence,
    expected_response,
):
    gitlab_client_mock.apost = AsyncMock(side_effect=successful_mock_sequence)

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    assert response == expected_response
    assert gitlab_client_mock.apost.call_count == 2

    first_call = gitlab_client_mock.apost.call_args_list[0]
    project_query_body = json.loads(first_call[1]["body"])
    assert "project(fullPath: $projectFullPath)" in project_query_body["query"]
    assert project_query_body["variables"]["projectFullPath"] == "gitlab-duo/test"

    second_call = gitlab_client_mock.apost.call_args_list[1]
    mutation_body = json.loads(second_call[1]["body"])
    assert "vulnerabilitiesCreateIssue" in mutation_body["query"]
    assert mutation_body["variables"]["projectId"] == "gid://gitlab/Project/1000000"
    assert mutation_body["variables"]["vulnerabilityIds"] == [
        "gid://gitlab/Vulnerability/542",
        "gid://gitlab/Vulnerability/543",
    ]


@pytest.mark.asyncio
async def test_create_vulnerability_issue_with_numeric_ids(
    gitlab_client_mock, metadata, successful_mock_sequence
):
    input_data = {
        "project_full_path": "gitlab-duo/test",
        "vulnerability_ids": ["542", "gid://gitlab/Vulnerability/543"],
    }

    gitlab_client_mock.apost = AsyncMock(side_effect=successful_mock_sequence)

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    mutation_call = gitlab_client_mock.apost.call_args_list[1]
    mutation_body = json.loads(mutation_call[1]["body"])
    assert mutation_body["variables"]["vulnerabilityIds"] == [
        "gid://gitlab/Vulnerability/542",
        "gid://gitlab/Vulnerability/543",
    ]

    response_data = json.loads(response)
    assert "error" not in response_data
    assert "issue" in response_data


@pytest.mark.asyncio
async def test_create_vulnerability_issue_project_not_found(
    gitlab_client_mock, metadata
):
    input_data = {
        "project_full_path": "nonexistent/project",
        "vulnerability_ids": ["gid://gitlab/Vulnerability/542"],
    }

    gitlab_client_mock.apost = AsyncMock(return_value={"data": {"project": None}})

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Project not found or access denied" in error_response["error"]
    assert error_response["project_path"] == "nonexistent/project"
    assert gitlab_client_mock.apost.call_count == 1


@pytest.mark.asyncio
async def test_create_vulnerability_issue_with_mutation_errors(
    gitlab_client_mock, metadata, input_data, successful_project_response
):
    error_issue_response = {
        "data": {
            "vulnerabilitiesCreateIssue": {
                "issue": None,
                "errors": [
                    "Vulnerability not found",
                    "Insufficient permissions",
                ],
            }
        }
    }

    gitlab_client_mock.apost = AsyncMock(
        side_effect=[successful_project_response, error_issue_response]
    )

    tool = CreateVulnerabilityIssue(metadata=metadata)
    test_input = {**input_data, "vulnerability_ids": ["gid://gitlab/Vulnerability/999"]}
    response = await tool.arun(test_input)

    error_response = json.loads(response)
    assert "error" in error_response
    assert (
        "Vulnerability not found; Insufficient permissions" in error_response["error"]
    )


@pytest.mark.asyncio
async def test_create_vulnerability_issue_exception(
    gitlab_client_mock, metadata, input_data
):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API Error"))

    tool = CreateVulnerabilityIssue(metadata=metadata)

    with pytest.raises(Exception) as exc_info:
        await tool.arun(input_data)

    assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_vulnerability_issue_invalid_project_response(
    gitlab_client_mock, metadata, input_data
):
    gitlab_client_mock.apost = AsyncMock(return_value=None)

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid GraphQL response" in error_response["error"]
    assert error_response["project_path"] == "gitlab-duo/test"


@pytest.mark.asyncio
async def test_create_vulnerability_issue_project_response_missing_data(
    gitlab_client_mock, metadata, input_data
):
    gitlab_client_mock.apost = AsyncMock(return_value={"errors": ["Some error"]})

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid GraphQL response" in error_response["error"]
    assert error_response["project_path"] == "gitlab-duo/test"


@pytest.mark.asyncio
async def test_create_vulnerability_issue_invalid_mutation_response(
    gitlab_client_mock, metadata, input_data, successful_project_response
):
    gitlab_client_mock.apost = AsyncMock(
        side_effect=[successful_project_response, None]
    )

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid GraphQL response" in error_response["error"]


@pytest.mark.asyncio
async def test_create_vulnerability_issue_mutation_response_missing_data(
    gitlab_client_mock, metadata, input_data, successful_project_response
):
    gitlab_client_mock.apost = AsyncMock(
        side_effect=[successful_project_response, {"errors": ["Some mutation error"]}]
    )

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid GraphQL response" in error_response["error"]


@pytest.mark.asyncio
async def test_create_vulnerability_issue_mutation_response_missing_key(
    gitlab_client_mock, metadata, input_data, successful_project_response
):
    gitlab_client_mock.apost = AsyncMock(
        side_effect=[successful_project_response, {"data": {"someOtherField": "value"}}]
    )

    tool = CreateVulnerabilityIssue(metadata=metadata)
    response = await tool.arun(input_data)

    error_response = json.loads(response)
    assert "error" in error_response
    assert "Invalid GraphQL response" in error_response["error"]


@pytest.mark.parametrize(
    "test_input_data,expected_message",
    [
        (
            CreateVulnerabilityIssueInput(
                project_full_path="gitlab-duo/test",
                vulnerability_ids=[
                    "gid://gitlab/Vulnerability/542",
                    "gid://gitlab/Vulnerability/543",
                ],
            ),
            "Create issue for vulnerabilities in project gitlab-duo/test",
        ),
        (
            CreateVulnerabilityIssueInput(
                project_full_path="namespace/project", vulnerability_ids=["542"]
            ),
            "Create issue for vulnerabilities in project namespace/project",
        ),
    ],
)
def test_create_vulnerability_issue_format_display_message(
    test_input_data, expected_message
):
    tool = CreateVulnerabilityIssue(metadata={})
    assert tool.format_display_message(test_input_data) == expected_message


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request(gitlab_client_mock, metadata):
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = (
            "18.5.0"  # Version 18.5 or above should work
        )

        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "vulnerabilityLinkMergeRequest": {
                        "vulnerability": {
                            "id": "gid://gitlab/Vulnerability/123",
                            "mergeRequests": {
                                "nodes": [
                                    {
                                        "id": "gid://gitlab/MergeRequest/456",
                                        "title": "Fix security vulnerability",
                                    }
                                ]
                            },
                        },
                        "errors": [],
                    }
                }
            }
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)

        test_input_data = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input_data)

        expected_response_data = json.dumps(
            {
                "vulnerability": {
                    "id": "gid://gitlab/Vulnerability/123",
                    "mergeRequests": {
                        "nodes": [
                            {
                                "id": "gid://gitlab/MergeRequest/456",
                                "title": "Fix security vulnerability",
                            }
                        ]
                    },
                }
            }
        )
        assert response == expected_response_data

        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["variables"]["vulnerabilityId"] == "gid://gitlab/Vulnerability/123"
        assert body["variables"]["mergeRequestId"] == "gid://gitlab/MergeRequest/456"


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_with_numeric_ids(
    gitlab_client_mock, metadata
):
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = (
            "18.5.0"  # Version 18.5 or above should work
        )

        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "vulnerabilityLinkMergeRequest": {
                        "vulnerability": {
                            "id": "gid://gitlab/Vulnerability/123",
                            "mergeRequests": {
                                "nodes": [
                                    {
                                        "id": "gid://gitlab/MergeRequest/456",
                                        "title": "Fix security vulnerability",
                                    }
                                ]
                            },
                        },
                        "errors": [],
                    }
                }
            }
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)

        test_input_data = {
            "vulnerability_id": "123",  # Numeric ID without GraphQL prefix
            "merge_request_id": "456",  # Numeric ID without GraphQL prefix
        }

        response = await tool.arun(test_input_data)

        # Should automatically add the GraphQL prefixes
        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["variables"]["vulnerabilityId"] == "gid://gitlab/Vulnerability/123"
        assert body["variables"]["mergeRequestId"] == "gid://gitlab/MergeRequest/456"

        response_data = json.loads(response)
        assert "error" not in response_data
        assert "vulnerability" in response_data


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_with_api_errors(
    gitlab_client_mock, metadata
):
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = (
            "18.5.0"  # Version 18.5 or above should work
        )

        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "vulnerabilityLinkMergeRequest": {
                        "vulnerability": None,
                        "errors": [
                            "Vulnerability not found",
                            "Merge request not found",
                        ],
                    }
                }
            }
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)

        test_input_data = {
            "vulnerability_id": "gid://gitlab/Vulnerability/999",
            "merge_request_id": "gid://gitlab/MergeRequest/999",
        }

        response = await tool.arun(test_input_data)

        error_response = json.loads(response)
        assert "error" in error_response
        assert (
            "Vulnerability not found; Merge request not found"
            in error_response["error"]
        )


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_unexpected_response_structure(
    gitlab_client_mock, metadata
):
    """Test handling of unexpected response structure that causes KeyError."""
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = (
            "18.5.0"  # Version 18.5 or above should work
        )

        gitlab_client_mock.apost = AsyncMock(
            return_value={"data": {"unexpectedField": {"someData": "value"}}}
        )
        gitlab_client_mock._process_http_response = (
            lambda identifier, response: response
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)
        test_input_data = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input_data)
        error_response = json.loads(response)
        assert "error" in error_response
        assert "Unexpected response structure" in error_response["error"]
        assert "response" in error_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            LinkVulnerabilityToMergeRequestInput(
                vulnerability_id="gid://gitlab/Vulnerability/123",
                merge_request_id="gid://gitlab/MergeRequest/456",
            ),
            "Link vulnerability gid://gitlab/Vulnerability/123 to merge request gid://gitlab/MergeRequest/456",
        ),
        (
            LinkVulnerabilityToMergeRequestInput(
                vulnerability_id="123",
                merge_request_id="456",
            ),
            "Link vulnerability 123 to merge request 456",
        ),
    ],
)
def test_link_vulnerability_to_merge_request_format_display_message(
    input_data, expected_message
):
    tool = LinkVulnerabilityToMergeRequest(metadata={})
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_version_18_5_works(
    gitlab_client_mock, metadata
):
    """Test that the tool works normally when GitLab version is exactly 18.5."""
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = "18.5.0"

        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "vulnerabilityLinkMergeRequest": {
                        "vulnerability": {
                            "id": "gid://gitlab/Vulnerability/123",
                            "mergeRequests": {
                                "nodes": [
                                    {
                                        "id": "gid://gitlab/MergeRequest/456",
                                        "title": "Fix security vulnerability",
                                    }
                                ]
                            },
                        },
                        "errors": [],
                    }
                }
            }
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)
        test_input = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input)
        response_data = json.loads(response)

        # Should work normally for version 18.5
        assert "error" not in response_data
        assert "vulnerability" in response_data

        # Verify that API call was made
        gitlab_client_mock.apost.assert_called_once()


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_version_18_6_works(
    gitlab_client_mock, metadata
):
    """Test that the tool works normally when GitLab version is 18.6 or greater."""
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = "18.6.0"

        gitlab_client_mock.apost = AsyncMock(
            return_value={
                "data": {
                    "vulnerabilityLinkMergeRequest": {
                        "vulnerability": {
                            "id": "gid://gitlab/Vulnerability/123",
                            "mergeRequests": {
                                "nodes": [
                                    {
                                        "id": "gid://gitlab/MergeRequest/456",
                                        "title": "Fix security vulnerability",
                                    }
                                ]
                            },
                        },
                        "errors": [],
                    }
                }
            }
        )

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)
        test_input = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input)
        response_data = json.loads(response)

        # Should work normally for version 18.6
        assert "error" not in response_data
        assert "vulnerability" in response_data

        # Verify that API call was made
        gitlab_client_mock.apost.assert_called_once()


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_version_below_18_5_fails(
    gitlab_client_mock, metadata
):
    """Test that the tool returns an error when GitLab version is below 18.5."""
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.return_value = "18.4.0"

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)
        test_input = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input)
        error_response = json.loads(response)

        # Should return error for version below 18.5
        assert "error" in error_response
        assert error_response["error"] == "This tool is not available"

        # Verify that no API call was made
        gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_link_vulnerability_to_merge_request_invalid_version_fails(
    gitlab_client_mock, metadata
):
    """Test that the tool returns error when GitLab version is invalid (uses fallback 18.2.0)."""
    with patch(
        "duo_workflow_service.tools.security.gitlab_version"
    ) as mock_gitlab_version:
        mock_gitlab_version.get.side_effect = InvalidVersion("invalid version")

        tool = LinkVulnerabilityToMergeRequest(metadata=metadata)
        test_input = {
            "vulnerability_id": "gid://gitlab/Vulnerability/123",
            "merge_request_id": "gid://gitlab/MergeRequest/456",
        }

        response = await tool.arun(test_input)
        error_response = json.loads(response)

        # Should return error with fallback version (18.2.0 < 18.5)
        assert "error" in error_response
        assert error_response["error"] == "This tool is not available"

        # Verify that no API call was made
        gitlab_client_mock.apost.assert_not_called()
