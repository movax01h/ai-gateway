import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.security import (
    DismissVulnerability,
    DismissVulnerabilityInput,
    ListVulnerabilities,
    ListVulnerabilitiesInput,
)

# Common URL test parameters
URL_SUCCESS_CASES = [
    # Test with only URL
    (
        "https://gitlab.com/namespace/project",
        None,
        "/api/v4/projects/namespace%2Fproject/vulnerabilities",
    ),
    # Test with URL and matching project_id
    (
        "https://gitlab.com/namespace/project",
        "namespace%2Fproject",
        "/api/v4/projects/namespace%2Fproject/vulnerabilities",
    ),
]

URL_ERROR_CASES = [
    # URL and project_id both given, but don't match
    (
        "https://gitlab.com/namespace/project",
        "different%2Fproject",
        "Project ID mismatch",
    ),
    # URL given isn't a valid GitLab URL
    (
        "https://example.com/not-gitlab",
        None,
        "Failed to parse URL",
    ),
]


@pytest.fixture
def vulnerability_data():
    """Fixture for common vulnerability data."""
    return [
        {
            "id": "gid://gitlab/Vulnerability/1",
            "title": "Test Vulnerability",
            "reportType": "SAST",
            "severity": "HIGH",
            "location": {
                "file": "app/controllers/users_controller.rb",
                "startLine": 42,
            },
        }
    ]


@pytest.fixture
def gitlab_client_mock():
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


async def tool_url_success_response(
    tool,
    url,
    project_id,
    gitlab_client_mock,
    response_data,
    **kwargs,
):
    gitlab_client_mock.aget = AsyncMock(return_value=response_data)

    response = await tool._arun(url=url, project_id=project_id, **kwargs)

    return response


async def assert_tool_url_error(
    tool,
    url,
    project_id,
    error_contains,
    gitlab_client_mock,
    **kwargs,
):
    response = await tool._arun(url=url, project_id=project_id, **kwargs)

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_contains in error_response["error"]

    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_vulnerabilities(gitlab_client_mock, metadata, vulnerability_data):
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

    input_data = {
        "project_full_path": "namespace/project",
        "per_page": 50,
        "page": 1,
        "fetch_all_pages": False,
        "severity": "HIGH",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerabilities": vulnerability_data,
            "pagination": {"total_items": len(vulnerability_data)},
        }
    )
    assert response == expected_response

    # editorconfig-checker-disable
    expected_query = """
        query($projectFullPath: ID!, $first: Int, $after: String, $severity: [VulnerabilitySeverity!]) {
          project(fullPath: $projectFullPath) {
            vulnerabilities(first: $first, after: $after, severity: $severity) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
                title
                reportType
                severity
                location{
                  ... on VulnerabilityLocationSast {
                    file
                    startLine
                  }
                }
              }
            }
          }
        }
        """
    # editorconfig-checker-enable

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/graphql",
        body=json.dumps(
            {
                "query": expected_query,
                "variables": {
                    "projectFullPath": "namespace/project",
                    "first": 50,
                    "after": None,
                    "severity": "HIGH",
                },
            }
        ),
    )


@pytest.mark.asyncio
async def test_list_vulnerabilities_with_pagination(
    gitlab_client_mock, metadata, vulnerability_data
):
    # First page response
    first_page_response = {
        "data": {
            "project": {
                "vulnerabilities": {
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
                    "nodes": vulnerability_data,
                }
            }
        }
    }

    # Second page response
    second_page_response = {
        "data": {
            "project": {
                "vulnerabilities": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": vulnerability_data,
                }
            }
        }
    }

    gitlab_client_mock.apost = AsyncMock(
        side_effect=[first_page_response, second_page_response]
    )

    tool = ListVulnerabilities(metadata=metadata)

    input_data = {
        "project_full_path": "namespace/project",
        "per_page": 50,
        "fetch_all_pages": True,
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "vulnerabilities": vulnerability_data + vulnerability_data,
            "pagination": {"total_items": len(vulnerability_data) * 2},
        }
    )
    assert response == expected_response

    assert gitlab_client_mock.apost.call_count == 2


@pytest.mark.asyncio
async def test_list_vulnerabilities_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API Error"))

    tool = ListVulnerabilities(metadata=metadata)

    response = await tool.arun({"project_full_path": "namespace/project"})

    error_response = json.loads(response)
    assert "error" in error_response
    assert "API Error" in error_response["error"]


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListVulnerabilitiesInput(project_full_path="namespace/project"),
            "List vulnerabilities in project namespace/project",
        ),
        (
            ListVulnerabilitiesInput(project_full_path="group/subgroup/project"),
            "List vulnerabilities in project group/subgroup/project",
        ),
    ],
)
def test_list_vulnerabilities_format_display_message(input_data, expected_message):
    tool = ListVulnerabilities(metadata={})
    assert tool.format_display_message(input_data) == expected_message


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

    # editorconfig-checker-disable
    expected_mutation = """
        mutation($vulnerabilityId: VulnerabilityID!, $comment: String, $dismissalReason: VulnerabilityDismissalReason) {
          vulnerabilityDismiss(input: {
            id: $vulnerabilityId,
            comment: $comment,
            dismissalReason: $dismissalReason
          }) {
            errors
            vulnerability {
              id
              description
              state
              dismissedAt
              dismissalReason
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
                    "comment": "Security review deemed this a false positive",
                    "dismissalReason": "FALSE_POSITIVE",
                },
            }
        ),
    )


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
    body = json.loads(call_args[1]["body"])
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
