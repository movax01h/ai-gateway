import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.vulnerabilities.severity import (
    UpdateVulnerabilitySeverity,
    UpdateVulnerabilitySeverityInput,
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


@pytest.fixture
def tool(metadata):
    return UpdateVulnerabilitySeverity(metadata=metadata)


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
async def test_update_vulnerability_severity_arun(
    gitlab_client_mock, metadata, vulnerability_data
):
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilitiesSeverityOverride": {
                    "errors": [],
                    "vulnerabilities": [
                        {"id": "gid://gitlab/Vulnerability/540", "severity": "CRITICAL"}
                    ],
                }
            }
        }
    )

    tool = UpdateVulnerabilitySeverity(metadata=metadata)

    input_data = {
        "vulnerability_ids": ["gid://gitlab/Vulnerability/540"],
        "severity": "CRITICAL",
        "comment": "Testing severity override via GraphQL explorer",
    }

    response = await tool.arun(input_data)

    expected_response = json.dumps(
        {
            "success": True,
            "updated_vulnerabilities": [
                {"id": "gid://gitlab/Vulnerability/540", "severity": "CRITICAL"}
            ],
            "message": "Successfully updated severity to CRITICAL for 1 vulnerability(s)",
        }
    )

    assert response == expected_response

    # editorconfig-checker-disable
    expected_query = """
        mutation vulnerabilitiesSeverityOverride($vulnerabilityIds: [VulnerabilityID!]!, $severity: VulnerabilitySeverity!, $comment: String!) {
          vulnerabilitiesSeverityOverride(
            input: {
              vulnerabilityIds: $vulnerabilityIds,
              severity: $severity,
              comment: $comment
            }
          ) {
            errors
            vulnerabilities {
              id
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
                "query": expected_query,
                "variables": {
                    "vulnerabilityIds": ["gid://gitlab/Vulnerability/540"],
                    "severity": "CRITICAL",
                    "comment": "Testing severity override via GraphQL explorer",
                },
            }
        ),
    )


@pytest.mark.parametrize(
    "vulnerability_ids,should_fail",
    [
        (["id123"], False),
        (["id1", "id2", "id3"], False),
        (["gid://gitlab/Vulnerability/123"], False),
        (["", "valid_id", None], False),  # At least one valid
        ([123], False),
        ([], True),
        ([""], True),
        (["", " ", None], True),
        ([None, None], True),
        ({}, True),
        (None, True),
    ],
)
def test_vulnerability_ids_validation(vulnerability_ids, should_fail, tool):
    if should_fail:
        with pytest.raises(ValueError):
            tool.validate_inputs(vulnerability_ids, "HIGH", "comment")
    else:
        tool.validate_inputs(vulnerability_ids, "HIGH", "comment")


@pytest.mark.parametrize(
    "severity,should_fail",
    [
        ("CRITICAL", False),
        ("HIGH", False),
        ("MEDIUM", False),
        ("LOW", False),
        ("INFO", False),
        ("UNKNOWN", False),
        ("INVALID", True),
        ("", True),
        (123, True),
        (["HIGH"], True),
        (None, True),
    ],
)
def test_severity_validation(severity, should_fail, tool):
    if should_fail:
        with pytest.raises(ValueError):
            tool.validate_inputs(["id"], severity, "comment")
    else:
        tool.validate_inputs(["id"], severity, "comment")


@pytest.mark.parametrize(
    "comment,should_fail",
    [
        ("Valid comment", False),
        ("", False),  # Empty comment is valid
        ("x" * 50000, False),  # Exactly at limit
        ("x" * 50001, True),  # Over limit
        (None, True),
        (123, True),
        (["comment"], True),
        ({"comment": "text"}, True),
    ],
)
def test_comment_validation(comment, should_fail, tool):
    if should_fail:
        with pytest.raises(ValueError):
            tool.validate_inputs(["id"], "HIGH", comment)
    else:
        tool.validate_inputs(["id"], "HIGH", comment)
