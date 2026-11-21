import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from duo_workflow_service.tools.security_inputs import (
    MAX_GITLAB_GLOBAL_ID_LENGTH,
    MAX_VULNERABILITY_COMMENT_LENGTH,
    MAX_VULNERABILITY_IDS,
)
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


@pytest.fixture(name="vulnerability_data")
def vulnerability_data_fixture():
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


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture(name="tool")
def tool_fixture(metadata):
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
@pytest.mark.usefixtures("vulnerability_data")
async def test_update_vulnerability_severity_arun(gitlab_client_mock, metadata):
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
        (["", "valid_id"], False),  # At least one valid
        ([], True),
        ([""], True),
        (["", " "], True),
    ],
)
def test_vulnerability_ids_validation(vulnerability_ids, should_fail, tool):
    if should_fail:
        with pytest.raises(ValueError):
            tool.validate_inputs(vulnerability_ids)
    else:
        tool.validate_inputs(vulnerability_ids)


def _severity_input(**overrides):
    args = {
        "vulnerability_ids": ["gid://gitlab/Vulnerability/123"],
        "severity": "HIGH",
        "comment": "comment",
    }
    args.update(overrides)
    return UpdateVulnerabilitySeverityInput(**args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"severity": severity}
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN"]
    ]
    + [
        {"comment": ""},
        {"comment": "x" * MAX_VULNERABILITY_COMMENT_LENGTH},
        {"vulnerability_ids": ["id"] * MAX_VULNERABILITY_IDS},
        {"vulnerability_ids": ["x" * MAX_GITLAB_GLOBAL_ID_LENGTH]},
    ],
)
def test_input_schema_accepts_valid_values(overrides):
    _severity_input(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"severity": "INVALID"},
        {"severity": ""},
        {"severity": "high"},
        {"severity": 123},
        {"severity": ["HIGH"]},
        {"severity": None},
        {"comment": "x" * (MAX_VULNERABILITY_COMMENT_LENGTH + 1)},
        {"comment": None},
        {"comment": 123},
        {"comment": ["comment"]},
        {"comment": {"comment": "text"}},
        {"vulnerability_ids": ["id"] * (MAX_VULNERABILITY_IDS + 1)},
        {"vulnerability_ids": ["x" * (MAX_GITLAB_GLOBAL_ID_LENGTH + 1)]},
        {"vulnerability_ids": [123]},
        {"vulnerability_ids": None},
        {"vulnerability_ids": "gid://gitlab/Vulnerability/123"},
    ],
)
def test_input_schema_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        _severity_input(**overrides)


@pytest.mark.asyncio
async def test_update_vulnerability_severity_rejects_oversized_comment_before_request(
    gitlab_client_mock, metadata
):
    tool = UpdateVulnerabilitySeverity(metadata=metadata)

    with pytest.raises(ValidationError):
        await tool.arun(
            {
                "vulnerability_ids": ["gid://gitlab/Vulnerability/123"],
                "severity": "HIGH",
                "comment": "x" * (MAX_VULNERABILITY_COMMENT_LENGTH + 1),
            }
        )

    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_update_vulnerability_severity_graphql_errors(
    gitlab_client_mock, metadata
):
    """Test that GraphQL errors raise ToolException."""
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "errors": [
                {
                    "message": "GraphQL validation error",
                    "locations": [{"line": 1, "column": 1}],
                }
            ]
        }
    )

    tool = UpdateVulnerabilitySeverity(metadata=metadata)

    input_data = {
        "vulnerability_ids": ["gid://gitlab/Vulnerability/540"],
        "severity": "CRITICAL",
        "comment": "Test comment",
    }

    # Should raise ToolException instead of returning error JSON
    with pytest.raises(ToolException) as exc_info:
        await tool.arun(input_data)

    assert "GraphQL errors:" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_vulnerability_severity_mutation_errors(
    gitlab_client_mock, metadata
):
    """Test that mutation errors raise ToolException."""
    gitlab_client_mock.apost = AsyncMock(
        return_value={
            "data": {
                "vulnerabilitiesSeverityOverride": {
                    "errors": ["Mutation failed: Invalid severity"],
                    "vulnerabilities": [],
                }
            }
        }
    )

    tool = UpdateVulnerabilitySeverity(metadata=metadata)

    input_data = {
        "vulnerability_ids": ["gid://gitlab/Vulnerability/540"],
        "severity": "CRITICAL",
        "comment": "Test comment",
    }

    # Should raise ToolException instead of returning error JSON
    with pytest.raises(ToolException) as exc_info:
        await tool.arun(input_data)

    assert "Mutation errors:" in str(exc_info.value)
