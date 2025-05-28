import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.ci_linter import CiLinter, CiLinterInput


@pytest.mark.asyncio
async def test_ci_linter_success():
    mock_response = {"valid": True, "errors": [], "warnings": []}
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = mock_response

    tool = CiLinter(metadata={"gitlab_client": gitlab_client_mock})  # type: ignore

    yaml_content = "image: ruby:2.6\ntest_job:\n  script: echo 'test'"
    response = await tool.arun({"project_id": 42, "content": yaml_content})

    assert response == json.dumps(mock_response)
    gitlab_client_mock.apost.assert_called_once()

    call_args = gitlab_client_mock.apost.call_args
    assert call_args[1]["path"] == "/api/v4/projects/42/ci/lint"
    assert json.loads(call_args[1]["body"]) == {"content": yaml_content}


@pytest.mark.asyncio
async def test_ci_linter_with_ref():
    mock_response = {"valid": True, "errors": [], "warnings": []}
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = mock_response

    tool = CiLinter(metadata={"gitlab_client": gitlab_client_mock})  # type: ignore

    yaml_content = (
        "include: '.gitlab-ci-template.yml'\ntest_job:\n  script: echo 'test'"
    )
    response = await tool.arun(
        {"project_id": 42, "content": yaml_content, "ref": "feature-branch"}
    )

    assert response == json.dumps(mock_response)
    gitlab_client_mock.apost.assert_called_once()

    call_args = gitlab_client_mock.apost.call_args
    assert call_args[1]["path"] == "/api/v4/projects/42/ci/lint"
    assert json.loads(call_args[1]["body"]) == {
        "content": yaml_content,
        "ref": "feature-branch",
    }


@pytest.mark.asyncio
async def test_ci_linter_with_empty_ref():
    mock_response = {"valid": True, "errors": [], "warnings": []}
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = mock_response

    tool = CiLinter(metadata={"gitlab_client": gitlab_client_mock})  # type: ignore

    yaml_content = "image: ruby:2.6\ntest_job:\n  script: echo 'test'"
    response = await tool.arun({"project_id": 42, "content": yaml_content, "ref": ""})

    assert response == json.dumps(mock_response)
    gitlab_client_mock.apost.assert_called_once()

    call_args = gitlab_client_mock.apost.call_args
    # Empty ref should not be included in body
    assert json.loads(call_args[1]["body"]) == {"content": yaml_content}


@pytest.mark.asyncio
async def test_ci_linter_invalid_yaml():
    mock_response = {
        "valid": False,
        "errors": ["jobs config should contain at least one visible job"],
        "warnings": [],
        "merged_yaml": '---\n".job":\n  script:\n  - echo "A hidden job"\n',
        "includes": [],
    }

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.apost.return_value = mock_response

    tool = CiLinter(metadata={"gitlab_client": gitlab_client_mock})  # type: ignore

    yaml_content = ".job:\n  script:\n    - echo 'A hidden job'"

    response = await tool.arun({"project_id": 42, "content": yaml_content})

    response_json = json.loads(response)
    assert response_json["valid"] is False
    assert (
        "jobs config should contain at least one visible job" in response_json["errors"]
    )
    call_args = gitlab_client_mock.apost.call_args
    assert json.loads(call_args[1]["body"]) == {"content": yaml_content}


def test_ci_linter_format_display_message():
    tool = CiLinter(description="Validate CI configuration")
    input_data = CiLinterInput(project_id=123, content="image: ruby:2.6")

    message = tool.format_display_message(input_data)

    assert message == "Validate CI/CD YAML configuration in context of project: 123"


def test_ci_linter_format_display_message_with_ref():
    tool = CiLinter(description="Validate CI configuration")
    input_data = CiLinterInput(
        project_id=123, content="image: ruby:2.6", ref="feature-branch"
    )

    message = tool.format_display_message(input_data)

    assert (
        message
        == "Validate CI/CD YAML configuration in context of project: 123 (ref: feature-branch)"
    )


@pytest.mark.asyncio
async def test_validate_project_id():
    tool = CiLinter(description="Validate CI configuration")
    input_data = CiLinterInput(project_id=1, content="")

    project_id, errors = tool._validate_project_url(
        url=None, project_id=input_data.project_id
    )
    assert project_id == "1"
    assert not errors
