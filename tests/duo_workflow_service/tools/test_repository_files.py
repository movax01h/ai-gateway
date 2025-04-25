import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.tools.repository_files import (
    GetRepositoryFile,
    RepositoryFileResourceInput,
)


@pytest.fixture
def gitlab_client_mock():
    mock = AsyncMock()
    return mock


@pytest.fixture
def binary_detection_tool(metadata):
    tool = GetRepositoryFile(metadata=metadata)
    return tool


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture
def tool(metadata):
    tool = GetRepositoryFile(metadata=metadata)
    return tool


# Test cases for successful file retrieval
@pytest.mark.parametrize(
    "input_params,expected_path,expected_ref,content,expected_result",
    [
        (
            {
                "project_id": "gitlab-org/gitlab",
                "ref": "master",
                "file_path": "README.md",
            },
            "/api/v4/projects/gitlab-org/gitlab/repository/files/README.md/raw",
            "master",
            "file content",
            {"content": "file content"},
        ),
        (
            {"url": "https://gitlab.com/namespace/project/-/blob/master/README.md"},
            "/api/v4/projects/namespace%2Fproject/repository/files/README.md/raw",
            "master",
            "file content",
            {"content": "file content"},
        ),
        (
            {
                "project_id": "gitlab-org/gitlab",
                "ref": "master",
                "file_path": "special_chars.py",
            },
            "/api/v4/projects/gitlab-org/gitlab/repository/files/special_chars.py/raw",
            "master",
            "# -*- coding: utf-8 -*-\n\ndef λ_function():\n    return '你好，世界！'",
            {
                "content": "# -*- coding: utf-8 -*-\n\ndef λ_function():\n    return '你好，世界！'"
            },
        ),
    ],
    ids=["Explicit params", "URL parameter", "Special characters in text content"],
)
@pytest.mark.asyncio
async def test_get_file_success(
    tool,
    gitlab_client_mock,
    input_params,
    expected_path,
    expected_ref,
    content,
    expected_result,
):
    gitlab_client_mock.aget.return_value = content

    result = await tool._arun(**input_params)

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"ref": expected_ref},
        parse_json=False,
    )

    assert json.loads(result) == expected_result


@pytest.mark.parametrize(
    "input_params,mock_setup,expected_error_contains",
    [
        (
            {"url": "https://gitlab.com/namespace/project"},
            {
                "mock_type": "validate_error",
                "mock_value": (None, None, None, ["Failed to parse URL"]),
            },
            "Failed to parse URL",
        ),
        (
            {"project_id": "3", "ref": "master", "file_path": "README.md"},
            {"mock_type": "api_error", "mock_value": Exception("API error")},
            "API error",
        ),
        (
            {
                "project_id": "gitlab-org/gitlab",
                "ref": "master",
                "file_path": "image.png",
            },
            {
                "mock_type": "binary_content",
                "mock_value": b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR".decode(
                    "latin-1"
                ),
            },
            "Binary file detected",
        ),
    ],
    ids=[
        "URL parsing error",
        "API error",
        "Binary file detection",
    ],
)
@pytest.mark.asyncio
async def test_get_file_errors(
    tool, gitlab_client_mock, input_params, mock_setup, expected_error_contains
):
    if mock_setup["mock_type"] == "validate_error":
        tool._validate_repository_file_url = (
            lambda url, project_id, ref, file_path: mock_setup["mock_value"]
        )
    elif mock_setup["mock_type"] == "api_error":
        gitlab_client_mock.aget.side_effect = mock_setup["mock_value"]
    elif mock_setup["mock_type"] == "binary_content":
        gitlab_client_mock.aget.return_value = mock_setup["mock_value"]

    result = await tool._arun(**input_params)
    error_response = json.loads(result)

    assert "error" in error_response
    assert expected_error_contains in error_response["error"]


@pytest.mark.parametrize(
    "args,expected_message",
    [
        (
            RepositoryFileResourceInput(
                project_id="gitlab-org/gitlab", ref="master", file_path="README.md"
            ),
            "Get repository file README.md from project gitlab-org/gitlab at ref master",
        ),
        (
            RepositoryFileResourceInput(
                url="https://gitlab.com/gitlab-org/gitlab/-/blob/master/README.md"
            ),
            "Get repository file content from https://gitlab.com/gitlab-org/gitlab/-/blob/master/README.md",
        ),
    ],
)
def test_format_display_message(tool, args, expected_message):
    msg = tool.format_display_message(args)
    assert msg == expected_message


@pytest.mark.parametrize("host", ["gitlab.com", "gitlab.example.com"])
def test_url_parser_repository_file(host):
    url = f"https://{host}/namespace/project/-/blob/master/path/to/file.py"
    project_path, ref, file_path = GitLabUrlParser.parse_repository_file_url(url, host)

    assert project_path == "namespace%2Fproject"
    assert ref == "master"
    assert file_path == "path/to/file.py"


@pytest.mark.parametrize(
    "content,expected_result",
    [
        ("", False),
        ("Hello, world!", False),
        ("def test_function():\n    return True", False),
        ("Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?", False),
        ("Unicode text: 你好，世界！", False),
        ("Hello\x00World", True),
        (b"\x89PNG\r\n\x1a\n\x00".decode("latin-1"), True),
        ("".join(chr(i) for i in range(0, 31) if i not in (9, 10, 13)), True),
        ("Text" + "".join(chr(i) for i in range(0, 31) if i not in (9, 10, 13)), True),
    ],
)
def test_is_binary_string(binary_detection_tool, content, expected_result):
    result = binary_detection_tool._is_binary_string(content)
    assert result == expected_result
