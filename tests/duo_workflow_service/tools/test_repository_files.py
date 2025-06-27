import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.tools.repository_files import (
    GetRepositoryFile,
    ListRepositoryTree,
    RepositoryFileResourceInput,
    RepositoryTreeResourceInput,
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
def tree_tool(metadata):
    tool = ListRepositoryTree(metadata=metadata)
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
        (
            {
                "project_id": "gitlab-org/gitlab",
                "ref": "master",
                "file_path": "file/path/with/slashes",
            },
            "/api/v4/projects/gitlab-org/gitlab/repository/files/file%2Fpath%2Fwith%2Fslashes/raw",
            "master",
            "file content",
            {"content": "file content"},
        ),
    ],
    ids=[
        "Explicit params",
        "URL parameter",
        "Special characters in text content",
        "Path with slashes",
    ],
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
        (
            {"url": "https://gitlab.com/namespace/project"},
            {
                "mock_type": "validate_error",
                "mock_value": (None, None, None, []),
            },
            "Missing file_path",
        ),
    ],
    ids=[
        "URL parsing error",
        "API error",
        "Binary file detection",
        "Missing file path",
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
def test_tree_format_display_message(tool, args, expected_message):
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


# Test cases for successful tree listing
@pytest.mark.parametrize(
    "input_params,expected_path,expected_params,mock_response,expected_result",
    [
        (
            {"project_id": 1},
            "/api/v4/projects/1/repository/tree",
            {},
            [
                {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
                {"id": "2", "name": "src", "type": "tree", "path": "src"},
            ],
            {
                "tree": [
                    {
                        "id": "1",
                        "name": "README.md",
                        "type": "blob",
                        "path": "README.md",
                    },
                    {"id": "2", "name": "src", "type": "tree", "path": "src"},
                ]
            },
        ),
        (
            {"url": "https://gitlab.com/namespace/project"},
            "/api/v4/projects/namespace%2Fproject/repository/tree",
            {},
            [{"id": "1", "name": "file.py", "type": "blob", "path": "file.py"}],
            {
                "tree": [
                    {"id": "1", "name": "file.py", "type": "blob", "path": "file.py"}
                ]
            },
        ),
        (
            {
                "project_id": 2,
                "path": "src",
                "ref": "main",
                "recursive": "true",
                "page": 2,
                "per_page": 50,
            },
            "/api/v4/projects/2/repository/tree",
            {
                "path": "src",
                "ref": "main",
                "recursive": "true",
                "page": 2,
                "per_page": 50,
            },
            [
                {"id": "3", "name": "utils.py", "type": "blob", "path": "src/utils.py"},
                {"id": "4", "name": "models", "type": "tree", "path": "src/models"},
            ],
            {
                "tree": [
                    {
                        "id": "3",
                        "name": "utils.py",
                        "type": "blob",
                        "path": "src/utils.py",
                    },
                    {"id": "4", "name": "models", "type": "tree", "path": "src/models"},
                ]
            },
        ),
        (
            {
                "project_id": 3,
                "path": "docs",
                "ref": "develop",
            },
            "/api/v4/projects/3/repository/tree",
            {"path": "docs", "ref": "develop"},
            [],
            {"tree": []},
        ),
        (
            {
                "project_id": 4,
                "recursive": "false",
                "page": 1,
                "per_page": 10,
            },
            "/api/v4/projects/4/repository/tree",
            {"recursive": "false", "page": 1, "per_page": 10},
            [{"id": "5", "name": "test.txt", "type": "blob", "path": "test.txt"}],
            {
                "tree": [
                    {"id": "5", "name": "test.txt", "type": "blob", "path": "test.txt"}
                ]
            },
        ),
    ],
    ids=[
        "Basic project listing",
        "URL parameter",
        "All optional parameters",
        "Empty tree response",
        "Pagination parameters",
    ],
)
@pytest.mark.asyncio
async def test_list_repository_tree_success(
    tree_tool,
    gitlab_client_mock,
    input_params,
    expected_path,
    expected_params,
    mock_response,
    expected_result,
):
    gitlab_client_mock.aget.return_value = mock_response

    result = await tree_tool._arun(**input_params)

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params=expected_params,
    )

    assert json.loads(result) == expected_result


@pytest.mark.parametrize(
    "input_params,mock_setup,expected_error_contains",
    [
        (
            {"url": "https://gitlab.com/invalid"},
            {
                "mock_type": "validate_error",
                "mock_value": (None, ["Failed to parse URL"]),
            },
            "Failed to parse URL",
        ),
        (
            {},
            {
                "mock_type": "validate_error",
                "mock_value": (
                    None,
                    ["'project_id' must be provided when 'url' is not"],
                ),
            },
            "'project_id' must be provided when 'url' is not",
        ),
        (
            {"project_id": 1},
            {"mock_type": "api_error", "mock_value": Exception("Repository not found")},
            "Repository not found",
        ),
        (
            {"project_id": 2},
            {"mock_type": "api_error", "mock_value": Exception("Permission denied")},
            "Permission denied",
        ),
    ],
    ids=[
        "URL parsing error",
        "Missing project_id",
        "Repository not found",
        "Permission denied",
    ],
)
@pytest.mark.asyncio
async def test_list_repository_tree_errors(
    tree_tool, gitlab_client_mock, input_params, mock_setup, expected_error_contains
):
    if mock_setup["mock_type"] == "validate_error":
        tree_tool._validate_project_url = lambda url, project_id: mock_setup[
            "mock_value"
        ]
    elif mock_setup["mock_type"] == "api_error":
        gitlab_client_mock.aget.side_effect = mock_setup["mock_value"]

    result = await tree_tool._arun(**input_params)
    error_response = json.loads(result)

    assert "error" in error_response
    assert expected_error_contains in error_response["error"]


@pytest.mark.parametrize(
    "args,expected_message",
    [
        (
            RepositoryTreeResourceInput(project_id="gitlab-org/gitlab"),
            "List repository tree in project gitlab-org/gitlab",
        ),
        (
            RepositoryTreeResourceInput(
                project_id="gitlab-org/gitlab", path="src", ref="main"
            ),
            "List repository tree in path 'src' at ref 'main' in project gitlab-org/gitlab",
        ),
        (
            RepositoryTreeResourceInput(project_id="gitlab-org/gitlab", recursive=True),
            "List repository tree recursively in project gitlab-org/gitlab",
        ),
        (
            RepositoryTreeResourceInput(
                project_id="gitlab-org/gitlab",
                path="docs",
                ref="develop",
                recursive=True,
            ),
            "List repository tree recursively in path 'docs' at ref 'develop' in project gitlab-org/gitlab",
        ),
        (
            RepositoryTreeResourceInput(url="https://gitlab.com/namespace/project"),
            "List repository tree from https://gitlab.com/namespace/project",
        ),
        (
            RepositoryTreeResourceInput(
                url="https://gitlab.com/namespace/project", path="src", recursive=True
            ),
            "List repository tree recursively in path 'src' from https://gitlab.com/namespace/project",
        ),
    ],
)
def test_format_display_message(tree_tool, args, expected_message):
    msg = tree_tool.format_display_message(args)
    assert msg == expected_message


# Test parameter validation
@pytest.mark.parametrize(
    "input_params,should_pass",
    [
        ({"project_id": "gitlab-org/gitlab"}, True),
        ({"url": "https://gitlab.com/namespace/project"}, True),
        ({"project_id": "gitlab-org/gitlab", "page": 1}, True),
        ({"project_id": "gitlab-org/gitlab", "per_page": 50}, True),
        ({"project_id": "gitlab-org/gitlab", "page": 1, "per_page": 100}, True),
        ({"project_id": "gitlab-org/gitlab", "recursive": True}, True),
        ({"project_id": "gitlab-org/gitlab", "path": "src/main"}, True),
        ({"project_id": "gitlab-org/gitlab", "ref": "feature-branch"}, True),
    ],
)
def test_parameter_validation(input_params, should_pass):
    """Test that RepositoryTreeResourceInput validates parameters correctly."""
    if should_pass:
        RepositoryTreeResourceInput(**input_params)
    else:
        with pytest.raises(Exception):
            RepositoryTreeResourceInput(**input_params)


# Test edge cases for optional parameters
@pytest.mark.parametrize(
    "input_params,expected_params_subset",
    [
        (
            {"project_id": "test", "path": None, "ref": None, "recursive": None},
            {},
        ),
        (
            {"project_id": "test", "path": "", "ref": "", "recursive": False},
            {"path": "", "ref": "", "recursive": "false"},
        ),
        (
            {"project_id": "test", "page": None, "per_page": None},
            {},
        ),
    ],
)
@pytest.mark.asyncio
async def test_optional_parameters_handling(
    tree_tool, gitlab_client_mock, input_params, expected_params_subset
):
    """Test that None values are not included in API params."""
    gitlab_client_mock.aget.return_value = []

    await tree_tool._arun(**input_params)

    call_args = gitlab_client_mock.aget.call_args
    actual_params = call_args[1]["params"]

    for key, value in expected_params_subset.items():
        assert actual_params.get(key) == value

    for value in actual_params.values():
        assert value is not None


# Test with complex tree structure
@pytest.mark.asyncio
async def test_complex_tree_structure(tree_tool, gitlab_client_mock):
    """Test with a complex tree structure containing various file types."""
    complex_tree = [
        {
            "id": "a1b2c3d4",
            "name": "README.md",
            "type": "blob",
            "path": "README.md",
            "mode": "100644",
        },
        {
            "id": "e5f6g7h8",
            "name": "src",
            "type": "tree",
            "path": "src",
            "mode": "040000",
        },
        {
            "id": "i9j0k1l2",
            "name": "LICENSE",
            "type": "blob",
            "path": "LICENSE",
            "mode": "100644",
        },
        {
            "id": "m3n4o5p6",
            "name": "docs",
            "type": "tree",
            "path": "docs",
            "mode": "040000",
        },
    ]

    gitlab_client_mock.aget.return_value = complex_tree

    result = await tree_tool._arun(project_id="test/project")

    response = json.loads(result)
    assert "tree" in response
    assert len(response["tree"]) == 4
    assert response["tree"] == complex_tree
