import base64
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.repository_files import (
    GetRepositoryFile,
    ListRepositoryTree,
    RepositoryFileResourceInput,
    RepositoryTreeResourceInput,
)


@pytest.fixture(autouse=True)
def mock_feature_flag():
    """Mock feature flag to return True for USE_DUO_CONTEXT_EXCLUSION."""
    with patch(
        "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled"
    ) as mock:
        mock.return_value = True
        yield mock


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = AsyncMock()
    return mock


@pytest.fixture(name="binary_detection_tool")
def binary_detection_tool_fixture(metadata):
    tool = GetRepositoryFile(metadata=metadata)
    return tool


@pytest.fixture(name="tree_tool")
def tree_tool_fixture(metadata):
    tool = ListRepositoryTree(metadata=metadata)
    return tool


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture
def metadata_with_project(gitlab_client_mock):
    """Metadata with a mock project that has exclusion rules."""
    project = {"exclusion_rules": ["*.log", "node_modules/", "*.tmp", "secret_*"]}
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project,
    }


@pytest.fixture
def metadata_no_exclusion_rules(gitlab_client_mock):
    """Metadata with a project that has no exclusion rules."""
    project = {"exclusion_rules": []}
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project,
    }


@pytest.fixture(name="tool")
def tool_fixture(metadata):
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
            "/api/v4/projects/gitlab-org/gitlab/repository/files/README.md",
            "master",
            json.dumps(
                {
                    "content": base64.b64encode("file content".encode("utf-8")).decode(
                        "utf-8"
                    )
                }
            ),
            {"content": "file content"},
        ),
        (
            {"url": "https://gitlab.com/namespace/project/-/blob/master/README.md"},
            "/api/v4/projects/namespace%2Fproject/repository/files/README.md",
            "master",
            json.dumps(
                {
                    "content": base64.b64encode("file content".encode("utf-8")).decode(
                        "utf-8"
                    )
                }
            ),
            {"content": "file content"},
        ),
        (
            {
                "project_id": "gitlab-org/gitlab",
                "ref": "master",
                "file_path": "special_chars.py",
            },
            "/api/v4/projects/gitlab-org/gitlab/repository/files/special_chars.py",
            "master",
            json.dumps(
                {
                    "content": base64.b64encode(
                        "# -*- coding: utf-8 -*-\n\ndef λ_function():\n    return '你好，世界！'".encode(
                            "utf-8"
                        )
                    ).decode("utf-8")
                }
            ),
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
            "/api/v4/projects/gitlab-org/gitlab/repository/files/file%2Fpath%2Fwith%2Fslashes",
            "master",
            json.dumps(
                {
                    "content": base64.b64encode("file content".encode("utf-8")).decode(
                        "utf-8"
                    )
                }
            ),
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
                "mock_value": json.dumps(
                    {
                        "content": base64.b64encode(
                            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
                        ).decode("latin-1")
                    }
                ),
            },
            "'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte",
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


# FileExclusionPolicy Tests
class TestFileExclusionPolicy:
    """Test FileExclusionPolicy functionality in repository tools."""

    @pytest.mark.parametrize(
        "file_path,exclusion_rules,expected_allowed",
        [
            # Basic pattern matching
            ("README.md", ["*.log"], True),
            ("debug.log", ["*.log"], False),
            ("app.log", ["*.log"], False),
            # Directory patterns
            ("node_modules/package.json", ["node_modules/"], False),
            ("src/node_modules/lib.js", ["node_modules/"], False),
            ("src/main.js", ["node_modules/"], True),
            # Multiple patterns
            ("config.tmp", ["*.log", "*.tmp"], False),
            ("debug.log", ["*.log", "*.tmp"], False),
            ("main.py", ["*.log", "*.tmp"], True),
            # Prefix patterns
            ("secret_key.txt", ["secret_*"], False),
            ("secret_config.json", ["secret_*"], False),
            ("public_key.txt", ["secret_*"], True),
            # Path normalization
            ("folder\\file.log", ["*.log"], False),
            ("/absolute/path/file.log", ["*.log"], False),
            ("./relative/file.log", ["*.log"], False),
            # Empty rules
            ("any_file.txt", [], True),
            ("debug.log", [], True),
        ],
    )
    def test_file_exclusion_policy_is_allowed(
        self, file_path, exclusion_rules, expected_allowed
    ):
        """Test FileExclusionPolicy.is_allowed with various patterns."""
        project = {"exclusion_rules": exclusion_rules}
        policy = FileExclusionPolicy(project)

        assert policy.is_allowed(file_path) == expected_allowed

    def test_file_exclusion_policy_no_project(self):
        """Test FileExclusionPolicy with None project."""
        policy = FileExclusionPolicy(None)

        assert policy.is_allowed("any_file.txt") is True
        assert policy.is_allowed("debug.log") is True

    def test_file_exclusion_policy_no_exclusion_rules(self):
        """Test FileExclusionPolicy with project but no exclusion rules."""
        project = {}
        policy = FileExclusionPolicy(project)

        assert policy.is_allowed("any_file.txt") is True
        assert policy.is_allowed("debug.log") is True

    @pytest.mark.parametrize(
        "filenames,exclusion_rules,expected_allowed",
        [
            # Basic filtering
            (
                ["README.md", "debug.log", "main.py"],
                ["*.log"],
                ["README.md", "main.py"],
            ),
            # Multiple patterns
            (
                ["config.json", "debug.log", "temp.tmp", "main.py"],
                ["*.log", "*.tmp"],
                ["config.json", "main.py"],
            ),
            # Directory filtering
            (
                ["src/main.py", "node_modules/lib.js", "package.json"],
                ["node_modules/"],
                ["src/main.py", "package.json"],
            ),
            # Empty input
            ([], ["*.log"], []),
            # No exclusions
            (["file1.txt", "file2.log"], [], ["file1.txt", "file2.log"]),
            # Whitespace handling
            (["  README.md  ", "\tdebug.log\t", "\n", "  "], ["*.log"], ["README.md"]),
        ],
    )
    def test_file_exclusion_policy_filter_allowed(
        self, filenames, exclusion_rules, expected_allowed
    ):
        """Test FileExclusionPolicy.filter_allowed with various inputs."""
        project = {"exclusion_rules": exclusion_rules}
        policy = FileExclusionPolicy(project)

        allowed_files, excluded_files = policy.filter_allowed(filenames)
        assert allowed_files == expected_allowed

        # Verify that excluded files are correctly identified
        expected_excluded = [
            f.strip()
            for f in filenames
            if f.strip() and f.strip() not in expected_allowed
        ]
        assert excluded_files == expected_excluded

    def test_format_user_exclusion_message(self):
        """Test user-facing exclusion message formatting."""
        blocked_files = ["secret_key.txt", "debug.log"]
        message = FileExclusionPolicy.format_user_exclusion_message(blocked_files)

        expected = " - files excluded:\nsecret_key.txt\ndebug.log"
        assert message == expected

    def test_format_llm_exclusion_message(self):
        """Test LLM-facing exclusion message formatting."""
        blocked_files = ["secret_key.txt", "debug.log"]
        message = FileExclusionPolicy.format_llm_exclusion_message(blocked_files)

        expected = "Files excluded due to policy, continue without files:\nsecret_key.txt\ndebug.log"
        assert message == expected

    def test_is_allowed_for_project_static_method(self):
        """Test static method is_allowed_for_project."""
        project = {"exclusion_rules": ["*.log"]}

        assert FileExclusionPolicy.is_allowed_for_project(project, "README.md") is True
        assert FileExclusionPolicy.is_allowed_for_project(project, "debug.log") is False
        assert FileExclusionPolicy.is_allowed_for_project(None, "any_file.txt") is True


# Integration Tests for FileExclusionPolicy with GetRepositoryFile
class TestGetRepositoryFileWithExclusion:
    """Test GetRepositoryFile tool with FileExclusionPolicy integration."""

    @pytest.fixture
    def mock_content(self):
        return json.dumps(
            {
                "content": base64.b64encode("file content".encode("utf-8")).decode(
                    "utf-8"
                )
            }
        )

    @pytest.mark.asyncio
    async def test_get_file_blocked_by_policy(self, metadata_with_project):
        """Test that GetRepositoryFile blocks files matching exclusion rules."""
        tool = GetRepositoryFile(metadata=metadata_with_project)

        input_params = {
            "project_id": "test/project",
            "ref": "main",
            "file_path": "debug.log",  # This should be blocked by *.log rule
        }

        result = await tool._arun(**input_params)
        response = json.loads(result)

        assert "error" in response
        assert "Files excluded due to policy" in response["error"]
        assert "debug.log" in response["error"]

    @pytest.mark.asyncio
    async def test_get_file_allowed_by_policy(
        self, metadata_with_project, gitlab_client_mock, mock_content
    ):
        """Test that GetRepositoryFile allows files not matching exclusion rules."""
        tool = GetRepositoryFile(metadata=metadata_with_project)
        gitlab_client_mock.aget.return_value = mock_content

        input_params = {
            "project_id": "test/project",
            "ref": "main",
            "file_path": "README.md",  # This should be allowed
        }

        result = await tool._arun(**input_params)
        response = json.loads(result)

        assert "content" in response
        assert response["content"] == "file content"
        gitlab_client_mock.aget.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_file_no_exclusion_rules(
        self, metadata_no_exclusion_rules, gitlab_client_mock, mock_content
    ):
        """Test GetRepositoryFile with no exclusion rules."""
        tool = GetRepositoryFile(metadata=metadata_no_exclusion_rules)
        gitlab_client_mock.aget.return_value = mock_content

        input_params = {
            "project_id": "test/project",
            "ref": "main",
            "file_path": "debug.log",  # Should be allowed when no rules
        }

        result = await tool._arun(**input_params)
        response = json.loads(result)

        assert "content" in response
        assert response["content"] == "file content"

    @pytest.mark.asyncio
    async def test_get_file_no_project(
        self, metadata, gitlab_client_mock, mock_content
    ):
        """Test GetRepositoryFile with no project (no exclusion policy)."""
        tool = GetRepositoryFile(metadata=metadata)
        gitlab_client_mock.aget.return_value = mock_content

        input_params = {
            "project_id": "test/project",
            "ref": "main",
            "file_path": "debug.log",  # Should be allowed when no project
        }

        result = await tool._arun(**input_params)
        response = json.loads(result)

        assert "content" in response
        assert response["content"] == "file content"

    def test_get_file_format_display_message_blocked(self, metadata_with_project):
        """Test display message formatting for blocked files."""
        tool = GetRepositoryFile(metadata=metadata_with_project)

        args = RepositoryFileResourceInput(
            project_id="test/project",
            ref="main",
            file_path="secret_key.txt",  # Blocked by secret_* rule
        )

        message = tool.format_display_message(args)
        assert " - files excluded:" in message
        assert "secret_key.txt" in message

    def test_get_file_format_display_message_allowed(self, metadata_with_project):
        """Test display message formatting for allowed files."""
        tool = GetRepositoryFile(metadata=metadata_with_project)

        args = RepositoryFileResourceInput(
            project_id="test/project", ref="main", file_path="README.md"  # Allowed file
        )

        message = tool.format_display_message(args)
        assert (
            "Get repository file README.md from project test/project at ref main"
            == message
        )

    @pytest.mark.parametrize(
        "file_path,should_be_blocked",
        [
            ("debug.log", True),  # *.log rule
            ("app.log", True),  # *.log rule
            ("node_modules/lib.js", True),  # node_modules/ rule
            ("temp.tmp", True),  # *.tmp rule
            ("secret_config.json", True),  # secret_* rule
            ("README.md", False),  # Allowed
            ("src/main.py", False),  # Allowed
            ("config.json", False),  # Allowed
        ],
    )
    @pytest.mark.asyncio
    async def test_get_file_various_exclusion_patterns(
        self,
        metadata_with_project,
        gitlab_client_mock,
        file_path,
        should_be_blocked,
        mock_content,
    ):
        """Test GetRepositoryFile with various exclusion patterns."""
        tool = GetRepositoryFile(metadata=metadata_with_project)
        gitlab_client_mock.aget.return_value = mock_content

        input_params = {
            "project_id": "test/project",
            "ref": "main",
            "file_path": file_path,
        }

        result = await tool._arun(**input_params)
        response = json.loads(result)

        if should_be_blocked:
            assert "error" in response
            assert "Files excluded due to policy" in response["error"]
            gitlab_client_mock.aget.assert_not_called()
        else:
            assert "content" in response
            assert response["content"] == "file content"
            gitlab_client_mock.aget.assert_called_once()


class TestListRepositoryTreeWithExclusion:
    """Test ListRepositoryTree tool with FileExclusionPolicy integration."""

    @pytest.mark.asyncio
    async def test_list_tree_filters_excluded_files(
        self, metadata_with_project, gitlab_client_mock
    ):
        """Test that ListRepositoryTree filters out excluded files."""
        tool = ListRepositoryTree(metadata=metadata_with_project)

        # Mock response with mixed allowed and excluded files
        mock_response = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
            {
                "id": "4",
                "name": "lib.js",
                "type": "blob",
                "path": "node_modules/lib.js",
            },
            {"id": "5", "name": "config.tmp", "type": "blob", "path": "config.tmp"},
            {
                "id": "6",
                "name": "secret_key.txt",
                "type": "blob",
                "path": "secret_key.txt",
            },
        ]
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # Should only include allowed files
        expected_files = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        assert response["tree"] == expected_files

    @pytest.mark.asyncio
    async def test_list_tree_no_exclusion_rules(
        self, metadata_no_exclusion_rules, gitlab_client_mock
    ):
        """Test ListRepositoryTree with no exclusion rules."""
        tool = ListRepositoryTree(metadata=metadata_no_exclusion_rules)

        mock_response = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # All files should be included
        assert response["tree"] == mock_response

    @pytest.mark.asyncio
    async def test_list_tree_empty_response(
        self, metadata_with_project, gitlab_client_mock
    ):
        """Test ListRepositoryTree with empty API response."""
        tool = ListRepositoryTree(metadata=metadata_with_project)

        gitlab_client_mock.aget.return_value = []

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        assert response["tree"] == []

    @pytest.mark.asyncio
    async def test_list_tree_all_files_excluded(
        self, metadata_with_project, gitlab_client_mock
    ):
        """Test ListRepositoryTree when all files are excluded."""
        tool = ListRepositoryTree(metadata=metadata_with_project)

        # All files match exclusion patterns
        mock_response = [
            {"id": "1", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "2", "name": "app.log", "type": "blob", "path": "app.log"},
            {
                "id": "3",
                "name": "lib.js",
                "type": "blob",
                "path": "node_modules/lib.js",
            },
            {"id": "4", "name": "temp.tmp", "type": "blob", "path": "temp.tmp"},
            {
                "id": "5",
                "name": "secret_config.json",
                "type": "blob",
                "path": "secret_config.json",
            },
        ]
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        assert response["tree"] == []

    @pytest.mark.asyncio
    async def test_list_tree_complex_patterns(self, gitlab_client_mock):
        """Test ListRepositoryTree with complex exclusion patterns."""
        # Custom project with more complex rules
        project = {
            "exclusion_rules": [
                "*.log",
                "*.tmp",
                "node_modules/",
                "build/",
                "dist/",
                "__pycache__/",
                "*.pyc",
                ".git/",
                "secret_*",
                "*.key",
                "test_*.py",
            ]
        }
        metadata = {
            "gitlab_client": gitlab_client_mock,
            "gitlab_host": "gitlab.com",
            "project": project,
        }
        tool = ListRepositoryTree(metadata=metadata)

        mock_response = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "main.py", "type": "blob", "path": "src/main.py"},
            {"id": "3", "name": "utils.py", "type": "blob", "path": "src/utils.py"},
            {
                "id": "4",
                "name": "test_helper.py",
                "type": "blob",
                "path": "test_helper.py",
            },
            {"id": "5", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "6", "name": "output.js", "type": "blob", "path": "build/output.js"},
            {
                "id": "7",
                "name": "index.js",
                "type": "blob",
                "path": "node_modules/react/index.js",
            },
            {
                "id": "8",
                "name": "module.pyc",
                "type": "blob",
                "path": "__pycache__/module.pyc",
            },
            {
                "id": "9",
                "name": "secret_config.json",
                "type": "blob",
                "path": "secret_config.json",
            },
            {"id": "10", "name": "api.key", "type": "blob", "path": "api.key"},
            {"id": "11", "name": "config", "type": "blob", "path": ".git/config"},
            {"id": "12", "name": "bundle.js", "type": "blob", "path": "dist/bundle.js"},
            {"id": "13", "name": "temp.tmp", "type": "blob", "path": "temp.tmp"},
        ]

        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # Only these should remain after filtering
        expected_files = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "main.py", "type": "blob", "path": "src/main.py"},
            {"id": "3", "name": "utils.py", "type": "blob", "path": "src/utils.py"},
        ]
        assert response["tree"] == expected_files

    @pytest.mark.asyncio
    async def test_list_tree_no_project(self, metadata, gitlab_client_mock):
        """Test ListRepositoryTree with no project (no exclusion policy)."""
        tool = ListRepositoryTree(metadata=metadata)

        mock_response = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # All files should be included when no project
        assert response["tree"] == mock_response
