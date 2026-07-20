# pylint: disable=too-many-lines
import asyncio
import base64
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.repository_files import (
    GetRepositoryFile,
    GetRepositoryFiles,
    ListRepositoryTree,
    RepositoryFileResourceInput,
    RepositoryFilesResourceInput,
    RepositoryTreeResourceInput,
)


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


@pytest.fixture(name="metadata_with_project")
def metadata_with_project_fixture(gitlab_client_mock):
    """Metadata with a mock project that has exclusion rules."""
    project = {"exclusion_rules": ["*.log", "node_modules/", "*.tmp", "secret_*"]}
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project,
    }


@pytest.fixture(name="metadata_no_exclusion_rules")
def metadata_no_exclusion_rules_fixture(gitlab_client_mock):
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
    # Parse the content to create a proper response
    content_dict = json.loads(content)
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=content_dict,
        headers={"content-type": "application/json"},
    )
    gitlab_client_mock.aget.return_value = mock_response

    result = await tool._arun(**input_params)

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"ref": expected_ref},
        parse_json=True,
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
        "Missing file path",
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("gitlab_client_mock")
async def test_get_file_errors(tool, input_params, mock_setup, expected_error_contains):
    if mock_setup["mock_type"] == "validate_error":
        tool._validate_repository_file_url = lambda url, project_id, ref, file_path: (
            mock_setup["mock_value"]
        )

    with pytest.raises(ToolException, match=expected_error_contains):
        await tool._arun(**input_params)


@pytest.mark.asyncio
async def test_get_file_api_exception_propagates(tool, gitlab_client_mock):
    """Test that API exceptions propagate instead of being swallowed."""
    gitlab_client_mock.aget.side_effect = Exception("API error")

    with pytest.raises(Exception, match="API error"):
        await tool._arun(project_id="3", ref="master", file_path="README.md")


@pytest.mark.asyncio
async def test_get_file_binary_content_exception_propagates(tool, gitlab_client_mock):
    """Test that binary content decode errors propagate instead of being swallowed."""
    binary_content = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").decode(
        "latin-1"
    )
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"content": binary_content},
        headers={"content-type": "application/json"},
    )
    gitlab_client_mock.aget.return_value = mock_response

    with pytest.raises(UnicodeDecodeError):
        await tool._arun(
            project_id="gitlab-org/gitlab", ref="master", file_path="image.png"
        )


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


class TestGetRepositoryFile404Dedup:
    """Gitlab-org/gitlab#604564: repeated requests for a 404'd path short-circuit."""

    @pytest.fixture
    def not_found_response(self):
        return GitLabHttpResponse(
            status_code=404,
            body={"message": "404 File Not Found"},
        )

    @pytest.mark.asyncio
    async def test_repeated_404_short_circuits(
        self, tool, gitlab_client_mock, not_found_response
    ):
        gitlab_client_mock.aget.return_value = not_found_response

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="master", file_path="missing.rb")

        with pytest.raises(
            ToolException, match="already returned 404 earlier in this session"
        ):
            await tool._arun(project_id="3", ref="master", file_path="missing.rb")

        gitlab_client_mock.aget.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_path_still_fetched(
        self, tool, gitlab_client_mock, not_found_response
    ):
        gitlab_client_mock.aget.return_value = not_found_response

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="master", file_path="missing_a.rb")

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="master", file_path="missing_b.rb")

        assert gitlab_client_mock.aget.call_count == 2

    @pytest.mark.asyncio
    async def test_different_ref_still_fetched(
        self, tool, gitlab_client_mock, not_found_response
    ):
        gitlab_client_mock.aget.return_value = not_found_response

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="master", file_path="missing.rb")

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="main", file_path="missing.rb")

        assert gitlab_client_mock.aget.call_count == 2

    @pytest.mark.asyncio
    async def test_different_project_still_fetched(
        self, tool, gitlab_client_mock, not_found_response
    ):
        gitlab_client_mock.aget.return_value = not_found_response

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="3", ref="master", file_path="missing.rb")

        with pytest.raises(ToolException, match="HTTP 404"):
            await tool._arun(project_id="4", ref="master", file_path="missing.rb")

        assert gitlab_client_mock.aget.call_count == 2

    @pytest.mark.asyncio
    async def test_fresh_instance_refetches(
        self, metadata, gitlab_client_mock, not_found_response
    ):
        gitlab_client_mock.aget.return_value = not_found_response

        with pytest.raises(ToolException, match="HTTP 404"):
            await GetRepositoryFile(metadata=metadata)._arun(
                project_id="3", ref="master", file_path="missing.rb"
            )

        with pytest.raises(ToolException, match="HTTP 404"):
            await GetRepositoryFile(metadata=metadata)._arun(
                project_id="3", ref="master", file_path="missing.rb"
            )

        assert gitlab_client_mock.aget.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_fetch_is_not_cached(self, tool, gitlab_client_mock):
        content = base64.b64encode(b"file content").decode("utf-8")
        gitlab_client_mock.aget.return_value = GitLabHttpResponse(
            status_code=200,
            body={"content": content},
        )

        for _ in range(2):
            result = await tool._arun(
                project_id="3", ref="master", file_path="README.md"
            )
            assert json.loads(result) == {"content": "file content"}

        assert gitlab_client_mock.aget.call_count == 2


# Test cases for offset/limit pagination
class TestGetRepositoryFilePagination:
    """Test GetRepositoryFile tool with offset/limit pagination."""

    @pytest.fixture
    def multiline_content(self):
        """A 10-line file for testing pagination."""
        lines = [f"line {i}" for i in range(10)]
        raw = "\n".join(lines)
        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
        return GitLabHttpResponse(
            status_code=200,
            body={"content": encoded},
            headers={"content-type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_no_pagination_returns_full_content(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Without offset/limit, returns the full file."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py"
        )
        content = json.loads(result)["content"]
        assert content == "\n".join(f"line {i}" for i in range(10))

    @pytest.mark.asyncio
    async def test_offset_only(self, tool, gitlab_client_mock, multiline_content):
        """Offset without limit reads from offset to end, with pagination hint."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=7
        )
        content = json.loads(result)["content"]
        expected = "\n".join(f"line {i}" for i in range(7, 10))
        expected += "\n\n[Showing lines 7-9 of 10 total.]"
        assert content == expected

    @pytest.mark.asyncio
    async def test_limit_only(self, tool, gitlab_client_mock, multiline_content):
        """Limit without offset reads first N lines, with continuation hint."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", limit=3
        )
        content = json.loads(result)["content"]
        expected = "\n".join(f"line {i}" for i in range(3))
        expected += (
            "\n\n[Showing lines 0-2 of 10 total. Use offset=3 to continue reading.]"
        )
        assert content == expected

    @pytest.mark.asyncio
    async def test_offset_and_limit(self, tool, gitlab_client_mock, multiline_content):
        """Offset + limit reads a specific window, with continuation hint."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=3, limit=4
        )
        content = json.loads(result)["content"]
        expected = "\n".join(f"line {i}" for i in range(3, 7))
        expected += (
            "\n\n[Showing lines 3-6 of 10 total. Use offset=7 to continue reading.]"
        )
        assert content == expected

    @pytest.mark.asyncio
    async def test_offset_beyond_file_length(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Offset past end of file returns helpful message."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=100
        )
        content = json.loads(result)["content"]
        assert "beyond end of file" in content
        assert "10 lines" in content
        assert "offset=0" in content

    @pytest.mark.asyncio
    async def test_limit_exceeds_remaining_lines(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Limit larger than remaining lines returns what's available, with hint."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=8, limit=100
        )
        content = json.loads(result)["content"]
        expected = "\n".join(f"line {i}" for i in range(8, 10))
        expected += "\n\n[Showing lines 8-9 of 10 total.]"
        assert content == expected

    @pytest.mark.asyncio
    async def test_offset_zero_with_limit(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Offset 0 with limit is equivalent to limit only."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=0, limit=5
        )
        content = json.loads(result)["content"]
        expected = "\n".join(f"line {i}" for i in range(5))
        expected += (
            "\n\n[Showing lines 0-4 of 10 total. Use offset=5 to continue reading.]"
        )
        assert content == expected

    @pytest.mark.asyncio
    async def test_explicit_full_range_returns_no_hint(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Offset=0 + limit=total_lines returns full content with no pagination hint."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=0, limit=10
        )
        content = json.loads(result)["content"]
        assert content == "\n".join(f"line {i}" for i in range(10))
        assert "[Showing" not in content

    @pytest.mark.asyncio
    async def test_trailing_newline_does_not_create_phantom_line(
        self, tool, gitlab_client_mock
    ):
        """File ending with newline should not produce an extra empty line."""
        raw = "line 0\nline 1\nline 2\n"  # trailing newline
        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
        mock_resp = GitLabHttpResponse(
            status_code=200,
            body={"content": encoded},
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_resp
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=0, limit=3
        )
        content = json.loads(result)["content"]
        assert content == "line 0\nline 1\nline 2"

    def test_negative_offset_rejected_by_validation(self):
        """Negative offset is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RepositoryFileResourceInput(
                project_id="org/repo", ref="main", file_path="file.py", offset=-1
            )

    def test_negative_limit_rejected_by_validation(self):
        """Negative limit is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RepositoryFileResourceInput(
                project_id="org/repo", ref="main", file_path="file.py", limit=-1
            )

    def test_zero_limit_rejected_by_validation(self):
        """Zero limit is rejected by Pydantic validation (must be >= 1)."""
        with pytest.raises(ValidationError):
            RepositoryFileResourceInput(
                project_id="org/repo", ref="main", file_path="file.py", limit=0
            )

    @pytest.mark.asyncio
    async def test_empty_file_with_offset_returns_beyond_message(
        self, tool, gitlab_client_mock
    ):
        """Empty file with offset > 0 returns helpful message, not empty string."""
        encoded = base64.b64encode(b"").decode("utf-8")
        mock_resp = GitLabHttpResponse(
            status_code=200,
            body={"content": encoded},
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_resp
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=5
        )
        content = json.loads(result)["content"]
        assert "beyond end of file" in content
        assert "0 lines" in content

    @pytest.mark.asyncio
    async def test_offset_zero_on_large_file_still_auto_paginates(
        self, tool, gitlab_client_mock, large_file_content
    ):
        """Offset=0 without limit should still auto-paginate large files."""
        gitlab_client_mock.aget.return_value = large_file_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", offset=0
        )
        content = json.loads(result)["content"]
        assert (
            "[Showing lines 0-1999 of 3000 total. Use offset=2000 to continue reading.]"
            in content
        )

    @pytest.fixture
    def large_file_content(self):
        """A 3000-line file to test auto-pagination."""
        lines = [f"line {i}" for i in range(3000)]
        raw = "\n".join(lines)
        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
        return GitLabHttpResponse(
            status_code=200,
            body={"content": encoded},
            headers={"content-type": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_large_file_auto_paginates(
        self, tool, gitlab_client_mock, large_file_content
    ):
        """Files exceeding 2000 lines are auto-paginated even without offset/limit."""
        gitlab_client_mock.aget.return_value = large_file_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py"
        )
        content = json.loads(result)["content"]
        assert (
            "[Showing lines 0-1999 of 3000 total. Use offset=2000 to continue reading.]"
            in content
        )
        # Should contain exactly 2000 lines of content + hint
        lines = content.split("\n")
        # Last 2 lines are blank + hint
        assert lines[0] == "line 0"
        assert lines[1999] == "line 1999"

    @pytest.mark.asyncio
    async def test_small_file_no_auto_pagination(
        self, tool, gitlab_client_mock, multiline_content
    ):
        """Files under 2000 lines return full content without auto-pagination."""
        gitlab_client_mock.aget.return_value = multiline_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py"
        )
        content = json.loads(result)["content"]
        assert content == "\n".join(f"line {i}" for i in range(10))
        assert "[Showing" not in content

    @pytest.mark.asyncio
    async def test_explicit_limit_overrides_auto_pagination(
        self, tool, gitlab_client_mock, large_file_content
    ):
        """Explicit limit=3000 returns all lines, overriding auto-pagination."""
        gitlab_client_mock.aget.return_value = large_file_content
        result = await tool._arun(
            project_id="org/repo", ref="main", file_path="file.py", limit=3000
        )
        content = json.loads(result)["content"]
        assert content == "\n".join(f"line {i}" for i in range(3000))
        assert "[Showing" not in content


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
    mock_http_response = GitLabHttpResponse(
        status_code=200,
        body=mock_response,
        headers={"content-type": "application/json"},
    )
    gitlab_client_mock.aget.return_value = mock_http_response

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
    ],
    ids=[
        "URL parsing error",
        "Missing project_id",
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("gitlab_client_mock")
async def test_list_repository_tree_errors(
    tree_tool, input_params, mock_setup, expected_error_contains
):
    if mock_setup["mock_type"] == "validate_error":
        tree_tool._validate_project_url = lambda url, project_id: mock_setup[
            "mock_value"
        ]

    with pytest.raises(ToolException, match=expected_error_contains):
        await tree_tool._arun(**input_params)


@pytest.mark.asyncio
async def test_list_repository_tree_api_exception_propagates(
    tree_tool, gitlab_client_mock
):
    """Test that API exceptions propagate instead of being swallowed."""
    gitlab_client_mock.aget.side_effect = Exception("Repository not found")

    with pytest.raises(Exception, match="Repository not found"):
        await tree_tool._arun(project_id=1)


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
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=[],
        headers={"content-type": "application/json"},
    )
    gitlab_client_mock.aget.return_value = mock_response

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

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=complex_tree,
        headers={"content-type": "application/json"},
    )
    gitlab_client_mock.aget.return_value = mock_response

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
        return GitLabHttpResponse(
            status_code=200,
            body={
                "content": base64.b64encode("file content".encode("utf-8")).decode(
                    "utf-8"
                )
            },
            headers={"content-type": "application/json"},
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

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(**input_params)

        assert "Files excluded due to policy" in str(exc_info.value)
        assert "debug.log" in str(exc_info.value)

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
            project_id="test/project",
            ref="main",
            file_path="README.md",  # Allowed file
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

        if should_be_blocked:
            with pytest.raises(ToolException) as exc_info:
                await tool._arun(**input_params)
            assert "Files excluded due to policy" in str(exc_info.value)
            gitlab_client_mock.aget.assert_not_called()
        else:
            result = await tool._arun(**input_params)
            response = json.loads(result)
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
        mock_content = [
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
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=mock_content,
            headers={"content-type": "application/json"},
        )
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

        mock_content = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=mock_content,
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # All files should be included
        assert response["tree"] == mock_content

    @pytest.mark.asyncio
    async def test_list_tree_empty_response(
        self, metadata_with_project, gitlab_client_mock
    ):
        """Test ListRepositoryTree with empty API response."""
        tool = ListRepositoryTree(metadata=metadata_with_project)

        mock_response = GitLabHttpResponse(
            status_code=200,
            body=[],
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_response

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
        mock_content = [
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
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=mock_content,
            headers={"content-type": "application/json"},
        )
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

        mock_content = [
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
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=mock_content,
            headers={"content-type": "application/json"},
        )
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

        mock_content = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "debug.log", "type": "blob", "path": "debug.log"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        mock_response = GitLabHttpResponse(
            status_code=200,
            body=mock_content,
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_response

        result = await tool._arun(project_id="test/project")
        response = json.loads(result)

        # All files should be included when no project
        assert response["tree"] == mock_content


# ---------------------------------------------------------------------------
# GetRepositoryFiles tests
# ---------------------------------------------------------------------------


@pytest.fixture(name="bulk_tool")
def bulk_tool_fixture(metadata):
    """GetRepositoryFiles tool with no project (no exclusion policy)."""
    return GetRepositoryFiles(metadata=metadata)


@pytest.fixture(name="bulk_tool_with_project")
def bulk_tool_with_project_fixture(metadata_with_project):
    """GetRepositoryFiles tool with exclusion rules."""
    return GetRepositoryFiles(metadata=metadata_with_project)


def _make_file_response(content: str) -> GitLabHttpResponse:
    """Build a mock GitLab file API response for the given text content."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    return GitLabHttpResponse(
        status_code=200,
        body={"content": encoded},
        headers={"content-type": "application/json"},
    )


def _make_tree_response(paths: list) -> GitLabHttpResponse:
    """Build a mock GitLab tree API response for the given list of blob paths.

    Note: _paginate_get uses parse_json=False so body must be a JSON string.
    """
    body = [
        {"id": str(i), "name": p.split("/")[-1], "type": "blob", "path": p}
        for i, p in enumerate(paths)
    ]
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps(body),
        headers={"content-type": "application/json", "X-Next-Page": ""},
    )


class TestGetRepositoryFilesExplicitPaths:
    """Explicit-paths-only: tree endpoint must never be called."""

    @pytest.mark.asyncio
    async def test_explicit_paths_no_tree_call(self, bulk_tool, gitlab_client_mock):
        """Tree endpoint is never called when all paths are explicit."""
        gitlab_client_mock.aget.return_value = _make_file_response("hello")

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "src/main.py"],
        )

        response = json.loads(result)
        assert "README.md" in response
        assert "src/main.py" in response
        assert response["README.md"] == {"content": "hello"}

        # Verify only file-content calls were made (no tree call)
        for call in gitlab_client_mock.aget.call_args_list:
            assert "/repository/tree" not in call.kwargs.get("path", "")

    @pytest.mark.asyncio
    async def test_explicit_paths_missing_ref_raises(
        self, bulk_tool, gitlab_client_mock
    ):
        """Missing ref raises ToolException."""
        with pytest.raises(ToolException, match="'ref' is required"):
            await bulk_tool._arun(
                project_id="org/repo",
                file_paths=["README.md"],
            )

    @pytest.mark.asyncio
    async def test_explicit_paths_missing_project_id_raises(
        self, bulk_tool, gitlab_client_mock
    ):
        """Missing project_id raises ToolException."""
        with pytest.raises(ToolException, match="'project_id' must be provided"):
            await bulk_tool._arun(
                ref="main",
                file_paths=["README.md"],
            )


def _make_tree_response_from_entries(entries: list) -> GitLabHttpResponse:
    """Build a mock GitLab tree API response (JSON string body) from raw entries."""
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps(entries),
        headers={"content-type": "application/json", "X-Next-Page": ""},
    )


class TestGetRepositoryFilesGlobPatterns:
    """Pure glob pattern(s): tree listing invoked with recursive=true."""

    @pytest.mark.asyncio
    async def test_glob_triggers_tree_call(self, bulk_tool, gitlab_client_mock):
        """A glob pattern triggers a recursive tree listing."""
        tree_paths = ["src/main.py", "src/utils.py", "README.md"]

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(tree_paths)
            # File content call
            return _make_file_response("py content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["src/*.py"],
        )

        response = json.loads(result)
        # Only .py files in src/ should be matched
        assert "src/main.py" in response
        assert "src/utils.py" in response
        assert "README.md" not in response

        # Verify tree was called with recursive=true
        tree_calls = [
            c
            for c in gitlab_client_mock.aget.call_args_list
            if "/repository/tree" in c.kwargs.get("path", "")
        ]
        assert len(tree_calls) >= 1
        tree_params = tree_calls[0].kwargs.get("params", {})
        assert tree_params.get("recursive") == "true"

    @pytest.mark.asyncio
    async def test_glob_matches_nested_paths_via_single_star(
        self, bulk_tool, gitlab_client_mock
    ):
        """Fnmatch with * matches nested paths (e.g. src/*.py matches src/sub/file.py)."""
        tree_paths = ["src/a.py", "src/sub/b.py", "other/c.py"]

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(tree_paths)
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["src/*.py"],
        )

        response = json.loads(result)
        # fnmatch * matches / so src/*.py matches src/sub/b.py
        assert "src/a.py" in response
        assert "src/sub/b.py" in response
        assert "other/c.py" not in response

    @pytest.mark.asyncio
    async def test_glob_no_matches_returns_empty(self, bulk_tool, gitlab_client_mock):
        """Glob pattern with no matches returns empty result (no truncation marker)."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(["README.md"])
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["*.py"],
        )

        response = json.loads(result)
        assert response == {}

    @pytest.mark.asyncio
    async def test_tree_entries_with_type_tree_are_excluded(
        self, bulk_tool, gitlab_client_mock
    ):
        """Tree entries with type='tree' (directories) are not fetched."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response_from_entries(
                    [
                        {"id": "1", "name": "src", "type": "tree", "path": "src"},
                        {
                            "id": "2",
                            "name": "main.py",
                            "type": "blob",
                            "path": "src/main.py",
                        },
                    ]
                )
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["src*"],
        )

        response = json.loads(result)
        assert "src" not in response
        assert "src/main.py" in response


class TestGetRepositoryFilesMixedPaths:
    """Mixed explicit + glob entries: correct union/dedupe/order."""

    @pytest.mark.asyncio
    async def test_mixed_explicit_and_glob(self, bulk_tool, gitlab_client_mock):
        """Explicit paths and glob patterns are merged with explicit paths first."""
        tree_paths = ["src/main.py", "src/utils.py"]

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(tree_paths)
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "src/*.py"],
        )

        response = json.loads(result)
        assert "README.md" in response
        assert "src/main.py" in response
        assert "src/utils.py" in response

    @pytest.mark.asyncio
    async def test_deduplication_explicit_and_glob_overlap(
        self, bulk_tool, gitlab_client_mock
    ):
        """A path appearing in both explicit list and glob matches is returned once."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(["src/main.py"])
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["src/main.py", "src/*.py"],
        )

        response = json.loads(result)
        # src/main.py should appear exactly once
        assert list(response.keys()).count("src/main.py") == 1
        assert "src/main.py" in response

    @pytest.mark.asyncio
    async def test_explicit_paths_appear_before_glob_matches(
        self, bulk_tool, gitlab_client_mock
    ):
        """Explicit paths come before glob-matched paths in the result."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response(["a.py", "b.py"])
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "*.py"],
        )

        response = json.loads(result)
        keys = list(response.keys())
        assert keys.index("README.md") < keys.index("a.py")
        assert keys.index("README.md") < keys.index("b.py")


class TestGetRepositoryFilesPerPageTruncation:
    """per_page truncation: default 20, custom value, truncation marker."""

    @pytest.mark.asyncio
    async def test_default_per_page_is_20(self, bulk_tool, gitlab_client_mock):
        """Default per_page caps results at 20."""
        # 25 explicit paths
        paths = [f"file{i}.txt" for i in range(25)]
        gitlab_client_mock.aget.return_value = _make_file_response("content")

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=paths,
        )

        response = json.loads(result)
        # 20 file results + __truncated__ marker
        file_results = {k: v for k, v in response.items() if k != "__truncated__"}
        assert len(file_results) == 20
        assert "__truncated__" in response
        assert response["__truncated__"]["total_matched"] == 25
        assert response["__truncated__"]["returned"] == 20

    @pytest.mark.asyncio
    async def test_custom_per_page(self, bulk_tool, gitlab_client_mock):
        """Custom per_page value is respected."""
        paths = [f"file{i}.txt" for i in range(10)]
        gitlab_client_mock.aget.return_value = _make_file_response("content")

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=paths,
            per_page=5,
        )

        response = json.loads(result)
        file_results = {k: v for k, v in response.items() if k != "__truncated__"}
        assert len(file_results) == 5
        assert "__truncated__" in response

    @pytest.mark.asyncio
    async def test_no_truncation_marker_when_within_limit(
        self, bulk_tool, gitlab_client_mock
    ):
        """No __truncated__ key when results fit within per_page."""
        paths = ["README.md", "src/main.py"]
        gitlab_client_mock.aget.return_value = _make_file_response("content")

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=paths,
            per_page=10,
        )

        response = json.loads(result)
        assert "__truncated__" not in response
        assert len(response) == 2

    @pytest.mark.asyncio
    async def test_truncation_marker_contains_helpful_message(
        self, bulk_tool, gitlab_client_mock
    ):
        """Truncation marker includes total_matched, returned, and a message."""
        paths = [f"file{i}.txt" for i in range(30)]
        gitlab_client_mock.aget.return_value = _make_file_response("content")

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=paths,
            per_page=10,
        )

        response = json.loads(result)
        marker = response["__truncated__"]
        assert marker["total_matched"] == 30
        assert marker["returned"] == 10
        assert "message" in marker
        assert "30" in marker["message"]
        assert "10" in marker["message"]

    def test_per_page_max_50_validation(self):
        """per_page > 50 is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RepositoryFilesResourceInput(
                project_id="org/repo",
                ref="main",
                file_paths=["README.md"],
                per_page=51,
            )

    def test_per_page_min_1_validation(self):
        """per_page < 1 is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RepositoryFilesResourceInput(
                project_id="org/repo",
                ref="main",
                file_paths=["README.md"],
                per_page=0,
            )

    def test_per_page_50_is_valid(self):
        """per_page=50 is the maximum valid value."""
        schema = RepositoryFilesResourceInput(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md"],
            per_page=50,
        )
        assert schema.per_page == 50


class TestGetRepositoryFilesExclusionPolicy:
    """Exclusion policy applied to both explicit and glob-matched paths."""

    @pytest.mark.asyncio
    async def test_excluded_explicit_path_returns_error(
        self, bulk_tool_with_project, gitlab_client_mock
    ):
        """Excluded explicit paths appear in result with error, not content."""
        gitlab_client_mock.aget.return_value = _make_file_response("content")

        result = await bulk_tool_with_project._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "debug.log"],  # debug.log excluded by *.log
        )

        response = json.loads(result)
        assert "content" in response["README.md"]
        assert "error" in response["debug.log"]
        assert "excluded" in response["debug.log"]["error"].lower()

    @pytest.mark.asyncio
    async def test_excluded_glob_matched_path_returns_error(
        self, bulk_tool_with_project, gitlab_client_mock
    ):
        """Glob-matched paths that are excluded appear with error."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return _make_tree_response_from_entries(
                    [
                        {
                            "id": "1",
                            "name": "main.py",
                            "type": "blob",
                            "path": "src/main.py",
                        },
                        {
                            "id": "2",
                            "name": "debug.log",
                            "type": "blob",
                            "path": "debug.log",
                        },
                    ]
                )
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool_with_project._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["*"],
        )

        response = json.loads(result)
        assert "content" in response["src/main.py"]
        assert "error" in response["debug.log"]

    @pytest.mark.asyncio
    async def test_all_paths_excluded(self, bulk_tool_with_project, gitlab_client_mock):
        """When all paths are excluded, result contains only error entries."""
        result = await bulk_tool_with_project._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["debug.log", "app.log"],
        )

        response = json.loads(result)
        assert "error" in response["debug.log"]
        assert "error" in response["app.log"]
        # No content fetches should have been made
        gitlab_client_mock.aget.assert_not_called()


class TestGetRepositoryFilesConcurrency:
    """Concurrency cap: no more than 4 concurrent file-content fetches."""

    @pytest.mark.asyncio
    async def test_concurrency_cap_respected(self, bulk_tool, gitlab_client_mock):
        """At most DEFAULT_REPOSITORY_FILES_CONCURRENCY fetches run concurrently."""
        max_concurrent = 0
        current_concurrent = 0

        async def slow_fetch(**kwargs):
            nonlocal max_concurrent, current_concurrent
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return GitLabHttpResponse(
                    status_code=200,
                    body=json.dumps([]),
                    headers={"X-Next-Page": ""},
                )
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0)  # yield to allow other coroutines to start
            current_concurrent -= 1
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = slow_fetch

        paths = [f"file{i}.txt" for i in range(10)]
        await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=paths,
        )

        assert max_concurrent <= 4

    @pytest.mark.asyncio
    async def test_tree_listing_not_throttled_by_semaphore(
        self, bulk_tool, gitlab_client_mock
    ):
        """Tree listing calls are not subject to the content-fetch semaphore."""
        tree_call_count = 0

        async def side_effect(**kwargs):
            nonlocal tree_call_count
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                tree_call_count += 1
                return _make_tree_response(["a.py"])
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["*.py"],
        )

        # Tree was called (not blocked by semaphore)
        assert tree_call_count >= 1


class TestGetRepositoryFilesPartialFailure:
    """Partial failure: one file errors, others still return content."""

    @pytest.mark.asyncio
    async def test_404_file_returns_error_others_succeed(
        self, bulk_tool, gitlab_client_mock
    ):
        """A 404 on one file returns error for that file; others succeed."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "missing.py" in path:
                return GitLabHttpResponse(
                    status_code=404,
                    body={"message": "404 File Not Found"},
                    headers={"content-type": "application/json"},
                )
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "missing.py", "src/main.py"],
        )

        response = json.loads(result)
        assert "content" in response["README.md"]
        assert "error" in response["missing.py"]
        assert "content" in response["src/main.py"]

    @pytest.mark.asyncio
    async def test_no_exception_raised_on_partial_failure(
        self, bulk_tool, gitlab_client_mock
    ):
        """A single file failure does not raise ToolException for the whole call."""

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "bad.py" in path:
                raise Exception("Network error")
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        # Should not raise
        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "bad.py"],
        )

        response = json.loads(result)
        assert "content" in response["README.md"]
        assert "error" in response["bad.py"]

    @pytest.mark.asyncio
    async def test_decode_error_returns_error_entry(
        self, bulk_tool, gitlab_client_mock
    ):
        """A UnicodeDecodeError on one file returns error for that file."""
        binary_content = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("latin-1")

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "image.png" in path:
                return GitLabHttpResponse(
                    status_code=200,
                    body={"content": binary_content},
                    headers={"content-type": "application/json"},
                )
            return _make_file_response("text content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "image.png"],
        )

        response = json.loads(result)
        assert "content" in response["README.md"]
        assert "error" in response["image.png"]


class TestGetRepositoryFilesFormatDisplayMessage:
    """format_display_message assertions covering truncation/pattern-count messaging."""

    def test_explicit_paths_only_message(self, bulk_tool):
        """Display message for explicit paths only."""
        args = RepositoryFilesResourceInput(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "src/main.py"],
        )
        msg = bulk_tool.format_display_message(args)
        assert "2 explicit path(s)" in msg
        assert "org/repo" in msg
        assert "main" in msg

    def test_glob_patterns_only_message(self, bulk_tool):
        """Display message for glob patterns only."""
        args = RepositoryFilesResourceInput(
            project_id="org/repo",
            ref="main",
            file_paths=["*.py", "src/*.md"],
        )
        msg = bulk_tool.format_display_message(args)
        assert "2 glob pattern(s)" in msg
        assert "`*.py`" in msg
        assert "`src/*.md`" in msg

    def test_mixed_paths_message(self, bulk_tool):
        """Display message for mixed explicit + glob."""
        args = RepositoryFilesResourceInput(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md", "*.py"],
        )
        msg = bulk_tool.format_display_message(args)
        assert "1 explicit path(s)" in msg
        assert "1 glob pattern(s)" in msg

    def test_per_page_shown_in_message(self, bulk_tool):
        """Display message includes per_page limit."""
        args = RepositoryFilesResourceInput(
            project_id="org/repo",
            ref="main",
            file_paths=["README.md"],
            per_page=5,
        )
        msg = bulk_tool.format_display_message(args)
        assert "5" in msg

    def test_url_shown_in_message(self, bulk_tool):
        """Display message uses URL when provided."""
        args = RepositoryFilesResourceInput(
            url="https://gitlab.com/org/repo/-/blob/main/README.md",
            file_paths=["README.md"],
        )
        msg = bulk_tool.format_display_message(args)
        assert "https://gitlab.com/org/repo" in msg


class TestGetRepositoryFilesInputValidation:
    """Input schema validation tests."""

    @pytest.mark.asyncio
    async def test_empty_file_paths_returns_empty_result(
        self, bulk_tool, gitlab_client_mock
    ):
        """Empty file_paths returns empty JSON object."""
        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=[],
        )
        response = json.loads(result)
        assert response == {}


class TestGetRepositoryFilesEdgeCases:
    """Edge-case coverage for GetRepositoryFiles branches."""

    @pytest.mark.asyncio
    async def test_invalid_url_raises_tool_exception(
        self, bulk_tool, gitlab_client_mock
    ):
        """An unparsable URL causes ToolException with the URL error message."""
        with pytest.raises(ToolException, match="Failed to parse URL"):
            await bulk_tool._arun(
                url="https://not-gitlab.com/invalid",
                file_paths=["README.md"],
            )

    @pytest.mark.asyncio
    async def test_resolve_glob_skips_non_dict_tree_entries(
        self, bulk_tool, gitlab_client_mock
    ):
        """Non-dict entries in the tree response are silently skipped."""
        # Mix a string (non-dict) entry with a valid blob entry
        entries = [
            "this-is-not-a-dict",
            {"id": "1", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return GitLabHttpResponse(
                    status_code=200,
                    body=json.dumps(entries),
                    headers={"content-type": "application/json", "X-Next-Page": ""},
                )
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["*.py"],
        )

        response = json.loads(result)
        # Only the valid blob entry should be matched; the string entry is skipped
        assert "src/main.py" in response
        assert len([k for k in response if k != "__truncated__"]) == 1

    @pytest.mark.asyncio
    async def test_resolve_glob_skips_entries_with_empty_path(
        self, bulk_tool, gitlab_client_mock
    ):
        """Tree entries with an empty or missing path are silently skipped."""
        entries = [
            {"id": "1", "name": "no-path", "type": "blob", "path": ""},
            {"id": "2", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]

        async def side_effect(**kwargs):
            path = kwargs.get("path", "")
            if "/repository/tree" in path:
                return GitLabHttpResponse(
                    status_code=200,
                    body=json.dumps(entries),
                    headers={"content-type": "application/json", "X-Next-Page": ""},
                )
            return _make_file_response("content")

        gitlab_client_mock.aget.side_effect = side_effect

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["*.py"],
        )

        response = json.loads(result)
        # The entry with empty path is skipped; only the valid one is returned
        assert "src/main.py" in response
        assert "" not in response

    @pytest.mark.asyncio
    async def test_fetch_file_content_truncates_large_file(
        self, bulk_tool, gitlab_client_mock
    ):
        """Files with >= 2000 newlines are truncated via GetRepositoryFile._paginate."""
        # Build a file with exactly DEFAULT_GET_REPOSITORY_FILE_LIMIT lines (2000 newlines)
        large_content = "\n".join(f"line {i}" for i in range(2001))
        encoded = base64.b64encode(large_content.encode("utf-8")).decode("utf-8")
        mock_response = GitLabHttpResponse(
            status_code=200,
            body={"content": encoded},
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = mock_response

        result = await bulk_tool._arun(
            project_id="org/repo",
            ref="main",
            file_paths=["big_file.txt"],
        )

        response = json.loads(result)
        content = response["big_file.txt"]["content"]
        # The content should be truncated with a pagination hint
        assert "Use offset=" in content
        assert "Showing lines" in content
