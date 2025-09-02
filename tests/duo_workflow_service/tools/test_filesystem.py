import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.tools.base import ToolException

from contract import contract_pb2
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.filesystem import (  # Mkdir,
    DEFAULT_CONTEXT_EXCLUSIONS,
    EditFile,
    EditFileInput,
    FindFiles,
    FindFilesInput,
    ListDir,
    ListDirInput,
    Mkdir,
    MkdirInput,
    ReadFile,
    ReadFileInput,
    ReadFiles,
    ReadFilesInput,
    WriteFile,
    WriteFileInput,
    validate_duo_context_exclusions,
)
from tests.duo_workflow_service.tools.constants import (
    NORMAL_FILES,
    SENSITIVE_DIRECTORIES,
    SENSITIVE_FILES,
    SUSPICIOUS_PATHS,
)


@pytest.fixture(autouse=True)
def mock_feature_flag():
    """Mock feature flag to return True for USE_DUO_CONTEXT_EXCLUSION."""
    with patch(
        "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled"
    ) as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_project():
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=None,
    )


@pytest.fixture
def metadata_with_project(mock_project):
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="test contents")
        )
    )

    return {"outbox": mock_outbox, "inbox": mock_inbox, "project": mock_project}


@pytest.mark.asyncio
async def test_read_file(metadata_with_project):
    tool = ReadFile(description="Read file content")
    tool.metadata = metadata_with_project
    path = "./somepath"

    response = await tool._arun(path)

    assert response == "test contents"

    metadata_with_project["outbox"].put.assert_called_once()
    action = metadata_with_project["outbox"].put.call_args[0][0]
    assert action.runReadFile.filepath == path


@pytest.mark.asyncio
async def test_read_file_not_implemented_error():
    tool = ReadFile(description="Read file content")

    with pytest.raises(NotImplementedError):
        tool._run("./main.py")


@pytest.mark.asyncio
async def test_write_file(metadata_with_project, mock_project):
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="done")
        )
    )

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox, "project": mock_project}

    tool = WriteFile(description="Write file content")
    tool.metadata = metadata
    path = "./somepath"
    contents = "test contents"

    response = await tool._arun(path, contents)

    assert response == "done"

    mock_outbox.put.assert_called_once()
    action = mock_outbox.put.call_args[0][0]
    assert action.runWriteFile.filepath == path
    assert action.runWriteFile.contents == contents


@pytest.mark.asyncio
async def test_write_file_not_implemented_error():
    tool = WriteFile(description="Write file content")

    with pytest.raises(NotImplementedError):
        tool._run("./main.py", "sum(1, 2)")


class TestFindFiles:
    @pytest.mark.asyncio
    async def test_find_files_arun_method(self, mock_project):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response="file1.py\nfile2.py"
                )
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox, "project": mock_project}
        tool = FindFiles()
        tool.metadata = metadata
        name_pattern = "*.py"
        result = await tool._arun(name_pattern=name_pattern)

        assert result == "file1.py\nfile2.py"

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.FindFiles", autospec=True)
    async def test_find_files_empty_result(self, mock_find_files_class):
        # Create a mock instance with a mocked _arun method
        mock_instance = mock_find_files_class.return_value
        mock_instance._arun = AsyncMock(
            return_value="No matches found for pattern '*.nonexistent'"
        )

        # Now use the mock instance instead of creating a real one
        name_pattern = "*.nonexistent"
        result = await mock_instance._arun(name_pattern)

        assert "No matches found for pattern '*.nonexistent'" in result
        mock_instance._arun.assert_called_once_with("*.nonexistent")

    def test_find_files_sync_run_method(self):
        tool = FindFiles()
        with pytest.raises(
            NotImplementedError, match="This tool can only be run asynchronously"
        ):
            tool._run(".", "*.py")


class TestLsDir:
    @pytest.mark.asyncio
    async def test_list_dir_success(self, mock_project):
        # Set up the mock outbox and inbox
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response="file1.txt file2.txt dir1 dir2"
                )
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox, "project": mock_project}

        # Create the tool and set its metadata
        list_dir_tool = ListDir()
        list_dir_tool.metadata = metadata

        # Call the method being tested
        result = await list_dir_tool._arun(directory=".")

        # Assert the result
        assert result == "file1.txt file2.txt dir1 dir2"

        # Verify the outbox was used as expected
        mock_outbox.put.assert_called_once()

        # You can add additional assertions to verify the details of what was put on the outbox
        action = mock_outbox.put.call_args[0][0]
        assert action.listDirectory.directory == "."

    @pytest.mark.asyncio
    async def test_list_dir_not_implemented_error(self):
        list_dir_tool = ListDir()

        with pytest.raises(NotImplementedError):
            list_dir_tool._run("test_dir")

    def test_list_dir_format_display_message(self):
        list_dir_tool = ListDir()

        input_data = ListDirInput(directory="./src")
        message = list_dir_tool.format_display_message(input_data)

        expected_message = "Using list_dir: directory=./src"
        assert message == expected_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
    )
    async def test_list_dir_rejects_excluded_paths(self, path):
        with pytest.raises(ToolException, match="Access denied"):
            await ListDir(description="List files")._arun(path)


class TestMkdir:
    @pytest.mark.asyncio
    async def test_mkdir_creates_directory(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response="")
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        mkdir_tool = Mkdir()
        mkdir_tool.metadata = metadata
        result = await mkdir_tool._arun("./test_dir")

        assert result == ""

        # Verify the action sent to outbox
        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.mkdir.directory_path == "./test_dir"

    @pytest.mark.asyncio
    async def test_mkdir_creates_nested_directories(self):
        # Set up mock outbox and inbox following the pattern from other tests
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response="")
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        # Create the tool and set its metadata
        mkdir_tool = Mkdir()
        mkdir_tool.metadata = metadata

        # Call the method being tested
        result = await mkdir_tool._arun("./test_dir/nested/dir")

        # Assert the result
        assert result == ""

        # Verify the outbox was called correctly
        mock_outbox.put.assert_called_once()

        # Verify the action details
        action = mock_outbox.put.call_args[0][0]
        assert action.mkdir.directory_path == "./test_dir/nested/dir"

    @pytest.mark.asyncio
    async def test_mkdir_validates_path(self):
        mkdir_tool = Mkdir()
        result = await mkdir_tool._arun("../test_dir")

        assert (
            result == "Creating directories above the current directory is not allowed"
        )


class TestEditFile:
    @pytest.mark.asyncio
    async def test_basic(self, mock_project):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response="success")
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox, "project": mock_project}

        tool = EditFile(metadata=metadata)
        path = "./somefile.txt"
        old_str = "old line"
        new_str = "new line"

        response = await tool._arun(path, old_str, new_str)

        assert response == "success"

        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.runEditFile.filepath == path
        assert action.runEditFile.oldString == old_str
        assert action.runEditFile.newString == new_str

    @pytest.mark.asyncio
    async def test_not_implemented_error(self):
        tool = EditFile()

        with pytest.raises(NotImplementedError):
            tool._run("./main.py", "old", "new")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
    )
    async def test_edit_file_rejects_excluded_paths(self, path):
        with pytest.raises(ToolException, match="Access denied"):
            await EditFile(description="Edit file content")._arun(path, "old", "new")


class TestReadFile:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
    )
    async def test_read_file_rejects_excluded_paths(self, path):
        with pytest.raises(ToolException, match="Access denied"):
            await ReadFile(description="Read file content")._arun(path)


class TestReadFiles:
    @pytest.mark.asyncio
    async def test_read_files_with_mixed_valid_invalid_paths(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        # Mock response with mixed success and error
        mock_response = '{"file1.py": {"content": "print(\'hello\')"}, "nonexistent.py": {"error": "File not found"}}'
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response=mock_response)
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        response = await tool._arun(file_paths)

        assert response == mock_response

        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.runReadFiles.filepaths == file_paths

    @pytest.mark.asyncio
    async def test_read_files_with_no_action_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        # Mock response with mixed success and error
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(actionResponse=None)
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        response = await tool._arun(file_paths)

        assert response == "Could not read files"

        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.runReadFiles.filepaths == file_paths

    @pytest.mark.asyncio
    async def test_read_files_with_none_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        # Mock response with mixed success and error
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response=None)
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        response = await tool._arun(file_paths)

        assert response == "Could not read files"

        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.runReadFiles.filepaths == file_paths

    @pytest.mark.asyncio
    async def test_read_files_with_error_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        # Mock response with mixed success and error
        mock_plaintextResponse = {"error": "Error reading files"}
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response=None, plainTextResponse=mock_plaintextResponse
                )
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        response = await tool._arun(file_paths)

        assert response == "Could not read files"

        mock_outbox.put.assert_called_once()
        action = mock_outbox.put.call_args[0][0]
        assert action.runReadFiles.filepaths == file_paths

    @pytest.mark.asyncio
    async def test_read_files_rejects_excluded_paths(self):
        tool = ReadFiles(description="Read multiple files")

        # Test with one excluded path
        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun([".ssh/config", "valid_file.py"])

        # Test with multiple excluded paths
        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun([".git/config", ".env"])

    @pytest.mark.asyncio
    async def test_read_files_not_implemented_error(self):
        tool = ReadFiles(description="Read multiple files")

        with pytest.raises(NotImplementedError):
            tool._run(["file1.py", "file2.py"])

    def test_read_files_format_display_message_single_file(self):
        tool = ReadFiles(description="Read multiple files")
        input_data = ReadFilesInput(file_paths=["single.py"])

        message = tool.format_display_message(input_data)
        assert message == "Read 1 file"

    def test_read_files_format_display_message_multiple_files(self):
        tool = ReadFiles(description="Read multiple files")
        input_data = ReadFilesInput(file_paths=["file1.py", "file2.py", "file3.py"])

        message = tool.format_display_message(input_data)
        assert message == "Read 3 files"

    def test_read_files_format_display_message_with_exclusions(self):
        """Test ReadFiles format_display_message includes exclusion information."""
        tool = ReadFiles(description="Read multiple files")

        # Mock tool response with excluded files
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "file1.py": {"content": "content"},
                "file2.secret": {"error": "excluded due to policy"},
                "config/private/secret.txt": {"error": "excluded due to policy"},
            }
        )

        input_data = ReadFilesInput(
            file_paths=["file1.py", "file2.secret", "config/private/secret.txt"]
        )

        message = tool.format_display_message(input_data, mock_response)

        expected_excluded_msg = FileExclusionPolicy.format_user_exclusion_message(
            ["file2.secret", "config/private/secret.txt"]
        )
        assert message == f"Read 1 file{expected_excluded_msg}"

    @pytest.mark.asyncio
    async def test_read_files_with_file_exclusion_policy(self, mock_project):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_response = '{"allowed_file.py": {"content": "hi"}}'
        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response=mock_response)
            )
        )

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = {
            "outbox": mock_outbox,
            "inbox": mock_inbox,
            "project": mock_project,
        }

        file_paths = ["allowed_file.py", ".env", ".ssh/config"]

        with patch.object(FileExclusionPolicy, "filter_allowed") as mock_filter_allowed:
            mock_filter_allowed.return_value = (
                ["allowed_file.py"],
                [".env", ".ssh/config"],
            )

            response = await tool._arun(file_paths)

            result_dict = json.loads(response)

            assert "allowed_file.py" in result_dict
            assert ".env" in result_dict
            assert ".ssh/config" in result_dict
            assert result_dict["allowed_file.py"]["content"] == "hi"
            assert result_dict[".env"]["error"] == "excluded due to policy"
            assert result_dict[".ssh/config"]["error"] == "excluded due to policy"

            # Should only call the action with allowed files
            action = mock_outbox.put.call_args[0][0]
            assert set(action.runReadFiles.filepaths) == {"allowed_file.py"}


class TestWriteFile:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
    )
    async def test_write_file_rejects_excluded_paths(self, path):
        with pytest.raises(ToolException, match="Access denied"):
            await WriteFile(description="Write file content")._arun(
                path, "file contents"
            )


def test_read_file_format_display_message(mock_project):
    tool = ReadFile(description="Read file description")
    tool.metadata = {"project": mock_project}

    input_data = ReadFileInput(file_path="./src/main.py")

    message = tool.format_display_message(input_data)

    expected_message = "Read file"
    assert message == expected_message


def test_write_file_format_display_message(mock_project):
    tool = WriteFile(description="Write file description")
    tool.metadata = {"project": mock_project}

    input_data = WriteFileInput(
        file_path="./src/new_file.py", contents="print('Hello, world!')"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Create file"
    assert message == expected_message


def test_find_files_format_display_message():
    tool = FindFiles(description="Find files description")

    # Test with default parameters
    input_data = FindFilesInput(name_pattern="*.py")

    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern '*.py'"
    assert message == expected_message

    # Test with tracked_only
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern '*.py'"
    assert message == expected_message

    # Test with untracked_only
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern '*.py'"
    assert message == expected_message

    # Test with modified
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern '*.py'"
    assert message == expected_message

    # Test with deleted
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern '*.py'"
    assert message == expected_message


def test_mkdir_format_display_message():
    tool = Mkdir(description="Mkdir description")

    input_data = MkdirInput(directory_path="./src/new_directory")

    message = tool.format_display_message(input_data)

    expected_message = "Create directory './src/new_directory'"
    assert message == expected_message


def test_edit_file_format_display_message(mock_project):
    tool = EditFile(description="Edit file description")
    tool.metadata = {"project": mock_project}

    input_data = EditFileInput(
        file_path="./src/main.py",
        old_str="print('Hello')",
        new_str="print('Hello, world!')",
    )

    message = tool.format_display_message(input_data)

    expected_message = "Edit file"
    assert message == expected_message


@pytest.mark.parametrize("path", NORMAL_FILES)
def test_validate_duo_context_exclusions_allows_normal_files(path):
    # These should not raise exceptions
    validate_duo_context_exclusions(path)


@pytest.mark.parametrize(
    "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
)
def test_validate_duo_context_exclusions_rejects_sensitive_files(path):
    with pytest.raises(ToolException, match="Access denied"):
        validate_duo_context_exclusions(path)


def test_default_context_exclusions_does_not_exclude_bang_patterns():
    match = DEFAULT_CONTEXT_EXCLUSIONS.match(".env.example")
    assert match is not None
    assert not bool(match)


@pytest.mark.parametrize(
    "path",
    [
        ".config/nvim",
        ".config/nvim",
        ".docker",
        ".emacs.d",
        ".git",
        ".git",
        ".gnupg",
        ".idea",
        ".metadata",
        ".settings",
        ".ssh",
        ".ssh",
        ".ssh",
        ".vim",
        ".vscode",
    ],
)
def test_default_context_exclusions_excludes_directories(path):
    dir_match = DEFAULT_CONTEXT_EXCLUSIONS.match(path)
    assert dir_match is not None
    assert bool(dir_match)


@pytest.mark.parametrize(
    "filepath",
    [
        ".config/nvim/init.lua",
        ".config/nvim/init.vim",
        ".docker/config.json",
        ".emacs.d/init.el",
        ".git/config",
        ".git/info/exclude",
        ".gnupg/gpg.conf",
        ".idea/gitlab.xml",
        ".metadata/.plugins/org.eclipse.jdt.core",
        ".settings/org.eclipse.jdt.core.prefs",
        ".ssh/authorized_keys",
        ".ssh/config",
        ".ssh/id_rsa",
        ".vim/pack/vendor/start/vim-lsp/autoload/lsp.vim",
        ".vscode/settings.json",
    ],
)
def test_default_context_exclusions_excludes_files_under_directories(filepath):
    path_match = DEFAULT_CONTEXT_EXCLUSIONS.match(filepath)
    assert path_match is not None
    assert bool(path_match)


@pytest.mark.parametrize(
    "path",
    [
        ".env.production",
        ".env.staging",
        ".env",
        ".vimrc",
        "Dockerfile.secrets",
    ],
)
def test_default_context_exclusions_excludes_patterns(path):
    match = DEFAULT_CONTEXT_EXCLUSIONS.match(path)
    assert match is not None
    assert bool(match)


class TestFileExclusionPolicy:
    """Test suite for FileExclusionPolicy features."""

    @pytest.fixture
    def project_with_exclusions(self):
        """Project with custom exclusion rules."""
        return Project(
            id=1,
            name="test-project",
            description="Test project with exclusions",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=[
                "*.secret",
                "config/private/*",
                "!config/private/allowed.txt",
                "temp/",
            ],
        )

    @pytest.fixture
    def project_without_exclusions(self):
        """Project without exclusion rules."""
        return Project(
            id=2,
            name="test-project-no-exclusions",
            description="Test project without exclusions",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=None,
        )

    @pytest.fixture
    def project_with_empty_exclusions(self):
        """Project with empty exclusion rules list."""
        return Project(
            id=3,
            name="test-project-empty-exclusions",
            description="Test project with empty exclusions",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=[],
        )

    @pytest.mark.asyncio
    async def test_read_file_with_exclusion_policy(self, project_with_exclusions):
        """Test ReadFile tool respects FileExclusionPolicy."""
        tool = ReadFile(description="Read file content")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        result = await tool._arun("file.secret")
        expected = FileExclusionPolicy.format_llm_exclusion_message(["file.secret"])
        assert result == expected

    @pytest.mark.asyncio
    async def test_write_file_with_exclusion_policy(self, project_with_exclusions):
        """Test WriteFile tool respects FileExclusionPolicy."""
        tool = WriteFile(description="Write file content")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        result = await tool._arun("file.secret", "content")
        expected = FileExclusionPolicy.format_llm_exclusion_message(["file.secret"])
        assert result == expected

    @pytest.mark.asyncio
    async def test_edit_file_with_exclusion_policy(self, project_with_exclusions):
        """Test EditFile tool respects FileExclusionPolicy."""
        tool = EditFile(description="Edit file content")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        result = await tool._arun("file.secret", "old", "new")
        expected = FileExclusionPolicy.format_llm_exclusion_message(["file.secret"])
        assert result == expected

    @pytest.mark.asyncio
    async def test_list_dir_with_exclusion_policy(self, project_with_exclusions):
        """Test ListDir tool respects FileExclusionPolicy."""
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response="file1.txt\nfile2.secret\nconfig/private/secret.txt\nconfig/private/allowed.txt\ntemp/cache.txt"
                )
            )
        )

        metadata = {
            "outbox": mock_outbox,
            "inbox": mock_inbox,
            "project": project_with_exclusions,
        }

        tool = ListDir()
        tool.metadata = metadata

        result = await tool._arun(".")

        # Should only include allowed files
        expected_files = ["file1.txt", "config/private/allowed.txt"]
        assert result == "\n".join(expected_files)

    @pytest.mark.asyncio
    async def test_find_files_with_exclusion_policy(self, project_with_exclusions):
        """Test FindFiles tool respects FileExclusionPolicy."""
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response="file1.txt\nfile2.secret\nconfig/private/secret.txt\nconfig/private/allowed.txt"
                )
            )
        )

        metadata = {
            "outbox": mock_outbox,
            "inbox": mock_inbox,
            "project": project_with_exclusions,
        }

        tool = FindFiles()
        tool.metadata = metadata

        result = await tool._arun("*")

        # Should only include allowed files
        expected_files = ["file1.txt", "config/private/allowed.txt"]
        assert result == "\n".join(expected_files)

    def test_read_file_format_display_message_with_exclusion(
        self, project_with_exclusions
    ):
        """Test ReadFile format_display_message includes exclusion message."""
        tool = ReadFile(description="Read file description")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        input_data = ReadFileInput(file_path="file.secret")
        message = tool.format_display_message(input_data)
        expected = "Read file" + FileExclusionPolicy.format_user_exclusion_message(
            ["file.secret"]
        )
        assert message == expected

        # Test allowed file
        input_data = ReadFileInput(file_path="file.txt")
        message = tool.format_display_message(input_data)
        assert message == "Read file"

    def test_write_file_format_display_message_with_exclusion(
        self, project_with_exclusions
    ):
        """Test WriteFile format_display_message includes exclusion message."""
        tool = WriteFile(description="Write file description")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        input_data = WriteFileInput(file_path="file.secret", contents="content")
        message = tool.format_display_message(input_data)
        expected = "Create file" + FileExclusionPolicy.format_user_exclusion_message(
            ["file.secret"]
        )
        assert message == expected

        # Test allowed file
        input_data = WriteFileInput(file_path="file.txt", contents="content")
        message = tool.format_display_message(input_data)
        assert message == "Create file"

    def test_edit_file_format_display_message_with_exclusion(
        self, project_with_exclusions
    ):
        """Test EditFile format_display_message includes exclusion message."""
        tool = EditFile(description="Edit file description")
        tool.metadata = {"project": project_with_exclusions}

        # Test excluded file
        input_data = EditFileInput(
            file_path="file.secret", old_str="old", new_str="new"
        )
        message = tool.format_display_message(input_data)
        expected = "Edit file" + FileExclusionPolicy.format_user_exclusion_message(
            ["file.secret"]
        )
        assert message == expected

        # Test allowed file
        input_data = EditFileInput(file_path="file.txt", old_str="old", new_str="new")
        message = tool.format_display_message(input_data)
        assert message == "Edit file"

    @pytest.mark.asyncio
    async def test_list_dir_excluded_directory(self, project_with_exclusions):
        """Test ListDir tool with excluded directory."""
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    response="temp/cache.txt\ntemp/log.txt"
                )
            )
        )

        metadata = {
            "outbox": mock_outbox,
            "inbox": mock_inbox,
            "project": project_with_exclusions,
        }

        tool = ListDir()
        tool.metadata = metadata

        # Test directory containing only excluded files
        result = await tool._arun("temp/")
        # Should return empty string since all files in temp/ are excluded
        assert result == ""
