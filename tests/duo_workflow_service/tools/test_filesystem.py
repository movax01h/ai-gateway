import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.tools import ReadFile, WriteFile
from duo_workflow_service.tools.filesystem import (
    EditFile,
    EditFileInput,
    FilesScopeEnum,
    FindFiles,
    FindFilesInput,
    Grep,
    GrepInput,
    LsFiles,
    LsFilesInput,
    Mkdir,
    MkdirInput,
    ReadFile,
    ReadFileInput,
    WriteFile,
    WriteFileInput,
)


@pytest.mark.asyncio
async def test_read_file():
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="test contents")
        )
    )

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

    tool = ReadFile(description="Read file content")
    tool.metadata = metadata
    path = "./somepath"

    response = await tool._arun(path)

    assert response == "test contents"

    mock_outbox.put.assert_called_once()
    action = mock_outbox.put.call_args[0][0]
    assert action.runReadFile.filepath == path


@pytest.mark.asyncio
async def test_read_file_not_implemented_error():
    tool = ReadFile(description="Read file content")

    with pytest.raises(NotImplementedError):
        tool._run("./main.py")


@pytest.mark.asyncio
async def test_write_file():
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="done")
        )
    )

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

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
    @patch("duo_workflow_service.tools.filesystem.GitCommand", autospec=True)
    async def test_find_files_arun_method(self, mock_git_command):
        mock_git_arun = AsyncMock(return_value="file1.py\nfile2.py")
        mock_git_command.return_value._arun = mock_git_arun

        tool = FindFiles()
        name_pattern = "*.py"
        result = await tool._arun(".", name_pattern)

        assert result == "file1.py\nfile2.py"
        mock_git_arun.assert_called_once_with(
            repository_url="",
            args="--exclude-standard --cached --others *.py",
            command="ls-files",
        )

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.GitCommand", autospec=True)
    async def test_find_files_empty_result(self, mock_git_command):
        mock_git_arun = AsyncMock(return_value="")
        mock_git_command.return_value._arun = mock_git_arun

        tool = FindFiles()
        name_pattern = "*.nonexistent"
        result = await tool._arun(".", name_pattern)

        assert "No matches found for pattern '*.nonexistent'" in result
        mock_git_arun.assert_called_once_with(
            repository_url="",
            args="--exclude-standard --cached --others *.nonexistent",
            command="ls-files",
        )

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.GitCommand", autospec=True)
    async def test_find_files_whitespace_result(self, mock_git_command):
        mock_git_arun = AsyncMock(return_value="  \n \t ")
        mock_git_command.return_value._arun = mock_git_arun

        tool = FindFiles()
        name_pattern = "*.nonexistent"
        result = await tool._arun(".", name_pattern)

        assert "No matches found for pattern '*.nonexistent'" in result
        mock_git_arun.assert_called_once_with(
            repository_url="",
            args="--exclude-standard --cached --others *.nonexistent",
            command="ls-files",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("files_scope", "expected_args"),
        [
            (FilesScopeEnum.ALL, "--exclude-standard --cached --others"),
            (FilesScopeEnum.TRACKED, "--exclude-standard --cached"),
            (FilesScopeEnum.UNTRACKED, "--exclude-standard --others"),
            (FilesScopeEnum.MODIFIED, "--exclude-standard --modified"),
            (FilesScopeEnum.DELETED, "--exclude-standard --deleted"),
        ],
    )
    @patch("duo_workflow_service.tools.filesystem.GitCommand")
    async def test_find_files_arun_method_with_params(
        self,
        mock_git_command,
        files_scope: FilesScopeEnum,
        expected_args: str,
    ):
        mock_git_arun = AsyncMock(return_value="file1.py\nfile2.py")
        mock_git_command.return_value._arun = mock_git_arun

        tool = FindFiles()
        name_pattern = "*.py"
        result = await tool._arun(".", name_pattern, files_scope=files_scope)

        assert result == "file1.py\nfile2.py"
        mock_git_arun.assert_called_once_with(
            repository_url="",
            args=f"{expected_args} *.py",
            command="ls-files",
        )

    def test_find_files_sync_run_method(self):
        tool = FindFiles()
        with pytest.raises(
            NotImplementedError, match="This tool can only be run asynchronously"
        ):
            tool._run(".", "*.py")


class TestLsFiles:
    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.GitCommand", autospec=True)
    async def test_run_git_command_success(self, mock_git_command):
        mock_arun = AsyncMock(return_value=".file1 file2 directory")
        mock_git_command.return_value._arun = mock_arun

        ls_command = LsFiles()

        assert (
            ls_command.description
            == """Lists the contents of a given directory by running the git ls-tree --name-only HEAD:dir command.
          The command lists only git tracked files (cached in Git’s index)."""
        )

        response = await ls_command._arun(directory="app")

        assert response == ".file1 file2 directory"
        mock_arun.assert_called_once_with(
            repository_url="", command="ls-tree", args="--name-only HEAD:app/"
        )

    @pytest.mark.asyncio
    async def test_run_command_not_implemented_error(self):
        ls_files = LsFiles()

        with pytest.raises(NotImplementedError):
            ls_files._run("")


class TestGrep:
    valid_test_cases = [
        # Basic recursive search
        pytest.param(
            {"pattern": "test", "search_directory": None, "recursive": True},
            "test.py:10:test line",
            "-r test",
            id="basic_recursive_grep",
        ),
        # Test with directory
        pytest.param(
            {
                "pattern": "test",
                "search_directory": "src",
            },
            "src/test.py:10:test line",
            "test -- src",
            id="with_directory",
        ),
        # Test with files_without_match
        pytest.param(
            {
                "pattern": "test",
                "search_directory": None,
                "files_without_match": True,
            },
            "file3.py",
            "--files-without-match test",
            id="files_without_match",
        ),
        # Test with files_with_matches
        pytest.param(
            {"pattern": "test", "files_with_matches": True},
            "file1.py\nfile2.py",
            "--files-with-matches test",
            id="files_with_matches",
        ),
        # Test with case_insensitive
        pytest.param(
            {"pattern": "test", "case_insensitive": True},
            "test.py:10:TEST line",
            "-i test",
            id="ignore_case",
        ),
        # Test with fixed_strings
        pytest.param(
            {"pattern": "<!-- tags:", "fixed_strings": True},
            "file1.py\nfile2.py",
            "-F <!-- tags:",
            id="fixed_strings",
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("params,expected_output,expected_args", valid_test_cases)
    @patch("duo_workflow_service.tools.filesystem.GitCommand", autospec=True)
    async def test_grep_arun(
        self, mock_run_git_command, params, expected_output, expected_args
    ):
        mock_arun = AsyncMock(return_value=expected_output)
        mock_run_git_command.return_value._arun = mock_arun

        grep_tool = Grep()
        result = await grep_tool._arun(**params)

        assert result == expected_output
        mock_arun.assert_called_once_with(
            args=expected_args,
            command="grep",
            repository_url="",
        )

    @pytest.mark.asyncio
    async def test_grep_security_check(self):
        grep_tool = Grep()
        result = await grep_tool._arun(
            pattern="test",
            search_directory="../parent",
        )

        assert result == "Searching above the current directory is not allowed"


class TestMkdir:
    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.RunCommand", autospec=True)
    async def test_mkdir_creates_directory(self, mock_run_command):
        mock_arun = AsyncMock(return_value="")
        mock_run_command.return_value._arun = mock_arun

        mkdir_tool = Mkdir()
        result = await mkdir_tool._arun("test_dir")

        assert result == ""
        mock_arun.assert_called_once_with(
            "mkdir", arguments=["./test_dir"], flags=["-p"]
        )

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.filesystem.RunCommand", autospec=True)
    async def test_mkdir_creates_nested_directories(self, mock_run_command):
        mock_arun = AsyncMock(return_value="")
        mock_run_command.return_value._arun = mock_arun

        mkdir_tool = Mkdir()
        result = await mkdir_tool._arun("./test_dir/nested/dir")

        assert result == ""
        mock_arun.assert_called_once_with(
            "mkdir", arguments=["./test_dir/nested/dir"], flags=["-p"]
        )

    @pytest.mark.asyncio
    async def test_mkdir_validates_path(self):
        mkdir_tool = Mkdir()
        result = await mkdir_tool._arun("../test_dir")

        assert (
            result == "Creating directories above the current directory is not allowed"
        )


class TestEditFile:
    @pytest.mark.asyncio
    async def test_basic(self):
        mock_outbox = MagicMock()
        mock_outbox.put = AsyncMock()

        mock_inbox = MagicMock()
        mock_inbox.get = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response="success")
            )
        )

        metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

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


def test_read_file_format_display_message():
    tool = ReadFile(description="Read file description")

    input_data = ReadFileInput(file_path="./src/main.py")

    message = tool.format_display_message(input_data)

    expected_message = "Read file"
    assert message == expected_message


def test_write_file_format_display_message():
    tool = WriteFile(description="Write file description")

    input_data = WriteFileInput(
        file_path="./src/new_file.py", contents="print('Hello, world!')"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Create file"
    assert message == expected_message


def test_find_files_format_display_message():
    tool = FindFiles(description="Find files description")

    # Test with default parameters
    input_data = FindFilesInput(directory=".", name_pattern="*.py")
    message = tool.format_display_message(input_data)
    expected_message = "Search files in '.' with pattern '*.py' (All files)"
    assert message == expected_message

    # Test with tracked_only
    input_data = FindFilesInput(
        directory=".",
        name_pattern="*.py",
        files_scope=FilesScopeEnum.TRACKED,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files in '.' with pattern '*.py' (tracked only)"
    assert message == expected_message

    # Test with untracked_only
    input_data = FindFilesInput(
        directory=".",
        name_pattern="*.py",
        files_scope=FilesScopeEnum.UNTRACKED,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files in '.' with pattern '*.py' (untracked only)"
    assert message == expected_message

    # Test with modified
    input_data = FindFilesInput(
        directory=".",
        name_pattern="*.py",
        files_scope=FilesScopeEnum.MODIFIED,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files in '.' with pattern '*.py' (modified only)"
    assert message == expected_message

    # Test with deleted
    input_data = FindFilesInput(
        directory=".",
        name_pattern="*.py",
        files_scope=FilesScopeEnum.DELETED,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files in '.' with pattern '*.py' (deleted only)"
    assert message == expected_message


def test_ls_files_format_display_message():
    tool = LsFiles(description="List files description")

    input_data = LsFilesInput(directory="./src")

    message = tool.format_display_message(input_data)

    expected_message = "List files in './src'"
    assert message == expected_message


def test_grep_format_display_message():
    tool = Grep(description="Grep description")

    # Basic test with directory
    input_data = GrepInput(pattern="TODO", search_directory="./src")
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with options
    input_data = GrepInput(
        pattern="TODO", search_directory="./src", recursive=True, case_insensitive=True
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message

    # Test with all options
    input_data = GrepInput(
        pattern="TODO",
        search_directory="./src",
        recursive=True,
        case_insensitive=True,
        include_untracked=True,
        files_with_matches=True,
        files_without_match=True,
        no_recursive=True,
        fixed_strings=True,
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in files in './src'"
    assert message == expected_message


def test_grep_format_display_message_no_directory():
    tool = Grep(description="Grep description")

    # Basic test with no directory
    input_data = GrepInput(pattern="TODO", search_directory=None)
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in directory"
    assert message == expected_message

    # Test with options and no directory
    input_data = GrepInput(
        pattern="TODO", search_directory=None, recursive=True, case_insensitive=True
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search for 'TODO' in directory"
    assert message == expected_message


def test_mkdir_format_display_message():
    tool = Mkdir(description="Mkdir description")

    input_data = MkdirInput(directory_path="./src/new_directory")

    message = tool.format_display_message(input_data)

    expected_message = "Create directory './src/new_directory'"
    assert message == expected_message


def test_edit_file_format_display_message():
    tool = EditFile(description="Edit file description")

    input_data = EditFileInput(
        file_path="./src/main.py",
        old_str="print('Hello')",
        new_str="print('Hello, world!')",
    )

    message = tool.format_display_message(input_data)

    expected_message = "Edit file"
    assert message == expected_message
