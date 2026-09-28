# pylint: disable=too-many-lines
import base64
import json
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.entities.image_response import (
    supported_image_formats_display,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.filesystem import (  # Mkdir,
    DEFAULT_CONTEXT_EXCLUSIONS,
    DEFAULT_READ_FILE_LIMIT,
    DEFAULT_READ_FILE_OFFSET,
    IMAGE_READ_CAPABILITY,
    TRUSTED_ABSOLUTE_PATH_SEGMENTS,
    EditFile,
    EditFileInput,
    FindFiles,
    FindFilesInput,
    ListDir,
    ListDirInput,
    Mkdir,
    MkdirInput,
    ReadFile,
    ReadFileChunked,
    ReadFileInput,
    ReadFiles,
    ReadFilesInput,
    WriteFile,
    WriteFileInput,
    _is_trusted_absolute_path,
    validate_duo_context_exclusions,
)
from lib.context import client_capabilities, gitlab_user_id, gitlab_version
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context
from tests.duo_workflow_service.tools.conftest import (
    create_mock_client_event_with_image_response,
    create_mock_client_event_with_response,
)
from tests.duo_workflow_service.tools.constants import (
    NORMAL_FILES,
    SENSITIVE_DIRECTORIES,
    SENSITIVE_FILES,
    SUSPICIOUS_PATHS,
)


@pytest.fixture(name="mock_project")
def mock_project_fixture():
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=None,
    )


@pytest.fixture(name="metadata_with_project")
def metadata_with_project_fixture(mock_project):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=create_mock_client_event_with_response("test contents")
    )

    return {"outbox": mock_outbox, "project": mock_project}


@pytest.mark.asyncio
async def test_read_file(metadata_with_project):
    tool = ReadFile(description="Read file content")
    tool.metadata = metadata_with_project
    path = "./somepath"

    response = await tool._arun(path)

    assert response == "test contents"

    outbox = metadata_with_project["outbox"]
    outbox.put_action_and_wait_for_response.assert_called_once()
    action = outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runReadFile.filepath == path


@pytest.mark.asyncio
async def test_read_file_not_implemented_error():
    tool = ReadFile(description="Read file content")

    with pytest.raises(NotImplementedError):
        tool._run("./main.py")


@pytest.mark.asyncio
async def test_write_file(mock_project):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=create_mock_client_event_with_response("done")
    )

    metadata = {"outbox": mock_outbox, "project": mock_project}

    tool = WriteFile(description="Write file content")
    tool.metadata = metadata
    path = "./somepath"
    contents = "test contents"

    response = await tool._arun(path, contents)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response("file1.py\nfile2.py")
        )

        metadata = {"outbox": mock_outbox, "project": mock_project}
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
        # Set up the mock outbox
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(
                "file1.txt file2.txt dir1 dir2"
            )
        )

        metadata = {"outbox": mock_outbox, "project": mock_project}

        # Create the tool and set its metadata
        list_dir_tool = ListDir()
        list_dir_tool.metadata = metadata

        # Call the method being tested
        result = await list_dir_tool._arun(directory=".")

        # Assert the result
        assert result == "file1.txt file2.txt dir1 dir2"

        # Verify the outbox was used as expected
        mock_outbox.put_action_and_wait_for_response.assert_called_once()

        # You can add additional assertions to verify the details of what was put on the outbox
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(response="")
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        mkdir_tool = Mkdir()
        mkdir_tool.metadata = metadata
        result = await mkdir_tool._arun("./test_dir")

        assert result == ""

        # Verify the action sent to outbox
        mock_outbox.put_action_and_wait_for_response.assert_called_once()
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.mkdir.directory_path == "./test_dir"

    @pytest.mark.asyncio
    async def test_mkdir_creates_nested_directories(self):
        # Set up mock outbox following the pattern from other tests
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(response="")
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        # Create the tool and set its metadata
        mkdir_tool = Mkdir()
        mkdir_tool.metadata = metadata

        # Call the method being tested
        result = await mkdir_tool._arun("./test_dir/nested/dir")

        # Assert the result
        assert result == ""

        # Verify the outbox was called correctly
        mock_outbox.put_action_and_wait_for_response.assert_called_once()

        # Verify the action details
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response("success")
        )

        metadata = {"outbox": mock_outbox, "project": mock_project}

        tool = EditFile(metadata=metadata)
        path = "./somefile.txt"
        old_str = "old line"
        new_str = "new line"

        response = await tool._arun(path, old_str, new_str)

        assert response == "success"

        mock_outbox.put_action_and_wait_for_response.assert_called_once()
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
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

    @pytest.mark.asyncio
    async def test_read_file_sends_filepath_only(self, metadata_with_project):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project
        path = "./somepath"

        response = await tool._arun(path)

        assert response == "test contents"

        outbox = metadata_with_project["outbox"]
        outbox.put_action_and_wait_for_response.assert_called_once()
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path
        assert action.runReadFile.offset == 0
        assert action.runReadFile.limit == 0

    UPLOAD_SECRET = "0123456789abcdef0123456789abcdef"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_rewrites_upload_reference_to_api_path(
        self, metadata_with_project
    ):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/screenshot.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == (
            f"/api/v4/projects/1/uploads/{self.UPLOAD_SECRET}/screenshot.png"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_percent_encodes_upload_filename(
        self, metadata_with_project
    ):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/a b&c.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath.endswith("/a%20b%26c.png")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_accepts_slashless_upload_reference(
        self, metadata_with_project
    ):
        # Models routinely normalize the markdown's `/uploads/...` to a
        # repo-relative-looking `uploads/...` (observed live in the M3 run).
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun(f"uploads/{self.UPLOAD_SECRET}/screenshot.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == (
            f"/api/v4/projects/1/uploads/{self.UPLOAD_SECRET}/screenshot.png"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_accepts_legacy_ten_hex_upload_secret(
        self, metadata_with_project
    ):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun("/uploads/0123456789/old.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == (
            "/api/v4/projects/1/uploads/0123456789/old.png"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            # Wrong secret alphabet / length.
            "/uploads/NOT-A-SECRET/x.png",
            "/uploads/0123456789abcdef/../../../etc/passwd",
            # Subdirectory in the filename (never valid for uploads).
            "/uploads/0123456789abcdef0123456789abcdef/sub/dir.png",
            # Prefixed lookalikes must not match.
            "/var/uploads/0123456789abcdef0123456789abcdef/x.png",
            # Directory addresses, not file reads.
            "/uploads/0123456789abcdef0123456789abcdef/.",
            "/uploads/0123456789abcdef0123456789abcdef/..",
        ],
    )
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_denies_upload_lookalikes(
        self, path, metadata_with_project
    ):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun(path)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_slashless_short_secret_stays_a_workspace_path(
        self, metadata_with_project
    ):
        # `uploads/<10 hex>/logo.png` is a believable repository directory, and
        # without the leading slash nothing else tells them apart.
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project
        path = "uploads/0123456789/logo.png"

        await tool._arun(path)

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_legacy_short_secret_still_resolves_in_markdown_form(
        self, metadata_with_project
    ):
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun("/uploads/0123456789/logo.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert (
            action.runReadFile.filepath
            == "/api/v4/projects/1/uploads/0123456789/logo.png"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_upload_read_is_logged_without_the_secret(
        self, metadata_with_project
    ):
        # Rails already audits the tool call; this records the service deciding
        # to ask. The secret is part of the URL, so it stays out of the event.
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        user_token = gitlab_user_id.set("42")
        try:
            with patch(
                "duo_workflow_service.tools.filesystem._security_log"
            ) as security_log:
                await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/screenshot.png")
        finally:
            gitlab_user_id.reset(user_token)

        security_log.info.assert_called_once()
        _, fields = security_log.info.call_args
        # The standard's common fields, in its names and placement, with
        # outcome inside details and standard values only.
        uuid.UUID(fields["id"])
        assert fields["event_type"] == "data.read.upload"
        assert fields["author_id"] == 42
        assert fields["entity_type"] == "Project"
        assert fields["entity_id"] == 1
        assert fields["entity_path"] == "repo"
        assert fields["target_type"] == "upload"
        assert fields["target_details"] == "screenshot.png"
        assert fields["details"] == {
            "outcome": "success",
            "provider": "duo_workflow_service",
            "response_type": "text",
        }
        assert fields["gitlab"] == {"data_type": "upload", "size_bytes": None}
        assert self.UPLOAD_SECRET not in str(security_log.info.call_args)

        # structlog forwards these as LogRecord extras in some configurations,
        # where shadowing a built-in attribute (filename, module, args, ...)
        # raises instead of logging. Local config may not, CI does.
        reserved = set(logging.LogRecord("n", 20, "p", 1, "m", None, None).__dict__)
        assert not set(fields) & reserved

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_failed_upload_download_is_still_logged(self, metadata_with_project):
        # A failed download is still credential spend the user may need to
        # account for, so the trail records the attempt, not just successes.
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project
        metadata_with_project["outbox"].put_action_and_wait_for_response = AsyncMock(
            side_effect=ToolException("download failed")
        )

        with patch(
            "duo_workflow_service.tools.filesystem._security_log"
        ) as security_log:
            with pytest.raises(ToolException, match="download failed"):
                await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/screenshot.png")

        security_log.info.assert_called_once()
        _, fields = security_log.info.call_args
        assert fields["details"]["outcome"] == "failure"
        assert fields["details"]["response_type"] is None
        assert fields["target_details"] == "screenshot.png"
        assert self.UPLOAD_SECRET not in str(security_log.info.call_args)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_denies_upload_reference_without_project(self):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock()
        tool = ReadFile(description="Read file content")
        tool.metadata = {"outbox": mock_outbox, "project": None}

        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/screenshot.png")

        mock_outbox.put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path_prefix", ["/", ""])
    @pytest.mark.usefixtures("image_support_enabled")
    async def test_read_file_chunked_rewrites_upload_reference(
        self, metadata_with_project, path_prefix
    ):
        # The class modern clients actually get, so it needs the same branch.
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project

        await tool._arun(f"{path_prefix}uploads/{self.UPLOAD_SECRET}/screenshot.png")

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == (
            f"/api/v4/projects/1/uploads/{self.UPLOAD_SECRET}/screenshot.png"
        )
        assert action.runReadFile.offset == 0
        assert action.runReadFile.limit == 0

    # Both classes: one supersedes the other under the same tool name.
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("image_support_enabled")
    @pytest.mark.parametrize("tool_class", [ReadFile, ReadFileChunked])
    async def test_upload_image_response_converts_for_both_read_tools(
        self, tool_class, mock_project
    ):
        payload = b"\x89PNG\r\n\x1a\n" + b"not real pixels"
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_image_response(
                "image/png", payload
            )
        )
        tool = tool_class(description="Read file content")
        tool.metadata = {"outbox": mock_outbox, "project": mock_project}

        with patch(
            "duo_workflow_service.tools.filesystem._security_log"
        ) as security_log:
            response = await tool._arun(f"/uploads/{self.UPLOAD_SECRET}/screenshot.png")

        assert isinstance(response, list)
        assert response[1]["type"] == "image"
        assert base64.b64decode(response[1]["base64"]) == payload

        # The data-access event records what was pulled in: an upload of this
        # many decoded bytes, delivered as an image.
        _, fields = security_log.info.call_args
        assert fields["details"]["response_type"] == "image"
        assert fields["gitlab"]["size_bytes"] == len(payload)


class TestReadFileChunked:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path", [*SENSITIVE_DIRECTORIES, *SENSITIVE_FILES, *SUSPICIOUS_PATHS]
    )
    async def test_rejects_excluded_paths(self, path):
        with pytest.raises(ToolException, match="Access denied"):
            await ReadFileChunked(description="Read file content")._arun(path)

    @pytest.mark.asyncio
    async def test_with_offset_and_limit(self, metadata_with_project):
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project
        path = "./somepath"

        response = await tool._arun(path, offset=10, limit=20)

        assert response == "test contents"

        outbox = metadata_with_project["outbox"]
        outbox.put_action_and_wait_for_response.assert_called_once()
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path
        assert action.runReadFile.offset == 10
        assert action.runReadFile.limit == 20

    @pytest.mark.asyncio
    async def test_with_offset_only(self, metadata_with_project):
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project
        path = "./somepath"

        response = await tool._arun(path, offset=5)

        assert response == "test contents"

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path
        assert action.runReadFile.offset == 5
        assert action.runReadFile.limit == DEFAULT_READ_FILE_LIMIT

    @pytest.mark.asyncio
    async def test_without_offset_and_limit(self, metadata_with_project):
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project
        path = "./somepath"

        response = await tool._arun(path)

        assert response == "test contents"

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path
        assert action.runReadFile.offset == DEFAULT_READ_FILE_OFFSET
        assert action.runReadFile.limit == DEFAULT_READ_FILE_LIMIT

    @pytest.mark.asyncio
    async def test_with_zero_offset(self, metadata_with_project):
        """Ensure offset=0 is not falsily converted — it should remain 0, not be treated as None."""
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project
        path = "./somepath"

        await tool._arun(path, offset=0, limit=50)

        action = metadata_with_project[
            "outbox"
        ].put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.offset == 0
        assert action.runReadFile.limit == 50

    def test_supersedes_read_file(self):
        assert ReadFileChunked.supersedes is ReadFile

    def test_required_capability(self):
        assert ReadFileChunked.required_capability == frozenset({"read_file_chunked"})

    def test_shares_tool_name_with_read_file(self):
        assert ReadFileChunked.model_fields["name"].default == "read_file"


class TestReadFiles:
    @pytest.mark.asyncio
    async def test_read_files_with_mixed_valid_invalid_paths(self):
        mock_outbox = MagicMock()

        # Mock response with mixed success and error
        mock_response = '{"file1.py": {"content": "print(\'hello\')"}, "nonexistent.py": {"error": "File not found"}}'

        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response=mock_response
                    )
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        response = await tool._arun(file_paths)

        assert response == mock_response

        mock_outbox.put_action_and_wait_for_response.assert_called_once()
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFiles.filepaths == file_paths

    @pytest.mark.asyncio
    async def test_read_files_with_no_action_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(actionResponse=None)
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        # Empty protobuf response leads to empty string which causes JSONDecodeError
        with pytest.raises(
            ToolException, match="Could not parse file contents as JSON"
        ):
            await tool._arun(file_paths)

    @pytest.mark.asyncio
    async def test_read_files_with_none_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(actionResponse=None)
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        # Empty protobuf response leads to empty string which causes JSONDecodeError
        with pytest.raises(
            ToolException, match="Could not parse file contents as JSON"
        ):
            await tool._arun(file_paths)

    @pytest.mark.asyncio
    async def test_read_files_with_error_response(self):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        error="Error reading files"
                    )
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "nonexistent.py"]

        with pytest.raises(ToolException, match="Action error: Error reading files"):
            await tool._arun(file_paths)

    @pytest.mark.asyncio
    async def test_read_files_with_json_decode_error(self):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response="invalid json {{{", error=""
                    )
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "file2.py"]

        with pytest.raises(
            ToolException, match="Could not parse file contents as JSON"
        ):
            await tool._arun(file_paths)

    @pytest.mark.asyncio
    async def test_read_files_with_json_decode_error_and_executor_error(self):
        # Note: This test verifies that when plainTextResponse has both error and response,
        # the error is raised by _execute_action_and_get_action_response before we even
        # try to parse JSON, so we get the Action error, not the JSON parse error
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response="invalid json {{{", error="Some executor error"
                    )
                )
            )
        )

        metadata = {"outbox": mock_outbox}

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = metadata
        file_paths = ["file1.py", "file2.py"]

        # When plainTextResponse.error is set, _execute_action_and_get_action_response
        # raises before we get to JSON parsing
        with pytest.raises(ToolException, match="Action error: Some executor error"):
            await tool._arun(file_paths)

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
        mock_response = '{"allowed_file.py": {"content": "hi"}}'
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response=mock_response
                    )
                )
            )
        )

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = {
            "outbox": mock_outbox,
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
            action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
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
    expected_message = "Search files with pattern `*.py`"
    assert message == expected_message

    # Test with tracked_only
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern `*.py`"
    assert message == expected_message

    # Test with untracked_only
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern `*.py`"
    assert message == expected_message

    # Test with modified
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern `*.py`"
    assert message == expected_message

    # Test with deleted
    input_data = FindFilesInput(
        name_pattern="*.py",
    )
    message = tool.format_display_message(input_data)
    expected_message = "Search files with pattern `*.py`"
    assert message == expected_message


def test_mkdir_format_display_message():
    tool = Mkdir(description="Mkdir description")

    input_data = MkdirInput(directory_path="./src/new_directory")

    message = tool.format_display_message(input_data)

    expected_message = "Create directory `./src/new_directory`"
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


class TestIsTrustedAbsolutePath:
    """Unit tests for the _is_trusted_absolute_path helper."""

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
            "/home/user/.agents/skills/ai-gateway-aws/SKILL.md",
            # Trusted sequence appearing at the filesystem root
            "/.agents/skills/glab/SKILL.md",
            "/.agents/skills",
            # Multi-segment trusted entry: .gitlab/duo/skills
            "/home/user/.gitlab/duo/skills/y/SKILL.md",
            # XDG_CONFIG_HOME variant: .config/gitlab/duo/skills
            "/home/user/.config/gitlab/duo/skills/z/SKILL.md",
            # Per-user Duo plugins store: .gitlab/duo/plugins
            "/home/user/.gitlab/duo/plugins/my-plugin/skill.md",
            # XDG_CONFIG_HOME variant: .config/gitlab/duo/plugins
            "/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md",
        ],
    )
    def test_trusted_paths_return_true(self, path):
        assert _is_trusted_absolute_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            # Non-absolute paths are never trusted
            ".agents/skills/glab/SKILL.md",
            "home/user/.agents/skills/glab/SKILL.md",
            # Lookalikes — segment must match exactly, not as substring
            "/x/.agentsfoo/SKILL.md",
            "/tmp/evil.agents-data/x",
            "/home/user/myagents/SKILL.md",
            # All trusted segments present but not consecutive (extra dir in-between)
            "/home/user/.agents/random/skills/glab/SKILL.md",
            # Trusted roots are only readable under their `skills/` subdir, not the
            # config root itself — .gitlab/duo/<file> stays denied.
            "/root/.gitlab/duo/chat-rules.md",
            "/home/user/.agents/config.yml",
            # Non-dotted `gitlab/duo/skills` inside a repo checkout must NOT be trusted —
            # only the dotfile-anchored `.config/gitlab/duo/skills` XDG variant is.
            "/builds/group/project/gitlab/duo/skills/secret.env",
            # Likewise for the plugins store: a repo checkout containing a non-dotted
            # `gitlab/duo/plugins/` directory must NOT be trusted.
            "/builds/group/project/gitlab/duo/plugins/secret.env",
            # Traversal alongside a trusted segment must fail closed even standalone.
            "/home/user/.agents/skills/../../../etc/passwd",
            "/home/user/.gitlab/duo/skills/..%2f..%2fetc/passwd",
            "/home/user/.gitlab/duo/plugins/../../etc/passwd",
            # Sensitive absolute paths that must stay rejected
            "/etc/passwd",
            "/home/user/.ssh/id_rsa",
            "/home/user/secrets/.env",
            # Relative paths
            "relative/path/file.md",
            "",
        ],
    )
    def test_non_trusted_paths_return_false(self, path):
        assert _is_trusted_absolute_path(path) is False

    def test_every_trusted_segment_is_dotfile_anchored(self):
        """Each entry's first segment must start with '.' so it cannot appear at a repo checkout root."""
        for entry in TRUSTED_ABSOLUTE_PATH_SEGMENTS:
            assert entry.split("/", maxsplit=1)[0].startswith("."), (
                f"{entry!r} is not dotfile-anchored"
            )


class TestValidateDuoContextExclusionsTrustedAbsolute:
    """Tests for the allow_trusted_absolute parameter of validate_duo_context_exclusions."""

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
            "/home/user/.gitlab/duo/skills/y/SKILL.md",
            "/home/user/.gitlab/duo/plugins/my-plugin/skill.md",
            "/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md",
        ],
    )
    def test_trusted_absolute_allowed_when_flag_is_true(self, path):
        """Trusted absolute paths must not raise when allow_trusted_absolute=True."""
        # Should not raise
        validate_duo_context_exclusions(path, allow_trusted_absolute=True)

    def test_empty_path_returns_early_when_flag_is_true(self):
        """The ``if not file_path`` guard must fire before the trusted-absolute check."""
        # Should not raise
        validate_duo_context_exclusions("", allow_trusted_absolute=True)

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
            "/home/user/.gitlab/duo/skills/y/SKILL.md",
            "/home/user/.gitlab/duo/plugins/my-plugin/skill.md",
            "/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md",
        ],
    )
    def test_trusted_absolute_rejected_when_flag_is_false(self, path):
        """Trusted absolute paths must still be rejected when allow_trusted_absolute=False (default)."""
        with pytest.raises(ToolException, match="Access denied"):
            validate_duo_context_exclusions(path, allow_trusted_absolute=False)

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",
            "/home/user/.ssh/id_rsa",
            "/home/user/secrets/.env",
        ],
    )
    def test_non_trusted_absolute_rejected_even_when_flag_is_true(self, path):
        """Non-trusted absolute paths must be rejected regardless of the flag."""
        with pytest.raises(ToolException, match="Access denied"):
            validate_duo_context_exclusions(path, allow_trusted_absolute=True)

    @pytest.mark.parametrize(
        "path",
        [
            # Traversal under a trusted prefix must still be rejected
            "/home/user/.agents/../../etc/passwd",
            "/root/.agents/../../../etc/shadow",
        ],
    )
    def test_traversal_rejected_even_under_trusted_prefix(self, path):
        """Path traversal must be caught before the trusted-absolute allow, even for trusted prefixes."""
        with pytest.raises(ToolException, match="Access denied"):
            validate_duo_context_exclusions(path, allow_trusted_absolute=True)

    @pytest.mark.parametrize(
        "path",
        [
            # Lookalikes — segment must match exactly
            "/x/.agentsfoo/SKILL.md",
            "/tmp/evil.agents-data/x",
        ],
    )
    def test_lookalike_segments_rejected(self, path):
        """Paths that look like trusted segments but are not exact matches must be rejected."""
        with pytest.raises(ToolException, match="Access denied"):
            validate_duo_context_exclusions(path, allow_trusted_absolute=True)

    def test_repo_relative_gitlab_duo_still_denied(self):
        """Repo-relative .gitlab/duo paths must remain denied (denylist unchanged)."""
        with pytest.raises(ToolException, match="Access denied"):
            validate_duo_context_exclusions(
                ".gitlab/duo/chat-rules.md", allow_trusted_absolute=True
            )


class TestToolsTrustedAbsolutePaths:
    """Integration tests covering all filesystem tools: read-only tools allow trusted absolute paths, while
    write/edit/list tools reject them."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
            "/home/user/.gitlab/duo/skills/y/SKILL.md",
            "/home/user/.gitlab/duo/plugins/my-plugin/skill.md",
            "/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md",
        ],
    )
    async def test_read_file_allows_trusted_absolute(self, metadata_with_project, path):
        """ReadFile must dispatch the action for trusted absolute paths."""
        tool = ReadFile(description="Read file content")
        tool.metadata = metadata_with_project

        response = await tool._arun(path)

        assert response == "test contents"
        outbox = metadata_with_project["outbox"]
        outbox.put_action_and_wait_for_response.assert_called_once()
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
            "/home/user/.gitlab/duo/skills/y/SKILL.md",
            "/home/user/.gitlab/duo/plugins/my-plugin/skill.md",
            "/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md",
        ],
    )
    async def test_read_file_chunked_allows_trusted_absolute(
        self, metadata_with_project, path
    ):
        """ReadFileChunked must dispatch the action for trusted absolute paths."""
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = metadata_with_project

        response = await tool._arun(path)

        assert response == "test contents"
        outbox = metadata_with_project["outbox"]
        outbox.put_action_and_wait_for_response.assert_called_once()
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "paths",
        [
            ["/home/user/.agents/skills/glab/SKILL.md"],
            ["/root/.agents/skills/x/SKILL.md"],
            ["/home/user/.gitlab/duo/skills/y/SKILL.md"],
            ["/home/user/.gitlab/duo/plugins/my-plugin/skill.md"],
            ["/home/user/.config/gitlab/duo/plugins/my-plugin/skill.md"],
            [
                "/home/user/.agents/skills/glab/SKILL.md",
                "/root/.agents/skills/x/SKILL.md",
                "/home/user/.gitlab/duo/skills/y/SKILL.md",
            ],
        ],
    )
    async def test_read_files_allows_trusted_absolute(self, paths):
        """ReadFiles must dispatch the action for trusted absolute paths."""
        mock_outbox = MagicMock()
        mock_response = json.dumps(
            {filepath: {"content": "test contents"} for filepath in paths}
        )
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response=mock_response
                    )
                )
            )
        )

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = {"outbox": mock_outbox}

        response = await tool._arun(paths)

        assert response == mock_response
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFiles.filepaths == paths

    @pytest.mark.asyncio
    async def test_read_files_mixed_batch_aborts_on_non_trusted_absolute(self):
        """A non-trusted absolute path anywhere in the batch must abort the whole read.

        ReadFiles validates each path in a loop before dispatching, so a single bad path (here ``/etc/passwd``) must
        raise before any action is dispatched, even when a valid relative path and a trusted absolute path are also
        present.
        """
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock()

        tool = ReadFiles(description="Read multiple files")
        tool.metadata = {"outbox": mock_outbox}

        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun(
                [
                    "src/main.py",
                    "/root/.agents/skills/x/SKILL.md",
                    "/etc/passwd",
                ]
            )

        mock_outbox.put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
        ],
    )
    async def test_write_file_rejects_trusted_absolute(self, path):
        """WriteFile must reject trusted absolute paths (read-only scope)."""
        with pytest.raises(ToolException, match="Access denied"):
            await WriteFile(description="Write file content")._arun(path, "contents")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
        ],
    )
    async def test_edit_file_rejects_trusted_absolute(self, path):
        """EditFile must reject trusted absolute paths (read-only scope)."""
        with pytest.raises(ToolException, match="Access denied"):
            await EditFile(description="Edit file content")._arun(path, "old", "new")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/.agents/skills/glab/SKILL.md",
            "/root/.agents/skills/x/SKILL.md",
        ],
    )
    async def test_list_dir_rejects_trusted_absolute(self, path):
        """ListDir must reject trusted absolute paths (read-only scope)."""
        with pytest.raises(ToolException, match="Access denied"):
            await ListDir(description="List directory")._arun(path)


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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(
                "file1.txt\nfile2.secret\nconfig/private/secret.txt\nconfig/private/allowed.txt\ntemp/cache.txt"
            )
        )

        metadata = {
            "outbox": mock_outbox,
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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(
                "file1.txt\nfile2.secret\nconfig/private/secret.txt\nconfig/private/allowed.txt"
            )
        )

        metadata = {
            "outbox": mock_outbox,
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
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response="temp/cache.txt\ntemp/log.txt"
                    )
                )
            )
        )

        metadata = {
            "outbox": mock_outbox,
            "project": project_with_exclusions,
        }

        tool = ListDir()
        tool.metadata = metadata

        # Test directory containing only excluded files
        result = await tool._arun("temp/")
        # Should return empty string since all files in temp/ are excluded
        assert result == ""


@pytest.fixture(name="image_flag_enabled")
def image_flag_enabled_fixture():
    """Turn on the instance flag only; the client half of the switch stays off."""
    token = current_feature_flag_context.set({FeatureFlag.DAP_TOOL_IMAGE_INPUT.value})
    yield
    current_feature_flag_context.reset(token)


@pytest.fixture(name="image_client_capable")
def image_client_capable_fixture():
    """Declare the client capability only, on a GitLab version that forwards capabilities."""
    caps_token = client_capabilities.set({IMAGE_READ_CAPABILITY})
    version_token = gitlab_version.set("19.5.0")
    yield
    gitlab_version.reset(version_token)
    client_capabilities.reset(caps_token)


@pytest.fixture(name="image_support_enabled")
def image_support_enabled_fixture():
    """Both switches on: what a capable client on a flagged instance sees."""
    flag_token = current_feature_flag_context.set(
        {FeatureFlag.DAP_TOOL_IMAGE_INPUT.value}
    )
    caps_token = client_capabilities.set({IMAGE_READ_CAPABILITY})
    version_token = gitlab_version.set("19.5.0")
    yield
    gitlab_version.reset(version_token)
    client_capabilities.reset(caps_token)
    current_feature_flag_context.reset(flag_token)


@pytest.mark.usefixtures("image_support_enabled")
class TestImageResponseConversion:
    """read_file tools convert typed executor image results into content blocks."""

    FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"not real pixels" * 4
    FAKE_PNG_BASE64 = base64.b64encode(FAKE_PNG_BYTES).decode()

    def metadata_with_image_response(
        self, mock_project, mime_type: str, data: bytes
    ) -> dict:
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_image_response(mime_type, data)
        )
        return {"outbox": mock_outbox, "project": mock_project}

    def metadata_with_response(self, mock_project, response: str) -> dict:
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(response)
        )
        return {"outbox": mock_outbox, "project": mock_project}

    @pytest.mark.asyncio
    async def test_read_file_converts_image_response(self, mock_project):
        tool = ReadFile(description="Read file content")
        tool.metadata = self.metadata_with_image_response(
            mock_project, "image/png", self.FAKE_PNG_BYTES
        )

        result = await tool._arun("./screenshot.png")

        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "./screenshot.png" in result[0]["text"]
        assert result[1]["type"] == "image"
        assert result[1]["base64"] == self.FAKE_PNG_BASE64
        assert result[1]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_read_file_chunked_converts_image_response(self, mock_project):
        tool = ReadFileChunked(description="Read file content")
        tool.metadata = self.metadata_with_image_response(
            mock_project, "image/png", self.FAKE_PNG_BYTES
        )

        result = await tool._arun("./screenshot.png")

        assert isinstance(result, list)
        assert result[1]["type"] == "image"
        assert result[1]["base64"] == self.FAKE_PNG_BASE64

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("file_name", "magic", "mime_type"),
        [
            ("shot.png", b"\x89PNG\r\n\x1a\n", "image/png"),
            ("photo.jpeg", b"\xff\xd8\xff\xe0", "image/jpeg"),
            (
                "pic.webp",
                b"RIFF\x10\x00\x00\x00WEBP",
                "image/webp",
            ),
        ],
    )
    async def test_every_supported_type_converts_through_the_tool(
        self, mock_project, file_name, magic, mime_type
    ):
        payload = magic + b"not real pixels"
        tool = ReadFile(description="Read file content")
        tool.metadata = self.metadata_with_image_response(
            mock_project, mime_type, payload
        )

        result = await tool._arun(f"./{file_name}")

        assert isinstance(result, list)
        assert result[1]["mime_type"] == mime_type
        assert base64.b64decode(result[1]["base64"]) == payload

    @pytest.mark.asyncio
    async def test_invalid_image_response_becomes_readable_error(self, mock_project):
        # PNG bytes declared as JPEG: caught at conversion, returned to the
        # model as a string instead of an opaque provider failure.
        tool = ReadFile(description="Read file content")
        tool.metadata = self.metadata_with_image_response(
            mock_project, "image/jpeg", self.FAKE_PNG_BYTES
        )

        result = await tool._arun("./photo.jpeg")

        assert isinstance(result, str)
        assert "does not match the declared type" in result

    @pytest.mark.asyncio
    async def test_read_file_plain_text_stays_string(self, mock_project):
        tool = ReadFile(description="Read file content")
        tool.metadata = self.metadata_with_response(mock_project, "plain content")

        result = await tool._arun("./notes.txt")

        assert result == "plain content"

    @pytest.mark.asyncio
    async def test_read_files_is_not_image_capable(self, mock_project):
        """ReadFiles unwraps plainTextResponse directly: an image response to
        its action carries no JSON payload, so it surfaces the decode error
        instead of silently converting."""
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_image_response(
                "image/png", self.FAKE_PNG_BYTES
            )
        )
        tool = ReadFiles(description="Read files content")
        tool.metadata = {"outbox": mock_outbox, "project": mock_project}

        with pytest.raises(
            ToolException, match="Could not parse file contents as JSON"
        ):
            await tool._arun(["./screenshot.png"])


@pytest.mark.usefixtures("image_support_enabled")
class TestImageSupportAdvertised:
    """The model refuses to read images unless the tool says it can (observed live): the descriptions must advertise
    image support."""

    # Instantiate rather than reading the class default: the description a
    # model sees is the instance's, after every model validator has run.
    @pytest.mark.parametrize("tool_class", [ReadFile, ReadFileChunked])
    def test_description_mentions_images(self, tool_class):
        # Derived from the allowlist, so the advertised list cannot drift from
        # what conversion actually accepts.
        assert (
            f"Image files ({supported_image_formats_display()}) are supported"
            in tool_class().description
        )

    def test_read_files_points_images_at_read_file(self):
        assert "read them individually with read_file" in ReadFiles().description


class TestImageSupportGated:
    """Two fail-closed switches: without both, tool-read image support must be invisible and inert."""

    def test_capability_name_is_the_wire_contract(self):
        # The client declares it and Workhorse allowlists it under this exact
        # string; renaming it here would silently turn image support off
        # everywhere.
        assert IMAGE_READ_CAPABILITY == "read_file_image"

    PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"not real pixels"

    def metadata_with_image(self, mock_project) -> dict:
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_image_response(
                "image/png", self.PNG_BYTES
            )
        )
        return {"outbox": mock_outbox, "project": mock_project}

    def metadata_with_response(self, mock_project, response: str) -> dict:
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response(response)
        )
        return {"outbox": mock_outbox, "project": mock_project}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked])
    async def test_image_becomes_a_refusal_when_flag_is_off(
        self, tool_cls, mock_project
    ):
        # A new client emits image responses regardless of the server-side
        # flag; the model must see a readable refusal, never a surprise.
        tool = tool_cls()
        tool.metadata = self.metadata_with_image(mock_project)

        response = await tool._arun("./screenshot.png")

        assert response == (
            'Cannot read file: "./screenshot.png" is an image file, and image '
            "support is not enabled on this instance."
        )

    @pytest.mark.asyncio
    async def test_plain_text_passes_through_when_flag_is_off(self, mock_project):
        mock_outbox = MagicMock()
        mock_outbox.put_action_and_wait_for_response = AsyncMock(
            return_value=create_mock_client_event_with_response("plain contents")
        )
        tool = ReadFile()
        tool.metadata = {"outbox": mock_outbox, "project": mock_project}

        assert await tool._arun("./notes.txt") == "plain contents"

    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked, ReadFiles])
    def test_descriptions_carry_no_image_lines_when_flag_is_off(self, tool_cls):
        description = tool_cls().description
        assert "Image files" not in description
        # The upload paragraph lives in the same note constants, so the same
        # strip must remove it.
        assert "uploaded" not in description

    UPLOAD_REF = "/uploads/0123456789abcdef0123456789abcdef/screenshot.png"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked])
    async def test_upload_reference_is_not_rewritten_when_flag_is_off(
        self, tool_cls, mock_project
    ):
        # Pre-feature behavior exactly: the reference is an ordinary absolute
        # path, rejected by the filesystem checks before any action is sent,
        # so a disabled instance never triggers a download.
        tool = tool_cls()
        tool.metadata = self.metadata_with_image(mock_project)

        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun(self.UPLOAD_REF)

        tool.metadata["outbox"].put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.usefixtures("image_flag_enabled")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked])
    async def test_upload_reference_is_not_rewritten_when_the_client_is_not_capable(
        self, tool_cls, mock_project
    ):
        # Flag on, capability absent: an older client could not download the
        # API path anyway, so the reference stays an ordinary path and is
        # refused by the filesystem checks before any action is sent.
        tool = tool_cls()
        tool.metadata = self.metadata_with_image(mock_project)

        with pytest.raises(ToolException, match="Access denied"):
            await tool._arun(self.UPLOAD_REF)

        tool.metadata["outbox"].put_action_and_wait_for_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_slashless_upload_reference_stays_a_plain_path_when_flag_is_off(
        self, mock_project
    ):
        # The relative-looking form passes the path checks and reaches the
        # executor untouched, exactly as on a pre-feature server.
        tool = ReadFile()
        tool.metadata = self.metadata_with_response(mock_project, "not found")

        path = "uploads/0123456789abcdef0123456789abcdef/x.png"
        assert await tool._arun(path) == "not found"

        outbox = tool.metadata["outbox"]
        action = outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.runReadFile.filepath == path

    @pytest.mark.usefixtures("image_flag_enabled")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked])
    async def test_image_is_refused_when_the_client_is_not_capable(
        self, tool_cls, mock_project
    ):
        # Flag on, but the client never declared read_file_image (or Workhorse
        # dropped it on the way): the refusal names the client, not the
        # instance, so a mismatch is diagnosable from the transcript.
        tool = tool_cls()
        tool.metadata = self.metadata_with_image(mock_project)

        response = await tool._arun("./screenshot.png")

        assert response == (
            'Cannot read file: "./screenshot.png" is an image file, and this '
            "client did not declare image support."
        )

    @pytest.mark.usefixtures("image_flag_enabled")
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked, ReadFiles])
    def test_descriptions_carry_no_image_lines_when_the_client_is_not_capable(
        self, tool_cls
    ):
        # The flag alone must not advertise image reads: an older client would
        # plan around vision and still refuse the binary.
        assert "Image files" not in tool_cls().description

    @pytest.mark.usefixtures("image_client_capable")
    @pytest.mark.asyncio
    async def test_capable_client_on_a_flag_off_instance_gets_the_instance_refusal(
        self, mock_project
    ):
        tool = ReadFile()
        tool.metadata = self.metadata_with_image(mock_project)

        response = await tool._arun("./screenshot.png")

        assert response == (
            'Cannot read file: "./screenshot.png" is an image file, and image '
            "support is not enabled on this instance."
        )

    @pytest.mark.usefixtures("image_client_capable")
    @pytest.mark.parametrize("tool_cls", [ReadFile, ReadFileChunked, ReadFiles])
    def test_descriptions_carry_no_image_lines_when_only_the_client_is_capable(
        self, tool_cls
    ):
        assert "Image files" not in tool_cls().description

    @pytest.mark.usefixtures("image_support_enabled")
    @pytest.mark.asyncio
    async def test_image_converts_when_both_switches_are_on(self, mock_project):
        tool = ReadFile()
        tool.metadata = self.metadata_with_image(mock_project)

        response = await tool._arun("./screenshot.png")

        assert isinstance(response, list)
        assert response[1]["type"] == "image"
