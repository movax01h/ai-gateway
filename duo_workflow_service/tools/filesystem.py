import json
from enum import IntEnum
from typing import Any, List, Type

import gitmatch
import structlog
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.policies.file_exclusion_policy import (
    CONTEXT_EXCLUSION_MESSAGE,
    FileExclusionPolicy,
)
from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

# Security denylist of sensitive directories and files that should not be accessed
DEFAULT_CONTEXT_EXCLUSIONS = gitmatch.compile(
    [
        ".config/nvim",
        ".docker",
        ".emacs.d",
        ".env.*",
        ".env",
        ".git",
        ".gitlab/duo",
        ".gitlab/rules",
        ".gnupg",
        ".idea",
        ".metadata",
        ".settings",
        ".ssh",
        ".sublime-project",
        ".sublime-workspace",
        ".vim",
        ".vimrc",
        ".vscode",
        "Dockerfile.secrets",
        "!.env.example",
    ]
)


def validate_duo_context_exclusions(file_path: str) -> None:
    """Check if the given file path is in the managed Duo Context Exclusion denylist of sensitive paths or contains path
    traversal attempts.

    Args:
        file_path: The file path to check

    Raises:
        ToolException: If the path is in the denylist or an invalid path.
    """
    if not file_path:
        return

    file_path = file_path.replace("\\", "/")
    while file_path.startswith("./"):
        file_path = file_path.replace("./", "", 1)
    for pattern in ["../", "..\\", "%2e%2e", "%252e%252e", "\u002e\u002e"]:
        if pattern in file_path:
            raise ToolException(
                f"Access denied: Cannot access '{file_path}' as it contains path traversal patterns"
            )

    try:
        excluded = DEFAULT_CONTEXT_EXCLUSIONS.match(file_path)
        if excluded is not None and bool(excluded):
            raise ToolException(
                f"Access denied: Cannot access '{file_path}' as it matches Duo Context Exclusion"
                f" patterns. Path '{excluded.path}' matches excluded pattern: '{excluded.pattern}'."
            )
    except gitmatch.InvalidPathError as ex:
        raise ToolException(
            f"Access denied: Not accessing invalid path '{file_path}'. {str(ex)}"
        )

    if file_path != file_path.lower():
        validate_duo_context_exclusions(file_path.lower())
        return


class ReadFileInput(BaseModel):
    file_path: str = Field(description="the file_path to read the file from")


class ReadFile(DuoBaseTool):
    name: str = "read_file"
    description: str = """Read the contents of a file.

    IMPORTANT:
    - When a task requires reading multiple files, include batches of tool calls in a single response
    - Do not make separate responses for each file - group related files together

    """
    args_schema: Type[BaseModel] = ReadFileInput
    handle_tool_error: bool = True
    eval_prompts: List[str] = [
        "I need to read the content of the `readme.md`",
        "Let me check if class `DuoBaseTool` exists in `./tools/base.py`",
    ]

    async def _execute(self, file_path: str) -> str:
        # Check file exclusion policy
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        # Check path security before proceeding
        validate_duo_context_exclusions(file_path)

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(runReadFile=contract_pb2.ReadFile(filepath=file_path)),
        )

    def format_display_message(
        self, args: ReadFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Read file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class ReadFilesInput(BaseModel):
    file_paths: list[str] = Field(description="List of file paths to read")


class ReadFiles(DuoBaseTool):
    name: str = "read_files"
    description: str = """Read one or more files in a single operation.
    """
    args_schema: Type[BaseModel] = ReadFilesInput
    handle_tool_error: bool = True

    async def _execute(self, file_paths: list[str]) -> str:
        policy = FileExclusionPolicy(self.project)
        file_paths, excluded_file_paths = policy.filter_allowed(file_paths)
        log = structlog.stdlib.get_logger("workflow")

        for file_path in file_paths:
            validate_duo_context_exclusions(file_path)

        result_dict = {}

        if file_paths:
            file_contents_result_action_response = (
                await _execute_action_and_get_action_response(
                    self.metadata,  # type: ignore
                    contract_pb2.Action(
                        runReadFiles=contract_pb2.ReadFiles(filepaths=file_paths)
                    ),
                )
            )

            if not file_contents_result_action_response:
                log.error("Received empty grpc response")
                return "Could not read files"

            try:
                file_contents_result = (
                    file_contents_result_action_response.plainTextResponse.response
                )
                result_dict = json.loads(file_contents_result)
            except json.JSONDecodeError as e:
                log.error(f"Could not read files: {e}")
                plain_text_response = (
                    file_contents_result_action_response.plainTextResponse
                )
                if plain_text_response:
                    log.info(
                        f"plainTextResponse.response_length={len(plain_text_response.response)}"
                    )
                    if plain_text_response.error:
                        log.info(f"plainTextResponse.error={plain_text_response.error}")

                return "Could not read files"

        # Add excluded files with error messages
        for path in excluded_file_paths:
            result_dict[path] = {"error": CONTEXT_EXCLUSION_MESSAGE}

        # Return as JSON string
        return json.dumps(result_dict)

    def format_display_message(
        self, args: ReadFilesInput, tool_response: Any = None
    ) -> str:
        file_count = len(args.file_paths)
        excluded_files_msg = ""

        if tool_response:
            excluded_files = [
                path
                for path, data in json.loads(tool_response.content).items()
                if data.get("error") == CONTEXT_EXCLUSION_MESSAGE
            ]

            excluded_files_msg = FileExclusionPolicy.format_user_exclusion_message(
                excluded_files
            )

            file_count -= len(excluded_files)

        return f"Read {file_count} file{'s' if file_count != 1 else ''}{excluded_files_msg}"


class WriteFileInput(BaseModel):
    file_path: str = Field(description="the file_path to write the file to")
    contents: str = Field(
        description="the contents to write in the file. *This is required*"
    )


class WriteFile(DuoBaseTool):
    name: str = "create_file_with_contents"
    description: str = (
        "Create and write the given contents to a file. Please specify the `file_path` and the `contents` to write."
    )
    args_schema: Type[BaseModel] = WriteFileInput
    handle_tool_error: bool = True
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, file_path: str, contents: str) -> str:
        # Check file exclusion policy
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        # Check path security before proceeding
        validate_duo_context_exclusions(file_path)

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runWriteFile=contract_pb2.WriteFile(
                    filepath=file_path, contents=contents
                )
            ),
        )

    def format_display_message(
        self, args: WriteFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Create file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class FilesScopeEnum(IntEnum):
    ALL = 0
    TRACKED = 1
    UNTRACKED = 2
    MODIFIED = 3
    DELETED = 4


class FindFilesInput(BaseModel):
    name_pattern: str = Field(description="The pattern to search for files.")


class FindFiles(DuoBaseTool):
    name: str = "find_files"
    description: str = """Find files by name patterns (equivalent to 'find' command).

    **Primary use cases:**
    - Find files by filename or extension patterns
    - Locate specific files across the codebase
    - Get list of files matching naming conventions

    **Replaces these commands:**
    - find . -name "*.py" → find_files(name_pattern="*.py")
    - find tests -name "test_*.js" → find_files(name_pattern="tests/test_*.js")
    - find src -name "*.json" → find_files(name_pattern="src/*.json")

    **Examples:**
    - All Python files: find_files(name_pattern="*.py")
    - Test files: find_files(name_pattern="test_*.py")
    - Config files: find_files(name_pattern="*.json")
    - Files in directory: find_files(name_pattern="src/*.js")

    **Don't use this for:**
    - Searching text content within files (use grep instead)
    - Finding where functions/variables are used (use grep instead)

    Uses bash filename expansion syntax. Searches recursively and respects .gitignore rules.
    """
    args_schema: Type[BaseModel] = FindFilesInput

    async def _execute(
        self,
        name_pattern: str,
    ) -> str:
        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                findFiles=contract_pb2.FindFiles(
                    name_pattern=name_pattern,
                )
            ),
        )

        # Filter results based on file exclusion policy
        policy = FileExclusionPolicy(self.project)
        lines = result.strip().split("\n") if result.strip() else []
        allowed_files, _excluded_files = policy.filter_allowed(lines)

        # Build the response
        response_parts = []
        if allowed_files:
            response_parts.append("\n".join(allowed_files))

        return (
            "\n\n".join(response_parts)
            if response_parts
            else _format_no_matches_message(name_pattern)
        )

    def format_display_message(
        self, args: FindFilesInput, _tool_response: Any = None
    ) -> str:
        return f"Search files with pattern `{args.name_pattern}`"


class MkdirInput(BaseModel):
    directory_path: str = Field(
        description="The directory path to create. Must be within the current working directory tree."
    )


class Mkdir(DuoBaseTool):
    name: str = "mkdir"
    description: str = """Create a new directory using the mkdir command.
    The directory creation is restricted to the current working directory tree."""

    args_schema: Type[BaseModel] = MkdirInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, directory_path: str) -> str:
        if ".." in directory_path:
            return "Creating directories above the current directory is not allowed"

        if not directory_path.startswith("./") and directory_path != ".":
            directory_path = f"./{directory_path}"

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                mkdir=contract_pb2.Mkdir(
                    directory_path=directory_path,
                )
            ),
        )

    def format_display_message(
        self, args: MkdirInput, _tool_response: Any = None
    ) -> str:
        return f"Create directory `{args.directory_path}`"


class EditFileInput(BaseModel):
    file_path: str = Field(description="the path of the file to edit.")
    old_str: str = Field(
        "",
        description="The string to replace. Please provide at least one line above and below to make it unique across "
        "the file. *This is required*",
    )
    new_str: str = Field(
        "", description="The new value of the string. *This is required*"
    )


class EditFile(DuoBaseTool):
    name: str = "edit_file"
    description: str = """Use this tool to edit an existing file.

IMPORTANT:
- When making similar changes to multiple files, include batches of tool calls in a single response
- Do not make separate responses for each file - group related files together

Examples of individual file edits:
- Update a function parameter:
    edit_file(
        file_path="src/utils.py",
        old_str="# Utility functions\n\ndef process_data(data):\n
            # Process the input data\n    return data.upper()\n\n# More functions below",
        new_str="# Utility functions\n\ndef process_data(data, transform=True):\n
            # Process the input data\n    return data.upper() if transform else data\n\n# More functions below"
    )

- Fix a bug in a specific file:
    edit_file(
        file_path="src/api/endpoints.py",
        old_str="# User endpoints\n@app.route('/users/<id>')\ndef get_user(id):\n
            return db.find_user(id)\n\n# Other endpoints",
        new_str="# User endpoints\n@app.route('/users/<id>')\ndef get_user(id):\n
            user = db.find_user(id)\n    return user if user else {'error': 'User not found'}\n\n# Other endpoints"
    )

- Add a new import statement:
    edit_file(
        file_path="src/models.py",
        old_str="import os\nimport sys\n\nclass User:",
        new_str="import os\nimport sys\nimport datetime\n\nclass User:"
    )

Examples of batched file edits:
- Rename a function across multiple files:
    edit_file(
        file_path="src/utils.py",
        old_str="# Configuration functions\ndef get_config():\n    return load_config()\n\n# Other utility functions",
        new_str="# Configuration functions\ndef fetch_config():\n    return load_config()\n\n# Other utility functions"
    )
    edit_file(
        file_path="src/app.py",
        old_str="from utils import get_config\n\nconfig = get_config()\n\n# Application setup",
        new_str="from utils import fetch_config\n\nconfig = fetch_config()\n\n# Application setup"
    )
    edit_file(
        file_path="tests/test_utils.py",
        old_str="# Test configuration\ndef test_get_config():\n    config = get_config()\n    assert config is not None",
        new_str="# Test configuration\ndef test_fetch_config():\n    config = fetch_config()\n    assert config is not None"
    )

- Update version number across the codebase:
    edit_file(
        file_path="src/version.py",
        old_str="# Version information\nVERSION = '1.0.0'\n# End of version info",
        new_str="# Version information\nVERSION = '1.1.0'\n# End of version info"
    )
    edit_file(
        file_path="README.md",
        old_str="# Project Documentation\n\n## MyApp v1.0.0\n\n### Features",
        new_str="# Project Documentation\n\n## MyApp v1.1.0\n\n### Features"
    )
    edit_file(
        file_path="docs/changelog.md",
        old_str="# Changelog\n\n## 1.0.0",
        new_str="# Changelog\n\n## 1.1.0\n- Bug fixes\n- Performance improvements\n\n## 1.0.0"
    )"""
    args_schema: Type[BaseModel] = EditFileInput
    handle_tool_error: bool = True
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, file_path: str, old_str: str, new_str: str) -> str:
        # Check file exclusion policy
        if not FileExclusionPolicy.is_allowed_for_project(self.project, file_path):
            return FileExclusionPolicy.format_llm_exclusion_message([file_path])

        # Check path security before proceeding
        validate_duo_context_exclusions(file_path)

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runEditFile=contract_pb2.EditFile(
                    filepath=file_path,
                    oldString=old_str,
                    newString=new_str,
                )
            ),
        )

    def format_display_message(
        self, args: EditFileInput, _tool_response: Any = None
    ) -> str:
        msg = "Edit file"
        if not FileExclusionPolicy.is_allowed_for_project(self.project, args.file_path):
            msg += FileExclusionPolicy.format_user_exclusion_message([args.file_path])

        return msg


class ListDirInput(BaseModel):
    directory: str = Field(description="Directory path relative to the repository root")


class ListDir(DuoBaseTool):
    name: str = "list_dir"
    description: str = """List directory contents (equivalent to 'ls -la' command).

    **Primary use cases:**
    - See all files and subdirectories in a directory
    - Check if files or directories exist
    - Explore project structure and organization
    - Get directory contents before reading specific files

    **Replaces these commands:**
    - ls -la → list_dir(directory=".")
    - ls -l → list_dir(directory=".")
    - ls → list_dir(directory=".")
    - ls src/ → list_dir(directory="src/")
    - ls -la tests/ → list_dir(directory="tests/")
    - dir → list_dir(directory=".") (Windows equivalent)

    **Examples:**
    - List current directory: list_dir(directory=".")
    - List source code: list_dir(directory="src/")
    - Check if directory exists: list_dir(directory="tests/")
    - Explore subdirectory: list_dir(directory="config/")

    Shows files and subdirectories relative to the repository root.
    Use this instead of trying to run 'ls' commands.
    """
    args_schema: Type[BaseModel] = ListDirInput

    async def _execute(self, directory: str) -> str:
        # Check file exclusion policy before executing action
        if not FileExclusionPolicy.is_allowed_for_project(self.project, directory):
            return FileExclusionPolicy.format_llm_exclusion_message([directory])

        # Check path security before proceeding
        validate_duo_context_exclusions(directory)

        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                listDirectory=contract_pb2.ListDirectory(directory=directory)
            ),
        )

        # Filter results based on file exclusion policy
        policy = FileExclusionPolicy(self.project)
        lines = result.strip().split("\n") if result.strip() else []
        allowed_files, _excluded_files = policy.filter_allowed(lines)

        # Build the response
        response_parts = []
        if allowed_files:
            response_parts.append("\n".join(allowed_files))

        return "\n\n".join(response_parts)


def _format_no_matches_message(pattern, search_directory=None):
    search_scope = f" in '{search_directory}'" if search_directory else ""
    return f"No matches found for pattern '{pattern}'{search_scope}."
