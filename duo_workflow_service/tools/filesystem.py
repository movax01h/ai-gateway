from enum import IntEnum
from typing import Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.command import RunCommand
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class ReadFileInput(BaseModel):
    file_path: str = Field(description="the file_path to read the file from")


class ReadFile(DuoBaseTool):
    name: str = "read_file"
    description: str = """Read the contents of a file.

    IMPORTANT:
    - When a task requires reading multiple files, include batches of tool calls in a single response
    - Do not make separate responses for each file - group related files together

    """
    args_schema: Type[BaseModel] = ReadFileInput  # type: ignore

    async def _arun(self, file_path: str) -> str:
        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(runReadFile=contract_pb2.ReadFile(filepath=file_path)),
        )

    def format_display_message(self, args: ReadFileInput) -> str:
        return "Read file"


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
    args_schema: Type[BaseModel] = WriteFileInput  # type: ignore

    async def _arun(self, file_path: str, contents: str) -> str:
        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runWriteFile=contract_pb2.WriteFile(
                    filepath=file_path, contents=contents
                )
            ),
        )

    def format_display_message(self, args: WriteFileInput) -> str:
        return "Create file"


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
    description: str = """Find files, recursively, with names matching a specific pattern in the repository.

It includes all files (tracked and untracked) and respects .gitignore rules.

This name_pattern uses the same syntax as `find --name` or `bash` filename expansion and matches are done against the
full path relative to the project root.
"""
    args_schema: Type[BaseModel] = FindFilesInput  # type: ignore

    async def _arun(
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

        return result

    def format_display_message(self, args: FindFilesInput) -> str:
        return f"Search files with pattern '{args.name_pattern}'"


class MkdirInput(BaseModel):
    directory_path: str = Field(
        description="The directory path to create. Must be within the current working directory tree."
    )


class Mkdir(DuoBaseTool):
    name: str = "mkdir"
    description: str = """Create a new directory using the mkdir command.
    The directory creation is restricted to the current working directory tree."""
    args_schema: Type[BaseModel] = MkdirInput  # type: ignore

    async def _arun(self, directory_path: str) -> str:
        if ".." in directory_path:
            return "Creating directories above the current directory is not allowed"

        if not directory_path.startswith("./") and directory_path != ".":
            directory_path = f"./{directory_path}"

        run_command = RunCommand(
            name="run_command",
            description="Run a shell command",
            metadata=self.metadata,
        )

        return await run_command._arun(
            "mkdir",
            arguments=[directory_path],
            flags=["-p"],  # -p flag creates parent directories as needed
        )

    def format_display_message(self, args: MkdirInput) -> str:
        return f"Create directory '{args.directory_path}'"


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
    # pylint: disable=line-too-long
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
    args_schema: Type[BaseModel] = EditFileInput  # type: ignore

    async def _arun(self, file_path: str, old_str: str, new_str: str) -> str:
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

    def format_display_message(self, args: EditFileInput) -> str:
        return "Edit file"


class ListDirInput(BaseModel):
    directory: str = Field(description="Directory path relative to the repository root")


class ListDir(DuoBaseTool):
    name: str = "list_dir"
    description: str = (
        """Lists files in the given directory relative to the root of the project."""
    )
    args_schema: Type[BaseModel] = ListDirInput  # type: ignore

    async def _arun(self, directory: str) -> str:
        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                listDirectory=contract_pb2.ListDirectory(directory=directory)
            ),
        )


def _format_no_matches_message(pattern, search_directory=None):
    search_scope = f" in '{search_directory}'" if search_directory else ""
    return f"No matches found for pattern '{pattern}'{search_scope}."
