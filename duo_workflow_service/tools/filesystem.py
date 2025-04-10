import os
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.command import RunCommand
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.git import Command as GitCommand


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
        "", description="the contents to write in the file. *This is required*"
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


class FindFilesInput(BaseModel):
    directory: str = Field(description="Always pass .")
    name_pattern: str = Field(
        description="The wildcard pattern to search for, e.g. '**/*.py' for all Python files."
    )
    flags: Optional[list[str]] = Field(
        None,
        description=(
            "The options to pass to the git ls-files command, e.g. ['--others'] to search for only untracked files."
            "All valid options for git ls-files can be used."
        ),
    )


class FindFiles(DuoBaseTool):
    name: str = "find_files"
    description: str = """Find files matching a specific pattern in the repository.

    IMPORTANT: By default, this tool uses git ls-files to find both tracked and untracked files
    This tools always passes `--exclude-standard` flag to git ls-files, thus does not have access
    to files ignored by git.

    Examples:
    - Find all Python files (both tracked and untracked) recursively: find_files(name_pattern="**/*.py")
    - Find a specific file with path (whether tracked or not): find_files(name_pattern="path/to/file.txt")
    - Find only tracked Python files in current directory: find_files(name_pattern="*.py", flags=["--cached"])
    """
    args_schema: Type[BaseModel] = FindFilesInput  # type: ignore

    async def _arun(
        self, directory: str, name_pattern: str, flags: Optional[list[str]] = None
    ) -> str:
        run_git_command = GitCommand(metadata=self.metadata)

        # Always exclude files ignored by git
        ls_files_args = ["--exclude-standard"]

        user_flags = flags or []
        ls_files_args.extend(user_flags)

        tracking_flags = ["--cached", "--others"]
        if not any(flag in user_flags for flag in tracking_flags):
            # By default, include both tracked and untracked files unless
            # a specific one is specified.
            ls_files_args.extend(tracking_flags)

        if name_pattern:
            ls_files_args.append(ensure_pattern_is_quoted(name_pattern))

        return await run_git_command._arun(
            repository_url="",
            command="ls-files",
            args=" ".join(ls_files_args),
        )

    def format_display_message(self, args: FindFilesInput) -> str:
        return f"Search files in '{args.directory}' with pattern '{args.name_pattern}'"


class LsFilesInput(BaseModel):
    directory: str = Field(
        description="The directory to run ls on. Pass `.` for current directory."
    )


class LsFiles(DuoBaseTool):
    name: str = "ls_files"
    description: str = "Run ls -a on a specific directory."
    args_schema: Type[BaseModel] = LsFilesInput  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (
            os.environ.get("FEATURE_GIT_LS_TREE_INSTEAD_OF_LS", "False").lower()
            == "true"
        ):
            self.description = """Lists the contents of a given directory by running the git ls-tree --name-only HEAD:dir command.
          The command lists only git tracked files (cached in Git’s index)."""

    async def _arun(
        self,
        directory: str,
    ) -> str:
        if (
            os.environ.get("FEATURE_GIT_LS_TREE_INSTEAD_OF_LS", "False").lower()
            == "true"
        ):
            run_git_command = GitCommand(metadata=self.metadata)

            if not directory.endswith("/"):
                directory += "/"

            return await run_git_command._arun(
                repository_url="",
                command="ls-tree",
                args=f"--name-only HEAD:{directory}",
            )

        run_command = RunCommand(
            name="run_command",
            description="Run a shell command",
            metadata=self.metadata,
        )

        return await run_command._arun("ls", arguments=[directory], flags=["-a"])

    def format_display_message(self, args: LsFilesInput) -> str:
        return f"List files in '{args.directory}'"


class GrepInput(BaseModel):
    search_directory: Optional[str] = Field(
        None,
        description="The relative path of directory in which to search. Leave blank to search in the current directory.",
    )
    flags: Optional[List[str]] = Field(
        None,
        description="Options to apply to the grep command. Standard git grep options are supported."
        "Possible values are (but not limited to) ['-r', '-i', '--untracked'].",
    )
    pattern: str = Field(description="The PATTERN to search for")


class Grep(DuoBaseTool):
    name: str = "grep_files"
    description: str = """Search for text patterns in git-tracked files in a directory using the git grep command.
    This tool uses git grep (NOT regular grep) to search through files tracked by git.

    IMPORTANT: By default, git grep only searches tracked files. To include untracked files, use the --untracked argument:
    - Default (tracked only): git_grep(pattern="TODO")
    - Include untracked: git_grep(pattern="TODO", flags=["--untracked"])

    Examples:
    - Search for "TODO" in all files: git_grep(pattern="TODO")
    - Case-insensitive search: git_grep(pattern="error", flags=["-i"])
    - Recursive search in subdirectories: git_grep(pattern="test", flags=["-r"])
    - Non-recursive (current dir only): git_grep(pattern="test", flags=["--no-recursive"])
    - Recursive search in specific dir: git_grep(pattern="bug", flags=["-r"], directory="src/")
    - Search only files in current dir: git_grep(pattern="fix", directory=".", flags=["--no-recursive"])
    - Complex pattern: git_grep(pattern='"<!-- tags:"', flags=["-F"]
    """
    args_schema: Type[BaseModel] = GrepInput  # type: ignore

    # pylint: disable=too-many-positional-arguments
    async def _arun(
        self,
        pattern: str,
        search_directory: Optional[str] = None,
        flags: Optional[List[str]] = None,
    ) -> str:
        if search_directory and ".." in search_directory:
            return "Searching above the current directory is not allowed"

        run_git_command = GitCommand(metadata=self.metadata)

        grep_args = []
        if flags:
            grep_args.extend(flags)

        grep_args.append(ensure_pattern_is_quoted(pattern))

        if search_directory:
            grep_args.append("--")
            grep_args.append(search_directory)

        result = await run_git_command._arun(
            repository_url="",
            command="grep",
            args=" ".join(grep_args),
        )

        if result == "Error running tool: exit status 1":
            return _format_no_matches_message(pattern, search_directory)

        return result

    # pylint: enable=too-many-positional-arguments

    def format_display_message(self, args: GrepInput) -> str:
        if args.search_directory is None:
            message = f"Search for '{args.pattern}' in directory"
        else:
            message = (
                f"Search for '{args.pattern}' in files in '{args.search_directory}'"
            )
        return message


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
        description="The string to replace. Please provide at least one line above and below to make it unique across the file. *This is required*",
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


def ensure_pattern_is_quoted(pattern: str) -> str:
    if pattern.startswith("'") and pattern.endswith("'"):
        return pattern

    return f"'{pattern}'"


def _format_no_matches_message(pattern, search_directory=None):
    search_scope = f" in '{search_directory}'" if search_directory else ""
    return (
        f"No matches found for pattern '{pattern}'{search_scope} in the searched files."
    )
