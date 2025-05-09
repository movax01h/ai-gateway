from enum import IntEnum
from typing import Optional, Type

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
    name_pattern: str = Field(
        description=(
            "The pattern to search for files. IMPORTANT: This pattern is delimited by spaces before being passed to git."
            "For complex patterns,you must handle proper escaping"
        )
    )
    files_scope: Optional[FilesScopeEnum] = Field(
        default=FilesScopeEnum.ALL.value,
        description="""
        - 0: (Default): Finds all files matching the pattern. (equivalent to using --cached --others flags together)",
        - 1: Finds only tracked files. (equivalent to using --cached flag only)
        - 2: Finds only untracked files. (equivalent to using --others flag only)
        - 3: Finds only modified files . (equivalent to using --modified flag only)
        - 4: Finds only deleted files (equivalent to using --deleted flag only)
        """,
    )


class FindFiles(DuoBaseTool):
    name: str = "find_files"
    description: str = """Find files matching a specific pattern in the repository

    IMPORTANT: This tool uses git ls-files to recursively find files.
        - The `name_pattern` is delimited by spaces before being passed to git (similar to shell word splitting)
        - Patterns with spaces need proper escaping or quoting
        - The tool always passes `--exclude-standard` flag, so files ignored by git won't be found
        - By default, both tracked and untracked files are included unless you specify otherwise via `files_scope`
    """
    args_schema: Type[BaseModel] = FindFilesInput  # type: ignore

    async def _arun(
        self, name_pattern: str, files_scope: FilesScopeEnum = FilesScopeEnum.ALL
    ) -> str:
        run_git_command = GitCommand(metadata=self.metadata)

        # Always exclude files ignored by git
        ls_files_args = ["--exclude-standard"]

        # Process tracking flags
        match files_scope:
            case FilesScopeEnum.ALL.value:
                ls_files_args.extend(["--cached", "--others"])

            case FilesScopeEnum.TRACKED.value:
                ls_files_args.append("--cached")

            case FilesScopeEnum.UNTRACKED.value:
                ls_files_args.append("--others")

            case FilesScopeEnum.MODIFIED.value:
                ls_files_args.append("--modified")

            case FilesScopeEnum.DELETED.value:
                ls_files_args.append("--deleted")

        if name_pattern:
            ls_files_args.append(name_pattern)

        result = await run_git_command._arun(
            repository_url="",
            command="ls-files",
            args=" ".join(ls_files_args),
        )

        if not result or result.isspace():
            return _format_no_matches_message(name_pattern)

        return result

    def format_display_message(self, args: FindFilesInput) -> str:
        mode = ""
        match args.files_scope:
            case FilesScopeEnum.ALL:
                mode = " (All files)"

            case FilesScopeEnum.TRACKED:
                mode = " (tracked only)"

            case FilesScopeEnum.UNTRACKED:
                mode = " (untracked only)"

            case FilesScopeEnum.MODIFIED:
                mode = " (modified only)"

            case FilesScopeEnum.DELETED:
                mode = " (deleted only)"

        return f"Search files with pattern '{args.name_pattern}'{mode}"


class LsFilesInput(BaseModel):
    directory: str = Field(
        description="The directory to run ls on. Pass `.` for current directory."
    )


class LsFiles(DuoBaseTool):
    name: str = "ls_files"
    description: str = """Lists the contents of a given directory by running the git ls-tree --name-only HEAD:dir command.
          The command lists only git tracked files (cached in Git’s index)."""
    args_schema: Type[BaseModel] = LsFilesInput  # type: ignore

    async def _arun(
        self,
        directory: str,
    ) -> str:
        run_git_command = GitCommand(metadata=self.metadata)

        if not directory.endswith("/"):
            directory += "/"

        return await run_git_command._arun(
            repository_url="",
            command="ls-tree",
            args=f"--name-only HEAD:{directory}",
        )

    def format_display_message(self, args: LsFilesInput) -> str:
        return f"List files in '{args.directory}'"


class GrepInput(BaseModel):
    search_directory: Optional[str] = Field(
        default=None,
        description="The relative path of directory in which to search. Leave blank to search in the current directory.",
    )
    pattern: str = Field(description="The PATTERN to search for")
    recursive: bool = Field(
        default=False,
        description="Search recursively through directories (equivalent to -r flag)",
    )
    case_insensitive: bool = Field(
        default=False,
        description="Ignore case distinctions (equivalent to -i or --ignore-case flag)",
    )
    include_untracked: bool = Field(
        default=False,
        description="Also search in untracked files (equivalent to --untracked flag)",
    )
    files_with_matches: bool = Field(
        default=False,
        description="Show only filenames that contain matches (equivalent to --files-with-matches flag)",
    )
    files_without_match: bool = Field(
        default=False,
        description="Show only filenames that don't contain matches (equivalent to --files-without-match flag)",
    )
    no_recursive: bool = Field(
        default=False,
        description="Don't search recursively (equivalent to --no-recursive flag)",
    )
    fixed_strings: bool = Field(
        default=False,
        description="Interpret patterns as fixed strings, not regular expressions (equivalent to -F flag)",
    )


class Grep(DuoBaseTool):
    name: str = "grep_files"
    description: str = """Search for text patterns in git-tracked files in a directory using the git grep command.
    This tool uses git grep (NOT regular grep) to search through files tracked by git.

    IMPORTANT: By default, git grep only searches tracked files. To include untracked files, use include_untracked=True.

    Examples:
    - Search for "TODO" in all files: grep_files(pattern="TODO")
    - Case-insensitive search: grep_files(pattern="error", case_insensitive=True)
    - Recursive search in subdirectories: grep_files(pattern="test", recursive=True)
    - Non-recursive (current dir only): grep_files(pattern="test", no_recursive=True)
    - Recursive search in specific dir: grep_files(pattern="bug", recursive=True, search_directory="src/")
    - Search only files in current dir: grep_files(pattern="fix", search_directory=".", no_recursive=True)
    - Find files with matches: grep_files(pattern="TODO", files_with_matches=True)
    - Find files without matches: grep_files(pattern="TODO", files_without_match=True)
    - Fixed string pattern (not regex): grep_files(pattern="<!-- tags:", fixed_strings=True)
    - Include untracked files: grep_files(pattern="TODO", include_untracked=True)
    """
    args_schema: Type[BaseModel] = GrepInput  # type: ignore

    # pylint: disable=R0913,R0917
    async def _arun(
        self,
        pattern: str,
        search_directory: Optional[str] = None,
        recursive: bool = False,
        case_insensitive: bool = False,
        include_untracked: bool = False,
        files_with_matches: bool = False,
        files_without_match: bool = False,
        no_recursive: bool = False,
        fixed_strings: bool = False,
    ) -> str:
        """
        Execute the grep command with the specified parameters.

        This method has many parameters to support various grep options.
        """
        if search_directory and ".." in search_directory:
            return "Searching above the current directory is not allowed"

        run_git_command = GitCommand(metadata=self.metadata)

        grep_args = []

        # Add all the boolean flags to the command
        if recursive:
            grep_args.append("-r")

        if case_insensitive:
            grep_args.append("-i")

        if include_untracked:
            grep_args.append("--untracked")

        if files_with_matches:
            grep_args.append("--files-with-matches")

        if files_without_match:
            grep_args.append("--files-without-match")

        if no_recursive:
            grep_args.append("--no-recursive")

        if fixed_strings:
            grep_args.append("-F")

        grep_args.append(pattern)

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


def _format_no_matches_message(pattern, search_directory=None):
    search_scope = f" in '{search_directory}'" if search_directory else ""
    return f"No matches found for pattern '{pattern}'{search_scope}."
