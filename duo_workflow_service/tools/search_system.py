from typing import Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.filesystem import _format_no_matches_message


class GrepInput(BaseModel):
    search_directory: Optional[str] = Field(
        default=".",
        description="The relative path of directory in which to search. Defaults to current directory.",
    )
    pattern: str = Field(description="The PATTERN to search for")
    case_insensitive: bool = Field(
        default=False,
        description="Ignore case distinctions (equivalent to -i flag)",
    )


class Grep(DuoBaseTool):
    name: str = "grep"
    description: str = """Search for text patterns in files.
    This tool uses searches, recursively, through all files in the given directory, respecting .gitignore rules.

    IMPORTANT: This tool automatically:
    - Includes all files (tracked and untracked)
    - Respects .gitignore rules
    - Searches recursively by default

    Examples:
    - Search for "TODO" in all files: grep(pattern="TODO")
    - Case-insensitive search: grep(pattern="error", case_insensitive=True)
    - Search in specific directory: grep(pattern="bug", search_directory="src/")
    """
    args_schema: Type[BaseModel] = GrepInput  # type: ignore

    async def _arun(
        self,
        pattern: str,
        search_directory: str = ".",
        case_insensitive: bool = False,
    ) -> str:
        """
        Execute the standard grep command with the specified parameters.
        """
        if search_directory and ".." in search_directory:
            return "Searching above the current directory is not allowed"

        result = await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                grep=contract_pb2.Grep(
                    pattern=pattern,
                    search_directory=search_directory,
                    case_insensitive=case_insensitive,
                )
            ),
        )

        if "No such file or directory" in result or "exit status 1" in result:
            return _format_no_matches_message(pattern, search_directory)

        return result

    def format_display_message(self, args: GrepInput) -> str:
        if not (search_dir := args.search_directory):
            search_dir = "directory"
        message = f"Search for '{args.pattern}' in files in '{search_dir}'"
        return message
