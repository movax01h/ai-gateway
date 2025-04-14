from typing import Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class ReadOnlyGitInput(BaseModel):
    repository_url: str = Field(description="Http git remote url")
    command: str = Field(description="Git command to run")
    args: Optional[str] = Field(
        description="Git command arguments, leave empty if none", default=None
    )


class ReadOnlyGit(DuoBaseTool):
    name: str = "run_read_only_git_command"
    description: str = """Runs a read-only git command in the repository working directory.
           The command doesn't modify the repository and is safe to use without
           worry about changing any data.
        """
    args_schema: Type[BaseModel] = ReadOnlyGitInput

    async def _arun(
        self, repository_url: str, command: str, args: Optional[str] = None
    ) -> str:
        return await _execute_action(
            self.metadata,  # type: ignore[arg-type]
            contract_pb2.Action(
                runGitCommand=contract_pb2.RunGitCommand(
                    command=command, arguments=args, repository_url=repository_url
                )
            ),
        )

    def format_display_message(self, git_command_args: ReadOnlyGitInput) -> str:
        return f"Running git command: git {git_command_args.command} {git_command_args.args} in repository"
