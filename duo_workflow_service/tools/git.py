import shlex
from typing import Any, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class GitCommandInput(BaseModel):
    repository_url: str = Field(description="Http git remote url")
    command: str = Field(description="Git command to run")
    args: Optional[str] = Field(
        description="Git command arguments, leave empty if none", default=None
    )


class Command(DuoBaseTool):
    name: str = "run_git_command"
    description: str = """Runs a git command in the repository working directory.

    DEPRECATED: this tool is now a thin shim over run_command so all git runs through
    one execution path that inherits the runner's git config (credential.helper,
    http.<url>.proactiveAuth, safe.directory, core.hooksPath trailers). Prefer
    run_command with program="git" for new flows."""
    args_schema: Type[BaseModel] = GitCommandInput

    async def _execute(
        self, repository_url: str, command: str, args: Optional[str] = None
    ) -> str:
        # DEPRECATED: run_git_command is now a thin shim over run_command so all git
        # runs through one execution path that inherits the runner's git config
        # (credential.helper, http.<url>.proactiveAuth, safe.directory,
        # core.hooksPath trailers). `repository_url` is intentionally ignored: the
        # executor runs git in the checkout cwd (matches the pre-existing
        # run_git_command single-repo limitation).
        try:
            argv = shlex.split(command) + (shlex.split(args) if args else [])
        except ValueError as e:
            raise ToolException(
                f"Error: could not parse git command arguments: {e}"
            ) from e
        if argv and argv[0] == "git":  # tolerate models that pass a leading "git"
            argv = argv[1:]
        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runCommand=contract_pb2.RunCommandAction(
                    program="git", arguments=argv, flags=[]
                )
            ),
        )

    def format_display_message(
        self, git_command_args: GitCommandInput, _tool_response: Any = None
    ) -> str:
        return f"Run git command: git {git_command_args.command} {git_command_args.args} in repository"
