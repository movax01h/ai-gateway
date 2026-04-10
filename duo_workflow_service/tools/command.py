import shlex
import textwrap
from typing import Any, ClassVar, List, Optional, Type

from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import _execute_action
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

_DISALLOWED_COMMANDS: List[str] = []
_DISALLOWED_OPERATORS = ["&&", "||", "|"]
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 120


class RunCommandInput(BaseModel):
    program: str = Field(description="The name of bash program to execute eg: 'cp'")
    args: Optional[str] = Field(
        description="All arguments and flags for the bash program as a single string. "
        "eg: '-v -p source.txt destination.txt'",
        default=None,
    )


class RunCommandWithTimeoutInput(RunCommandInput):
    timeout: int = Field(
        description="Timeout in seconds. "
        "Use a higher value for long-running commands like 'npm install' or 'docker build'.",
        default=_DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )


class RunCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Run a bash command in the current working directory. "
        "This tool should be reserved for cases where specialized tools cannot accomplish the task. "
        f"Following bash commands are not supported: {', '.join(_DISALLOWED_COMMANDS)} "
        "and will result in error. "
        "Pay extra attention to correctly escape special characters like '`'"
    )
    args_schema: Type[BaseModel] = RunCommandInput

    async def _execute(
        self,
        program: str,
        args: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        args = args or ""

        for disallowed_operator in _DISALLOWED_OPERATORS:
            if disallowed_operator in program or disallowed_operator in args:
                # pylint: disable=line-too-long
                return f"""'{disallowed_operator}' operators are not supported with {self.name} tool.
Instead of '{disallowed_operator}' please use {self.name} multiple times consecutively to emulate '{disallowed_operator}' behaviour
"""
        for disallowed_command in _DISALLOWED_COMMANDS:
            if program.startswith(disallowed_command):
                return f"{disallowed_command} commands are not supported with {self.name} tool."

        try:
            arguments = shlex.split(args)
        except ValueError as e:
            return f"Invalid argument syntax: {e}. Check for unclosed quotes."

        run_command_kwargs: dict = {
            "program": program,
            "arguments": arguments,
            "flags": [],
        }
        if timeout is not None:
            run_command_kwargs["timeout"] = timeout

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runCommand=contract_pb2.RunCommandAction(**run_command_kwargs)
            ),
        )

    def format_display_message(
        self,
        args: RunCommandInput,
        _tool_response: Any = None,
        max_len: int = 100,
    ) -> str:
        command = f"{args.program} {args.args}".strip()
        message = f"Run command: {command}"
        if _tool_response is not None:
            return f"{message} {textwrap.shorten(_tool_response, max_len)}"
        return message


class RunCommandWithTimeout(RunCommand):
    """Enhanced run_command tool with optional timeout support.

    Supersedes RunCommand when the client declares the 'command_timeout' capability, ensuring the LLM only receives the
    timeout parameter when the executor can honour it.
    """

    args_schema: Type[BaseModel] = RunCommandWithTimeoutInput
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = RunCommand
    required_capability: ClassVar[frozenset[str]] = frozenset({"command_timeout"})

    async def _execute(
        self,
        program: str,
        args: Optional[str] = None,
        timeout: Optional[int] = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        if timeout is None:
            timeout = _DEFAULT_COMMAND_TIMEOUT_SECONDS
        return await super()._execute(program=program, args=args, timeout=timeout)


class ShellCommandInput(BaseModel):
    command: str = Field(
        description="The shell script to execute in the user's default shell"
    )


class ShellCommandWithTimeoutInput(ShellCommandInput):
    timeout: int = Field(
        description="Timeout in seconds. "
        "Use a higher value for long-running commands like 'npm install' or 'docker build'.",
        default=_DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )


class ShellCommand(DuoBaseTool):
    name: str = "run_command"
    description: str = (
        "Execute a shell command in the current project directory. "
        "Do not prefix command with 'cd' as the working directory is already set."
    )
    args_schema: Type[BaseModel] = ShellCommandInput
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = RunCommand
    required_capability: ClassVar[frozenset[str]] = frozenset({"shell_command"})

    async def _execute(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> str:
        run_shell_kwargs: dict = {"command": command}
        if timeout is not None:
            run_shell_kwargs["timeout"] = timeout

        return await _execute_action(
            self.metadata,  # type: ignore
            contract_pb2.Action(
                runShellCommand=contract_pb2.RunShellCommand(**run_shell_kwargs)
            ),
        )

    def format_display_message(
        self,
        args: ShellCommandInput,
        _tool_response: Any = None,
        max_len: int = 100,
    ) -> str:
        message = f"Run shell command: {args.command}"
        if _tool_response is not None:
            return f"{message} {textwrap.shorten(_tool_response, max_len)}"
        return message


class ShellCommandWithTimeout(ShellCommand):
    """Enhanced shell command tool with timeout support.

    Supersedes ShellCommand when the client declares both 'shell_command' and 'command_timeout' capabilities, ensuring
    the LLM only receives the timeout parameter when the executor can honour it.
    """

    args_schema: Type[BaseModel] = ShellCommandWithTimeoutInput
    supersedes: ClassVar[Optional[Type[DuoBaseTool]]] = ShellCommand
    required_capability: ClassVar[frozenset[str]] = frozenset(
        {"shell_command", "command_timeout"}
    )

    async def _execute(
        self,
        command: str,
        timeout: Optional[int] = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        if timeout is None:
            timeout = _DEFAULT_COMMAND_TIMEOUT_SECONDS
        return await super()._execute(command=command, timeout=timeout)
