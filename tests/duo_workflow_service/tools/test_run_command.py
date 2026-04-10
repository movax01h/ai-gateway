from typing import List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.command import (
    _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    RunCommand,
    RunCommandInput,
    RunCommandWithTimeout,
    RunCommandWithTimeoutInput,
    ShellCommand,
    ShellCommandWithTimeout,
    ShellCommandWithTimeoutInput,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("program", "args", "expected_action_args"),
    [
        (
            "poetry",
            " run uvicorn  main:app --host 0.0.0.0 --port 8018 ",
            ["run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8018"],
        ),
        (
            "pytest",
            "  tests/test_main.py::test_app_start ",
            ["tests/test_main.py::test_app_start"],
        ),
        (
            "grep",
            '-n "Duo CLI" packages/cli/scripts/test-headless-in-docker/README.md',
            ["-n", "Duo CLI", "packages/cli/scripts/test-headless-in-docker/README.md"],
        ),
        (
            "echo",
            "'single quotes work too'",
            ["single quotes work too"],
        ),
    ],
)
async def test_run_command_success(
    program: str, args: str, expected_action_args: List[str], mock_success_client_event
):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_success_client_event
    )

    metadata = {"outbox": mock_outbox}

    run_command = RunCommand(name="run_command", description="Run a shell command")
    run_command.metadata = metadata

    response = await run_command._arun(program=program, args=args)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.arguments == expected_action_args
    assert action.runCommand.flags == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("program", "args", "should_be_blocked", "expected_message_contains"),
    [
        # Allows git commands in local flows
        ("git", "status", False, None),
        ("git", "", False, None),
        # Disallowed operators in program
        ("echo && ls", "", True, "operators are not supported"),
        ("echo || ls", "", True, "operators are not supported"),
        ("cat | grep", "pattern", True, "operators are not supported"),
        ("ls && git", "", True, "operators are not supported"),
        ("echo 1 || git", "", True, "operators are not supported"),
        ("echo / | xargs rm -rf", "", True, "operators are not supported"),
        # Disallowed operators in args
        ("echo", "foo && bar", True, "operators are not supported"),
        ("echo", "foo || bar", True, "operators are not supported"),
        ("echo", "foo | bar", True, "operators are not supported"),
        # Operators without spaces (tight coupling)
        ("cat|grep", "pattern", True, "operators are not supported"),
        ("ls&&echo", "hello", True, "operators are not supported"),
        ("cmd||fallback", "", True, "operators are not supported"),
        # Allowed cases
        ("echo", "hello world", False, None),
        ("ls", "-la", False, None),
        ("python", "script.py", False, None),
        ("echo", None, False, None),
        ("echo", "", False, None),
        # Malformed quotes (shlex.split raises ValueError)
        ("echo", '"unclosed quote', True, "Invalid argument syntax"),
        ("echo", "it's a trap", True, "Invalid argument syntax"),
    ],
)
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_run_command_validation(
    execute_action_mock, program, args, should_be_blocked, expected_message_contains
):
    run_command = RunCommand(name="run_command", description="Run a shell command")

    result = await run_command._arun(program=program, args=args)

    if should_be_blocked:
        execute_action_mock.assert_not_called()
        if expected_message_contains:
            assert expected_message_contains in result
        assert isinstance(result, str)
    else:
        execute_action_mock.assert_called_once()


def test_run_command_format_display_message():
    tool = RunCommand(description="Run a shell command")

    input_data = RunCommandInput(program="ls", args="-l -a /home ")

    message = tool.format_display_message(input_data)

    expected_message = "Run command: ls -l -a /home"
    assert message == expected_message

    # Format message with tool response
    tool_response = "Exit status 0"
    message = tool.format_display_message(input_data, _tool_response=tool_response)

    expected_message = "Run command: ls -l -a /home Exit status 0"
    assert message == expected_message

    # Format message with long tool response
    tool_response = "Exit status 0"
    message = tool.format_display_message(
        input_data, _tool_response=tool_response, max_len=10
    )

    expected_message = "Run command: ls -l -a /home Exit [...]"
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("program", "args", "timeout", "expected_timeout"),
    [
        ("npm", "install", 300, 300),
        ("docker", "build .", 600, 600),
        (
            "ls",
            "-la",
            _DEFAULT_COMMAND_TIMEOUT_SECONDS,
            _DEFAULT_COMMAND_TIMEOUT_SECONDS,
        ),
    ],
)
async def test_run_command_with_timeout(
    program: str,
    args: str,
    timeout: int,
    expected_timeout: int,
    mock_success_client_event,
):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_success_client_event
    )

    metadata = {"outbox": mock_outbox}

    tool = RunCommandWithTimeout(name="run_command", description="Run a shell command")
    tool.metadata = metadata

    response = await tool._arun(program=program, args=args, timeout=timeout)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.timeout == expected_timeout


def test_run_command_with_timeout_supersedes_run_command():
    assert RunCommandWithTimeout.supersedes is RunCommand
    assert RunCommandWithTimeout.required_capability == frozenset({"command_timeout"})


def test_run_command_with_timeout_format_display_message():
    tool = RunCommandWithTimeout(description="Run a shell command")
    input_data = RunCommandWithTimeoutInput(program="npm", args="install", timeout=300)
    message = tool.format_display_message(input_data)
    assert message == "Run command: npm install"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "timeout", "expected_timeout"),
    [
        ("npm install", 300, 300),
        ("docker build .", 600, 600),
        ("ls -la", _DEFAULT_COMMAND_TIMEOUT_SECONDS, _DEFAULT_COMMAND_TIMEOUT_SECONDS),
    ],
)
async def test_shell_command_with_timeout(
    command: str,
    timeout: int,
    expected_timeout: int,
    mock_success_client_event,
):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_success_client_event
    )

    tool = ShellCommandWithTimeout(
        name="run_command", description="Run a shell command"
    )
    tool.metadata = {"outbox": mock_outbox}

    response = await tool._arun(command=command, timeout=timeout)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runShellCommand.command == command
    assert action.runShellCommand.timeout == expected_timeout


def test_shell_command_with_timeout_supersedes_shell_command():
    assert ShellCommandWithTimeout.supersedes is ShellCommand
    assert ShellCommandWithTimeout.required_capability == frozenset(
        {"shell_command", "command_timeout"}
    )


def test_shell_command_with_timeout_format_display_message():
    tool = ShellCommandWithTimeout(description="Run a shell command")
    input_data = ShellCommandWithTimeoutInput(command="npm install", timeout=300)
    message = tool.format_display_message(input_data)
    assert message == "Run shell command: npm install"


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_run_command_with_timeout_uses_default_when_not_provided(
    execute_action_mock,
):
    """Test that RunCommandWithTimeout uses default timeout when LLM doesn't provide one."""
    execute_action_mock.return_value = "done"

    tool = RunCommandWithTimeout(name="run_command", description="Run a shell command")
    tool.metadata = {"outbox": None}

    # Call without timeout parameter (simulating LLM not providing it)
    await tool._arun(program="ls", args="-la")

    execute_action_mock.assert_called_once()
    action = execute_action_mock.call_args[0][1]
    assert action.runCommand.HasField("timeout")
    assert action.runCommand.timeout == _DEFAULT_COMMAND_TIMEOUT_SECONDS


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_shell_command_with_timeout_uses_default_when_not_provided(
    execute_action_mock,
):
    """Test that ShellCommandWithTimeout uses default timeout when LLM doesn't provide one."""
    execute_action_mock.return_value = "done"

    tool = ShellCommandWithTimeout(
        name="run_command", description="Run a shell command"
    )
    tool.metadata = {"outbox": None}

    # Call without timeout parameter (simulating LLM not providing it)
    await tool._arun(command="ls -la")

    execute_action_mock.assert_called_once()
    action = execute_action_mock.call_args[0][1]
    assert action.runShellCommand.HasField("timeout")
    assert action.runShellCommand.timeout == _DEFAULT_COMMAND_TIMEOUT_SECONDS
