# pylint: disable=file-naming-for-tests
from typing import List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.tools import ToolException

from duo_workflow_service.tools.command import (
    _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    RunCommand,
    RunCommandInput,
    ShellCommand,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("program", "args"),
    [
        ("echo", '"unclosed quote'),
        ("echo", "it's a trap"),
    ],
)
async def test_run_command_malformed_quotes(program, args):
    """Malformed quotes should raise ToolException."""
    run_command = RunCommand(name="run_command", description="Run a shell command")

    with pytest.raises(ToolException) as exc_info:
        await run_command._arun(program=program, args=args)
    assert "Invalid argument syntax" in str(exc_info.value)


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
        ("ls", "-la", None, _DEFAULT_COMMAND_TIMEOUT_SECONDS),
    ],
)
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=True)
async def test_run_command_with_timeout_capable(
    _mock_is_client_capable,
    program: str,
    args: str,
    timeout,
    expected_timeout: int,
    mock_success_client_event,
):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_success_client_event
    )

    tool = RunCommand(name="run_command", description="Run a shell command")
    tool.metadata = {"outbox": mock_outbox}

    response = await tool._arun(program=program, args=args, timeout=timeout)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.timeout == expected_timeout


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=False)
async def test_run_command_with_timeout_not_capable(_mock_is_client_capable):
    """Providing timeout when client lacks command_timeout capability raises ToolException."""
    tool = RunCommand(name="run_command", description="Run a shell command")

    with pytest.raises(ToolException, match="not supported by this client version"):
        await tool._arun(program="npm", args="install", timeout=300)


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=False)
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_run_command_without_timeout_not_capable_skips_timeout(
    execute_action_mock, _mock_is_client_capable
):
    """When client is not capable and no timeout provided, command runs without timeout."""
    execute_action_mock.return_value = "done"

    tool = RunCommand(name="run_command", description="Run a shell command")
    tool.metadata = {"outbox": None}

    await tool._arun(program="ls", args="-la")

    execute_action_mock.assert_called_once()
    action = execute_action_mock.call_args[0][1]
    assert not action.runCommand.HasField("timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "timeout", "expected_timeout"),
    [
        ("npm install", 300, 300),
        ("docker build .", 600, 600),
        ("ls -la", None, _DEFAULT_COMMAND_TIMEOUT_SECONDS),
    ],
)
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=True)
async def test_shell_command_with_timeout_capable(
    _mock_is_client_capable,
    command: str,
    timeout,
    expected_timeout: int,
    mock_success_client_event,
):
    mock_outbox = MagicMock()
    mock_outbox.put_action_and_wait_for_response = AsyncMock(
        return_value=mock_success_client_event
    )

    tool = ShellCommand(name="run_command", description="Run a shell command")
    tool.metadata = {"outbox": mock_outbox}

    response = await tool._arun(command=command, timeout=timeout)

    assert response == "done"

    mock_outbox.put_action_and_wait_for_response.assert_called_once()
    action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
    assert action.runShellCommand.command == command
    assert action.runShellCommand.timeout == expected_timeout


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=False)
async def test_shell_command_with_timeout_not_capable(_mock_is_client_capable):
    """Providing timeout when client lacks command_timeout capability raises ToolException."""
    tool = ShellCommand(name="run_command", description="Run a shell command")

    with pytest.raises(ToolException, match="not supported by this client version"):
        await tool._arun(command="npm install", timeout=300)


@pytest.mark.asyncio
@mock.patch("duo_workflow_service.tools.command.is_client_capable", return_value=False)
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_shell_command_without_timeout_not_capable_skips_timeout(
    execute_action_mock, _mock_is_client_capable
):
    """When client is not capable and no timeout provided, command runs without timeout."""
    execute_action_mock.return_value = "done"

    tool = ShellCommand(name="run_command", description="Run a shell command")
    tool.metadata = {"outbox": None}

    await tool._arun(command="ls -la")

    execute_action_mock.assert_called_once()
    action = execute_action_mock.call_args[0][1]
    assert not action.runShellCommand.HasField("timeout")
