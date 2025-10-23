from typing import List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.command import RunCommand, RunCommandInput


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
        # Disallowed commands
        ("git", "status", True, "git commands are not supported"),
        ("git", "", True, "git commands are not supported"),
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


def test_run_command_format_display_message():
    tool = RunCommand(description="Run a shell command")

    input_data = RunCommandInput(program="ls", args="-l -a /home ")

    message = tool.format_display_message(input_data)

    expected_message = "Run command: ls -l -a /home"
    assert message == expected_message
