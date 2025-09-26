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
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(return_value=mock_success_client_event)

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

    run_command = RunCommand(name="run_command", description="Run a shell command")
    run_command.metadata = metadata

    response = await run_command._arun(program=program, args=args)

    assert response == "done"

    mock_outbox.put.assert_called_once()
    action = mock_outbox.put.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.arguments == expected_action_args
    assert action.runCommand.flags == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "program",
    [
        "git",
        "ls && git",
        "echo 1 || git",
        "echo / | xargs rm -rf",
    ],
)
@mock.patch("duo_workflow_service.tools.command._execute_action")
async def test_run_disallowed_command(execute_action_mock, program):
    run_command = RunCommand(name="run_command", description="Run a shell command")

    await run_command._arun(program=program, args="")

    execute_action_mock.assert_not_called()


def test_run_command_format_display_message():
    tool = RunCommand(description="Run a shell command")

    input_data = RunCommandInput(program="ls", args="-l -a /home ")

    message = tool.format_display_message(input_data)

    expected_message = "Run command: ls -l -a /home"
    assert message == expected_message
