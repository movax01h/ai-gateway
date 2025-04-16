from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.command import RunCommand, RunCommandInput


@pytest.mark.asyncio
async def test_run_command_success():
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="done")
        )
    )

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

    run_command = RunCommand(name="run_command", description="Run a shell command")
    run_command.metadata = metadata

    program = "echo"
    args = ["Hello"]
    flags: list[str] = []
    expected_response = "done"

    response = await run_command._arun(program=program, arguments=args, flags=flags)

    assert response == expected_response

    mock_outbox.put.assert_called_once()
    action = mock_outbox.put.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.arguments == args
    assert action.runCommand.flags == flags


@pytest.mark.asyncio
async def test_run_command_with_flags_success():
    mock_outbox = MagicMock()
    mock_outbox.put = AsyncMock()

    mock_inbox = MagicMock()
    mock_inbox.get = AsyncMock(
        return_value=contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(response="done")
        )
    )

    metadata = {"outbox": mock_outbox, "inbox": mock_inbox}

    run_command = RunCommand(name="run_command", description="Run a shell command")
    run_command.metadata = metadata

    program = "ls"
    args = ["/home"]
    flags = ["-l", "-a"]
    expected_response = "done"

    response = await run_command._arun(program=program, arguments=args, flags=flags)

    assert response == expected_response

    mock_outbox.put.assert_called_once()
    action = mock_outbox.put.call_args[0][0]
    assert action.runCommand.program == program
    assert action.runCommand.arguments == args
    assert action.runCommand.flags == flags


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
async def test_run_git_command(execute_action_mock, program):
    run_command = RunCommand(name="run_command", description="Run a shell command")

    await run_command._arun(program=program)

    execute_action_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_command_not_implemented_error():
    run_command = RunCommand(name="run_command", description="Run a shell command")

    with pytest.raises(NotImplementedError):
        run_command._run("", [], [])


def test_run_command_format_display_message():
    tool = RunCommand(description="Run a shell command")

    input_data = RunCommandInput(program="ls", arguments=["/home"], flags=["-l", "-a"])

    message = tool.format_display_message(input_data)

    expected_message = "Run command: ls -l -a /home"
    assert message == expected_message
