from unittest.mock import call, patch

import pytest
from langchain_core.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.tools.git import Command, GitCommandInput


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_emits_run_command_action(execute_action_mock):
    """run_git_command shim must emit a runCommand action with program="git"."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "done"
    repository_url = "git@gdk.test:2222/duo-workflow-test/ai-assist.git"

    response = await tool._arun(command="add", repository_url=repository_url, args=".")

    assert response == "done"
    execute_action_mock.assert_has_calls(
        [
            call(
                metadata,
                contract_pb2.Action(
                    runCommand=contract_pb2.RunCommandAction(
                        program="git", arguments=["add", "."], flags=[]
                    )
                ),
            )
        ]
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_strips_leading_git_token(execute_action_mock):
    """If the model passes 'git commit ...' as the command, strip the leading 'git'."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "ok"

    await tool._arun(command="git commit", repository_url="", args="-m 'fix'")

    execute_action_mock.assert_called_once_with(
        metadata,
        contract_pb2.Action(
            runCommand=contract_pb2.RunCommandAction(
                program="git", arguments=["commit", "-m", "fix"], flags=[]
            )
        ),
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_quoted_args_stay_one_element(execute_action_mock):
    """Quoted args with spaces must be kept as a single argv element (shlex semantics)."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "ok"

    await tool._arun(command="commit", repository_url="", args='-m "msg with spaces"')

    execute_action_mock.assert_called_once_with(
        metadata,
        contract_pb2.Action(
            runCommand=contract_pb2.RunCommandAction(
                program="git",
                arguments=["commit", "-m", "msg with spaces"],
                flags=[],
            )
        ),
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_no_args(execute_action_mock):
    """When args is None only the command tokens are passed as arguments."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "ok"

    await tool._arun(command="status", repository_url="")

    execute_action_mock.assert_called_once_with(
        metadata,
        contract_pb2.Action(
            runCommand=contract_pb2.RunCommandAction(
                program="git", arguments=["status"], flags=[]
            )
        ),
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_repository_url_ignored(execute_action_mock):
    """repository_url is intentionally ignored; the executor uses cwd."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "ok"

    await tool._arun(
        command="push",
        repository_url="https://gitlab.com/some/repo.git",
        args="origin main",
    )

    action = execute_action_mock.call_args[0][1]
    # The action must be a runCommand, not a runGitCommand
    assert action.HasField("runCommand")
    assert not action.HasField("runGitCommand")
    assert action.runCommand.program == "git"
    assert list(action.runCommand.arguments) == ["push", "origin", "main"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,args",
    [
        ("commit -m 'unterminated", None),
        ("commit", "-m 'unterminated"),
    ],
)
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_command_malformed_shlex_input_raises_tool_exception(
    execute_action_mock, command, args
):
    """Malformed shlex input (e.g. unterminated quote) must raise ToolException with a readable message instead of
    propagating a raw ValueError."""
    metadata = {"key": 1}
    tool = Command(metadata=metadata, description="Git command execution")

    with pytest.raises(ToolException, match="could not parse git command arguments"):
        await tool._execute(command=command, repository_url="", args=args)

    execute_action_mock.assert_not_called()


def test_run_command_not_implemented_error():
    run_command = Command(description="Test command execution")

    with pytest.raises(NotImplementedError):
        run_command._run("echo Hello")


def test_git_command_format_display_message():
    tool = Command(description="Git command execution")

    input_data = GitCommandInput(
        repository_url="git@gdk.test:2222/duo-workflow-test/ai-assist.git",
        command="pull",
        args="origin main",
    )

    message = tool.format_display_message(input_data)

    expected_message = "Run git command: git pull origin main in repository"
    assert message == expected_message
