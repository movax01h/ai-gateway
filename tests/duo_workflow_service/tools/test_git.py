from unittest.mock import call, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.git import Command, GitCommandInput


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.git._execute_action")
async def test_git_clone_success(execute_action_mock):
    metadata = {"key": 1}
    git_pull = Command(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "done"
    repository_url = "git@gdk.test:2222/duo-workflow-test/ai-assist.git"

    response = await git_pull._arun(
        command="add", repository_url=repository_url, args="."
    )

    assert response == "done"
    execute_action_mock.assert_has_calls(
        [
            call(
                metadata,
                contract_pb2.Action(
                    runGitCommand=contract_pb2.RunGitCommand(
                        command="add", repository_url=repository_url, arguments="."
                    )
                ),
            )
        ]
    )


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
