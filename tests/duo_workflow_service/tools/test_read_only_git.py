from unittest.mock import call, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.tools.read_only_git import ReadOnlyGit, ReadOnlyGitInput


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.read_only_git._execute_action")
async def test_read_only_git_success(execute_action_mock):
    metadata = {"key": 1}
    tool = ReadOnlyGit(metadata=metadata, description="Git command execution")
    execute_action_mock.return_value = "done"
    repository_url = "git@gdk.test:2222/duo-workflow-test/ai-assist.git"

    response = await tool._arun(command="log", repository_url=repository_url, args="")

    assert response == "done"
    execute_action_mock.assert_has_calls(
        [
            call(
                metadata,
                contract_pb2.Action(
                    runGitCommand=contract_pb2.RunGitCommand(
                        command="log", repository_url=repository_url, arguments=""
                    )
                ),
            )
        ]
    )


def test_not_implemented_error():
    tool = ReadOnlyGit(description="Test command execution")

    with pytest.raises(NotImplementedError):
        tool._run("echo Hello")


def test_read_only_git_format_display_message():
    tool = ReadOnlyGit(description="Run read-only git command")

    input_data = ReadOnlyGitInput(
        repository_url="https://example.com/repo.git", command="log", args="--oneline"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Running git command: git log --oneline in repository"
    assert message == expected_message
