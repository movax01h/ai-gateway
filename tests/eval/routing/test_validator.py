import datetime
import uuid
from unittest.mock import Mock

import pytest
from langsmith.schemas import Example, Run

from eval.routing.validator import execute_routing, is_correct


def test_execute_routing(monkeypatch):
    mock_chat = Mock()
    mock_llm = Mock()
    mock_chat.bind_tools.return_value = mock_llm
    monkeypatch.setattr("eval.routing.validator.CHAT", mock_chat)

    mock_response = {"response": "test"}
    mock_llm.invoke.return_value = mock_response

    inputs = {"tools": ["tool1", "tool2"], "messages": ["message1", "message2"]}

    result = execute_routing(inputs)

    mock_chat.bind_tools.assert_called_once_with(["tool1", "tool2"])
    mock_llm.invoke.assert_called_once_with(["message1", "message2"])
    assert result == mock_response


@pytest.mark.parametrize(
    ("run_outputs", "example_outputs", "score"),
    [
        ({}, {"tool": {"name": "test_tool", "args": {"key": "value"}}}, 0.0),
        (
            {"tool_calls": [{"name": "wrong_tool", "args": {}}]},
            {"tool": {"name": "test_tool", "args": {"key": "value"}}},
            0.0,
        ),
        (
            {"tool_calls": [{"name": "test_tool", "args": {}}]},
            {
                "tool": {
                    "name": "test_tool",
                    "args": {"required_arg": "test"},
                }
            },
            0.0,
        ),
        (
            {"tool_calls": [{"name": "test_tool", "args": {"test_arg": 5}}]},
            {
                "tool": {
                    "name": "test_tool",
                    "args": {"test_arg": 10},
                }
            },
            0.0,
        ),
        (
            {
                "tool_calls": [
                    {"name": "test_tool", "args": {"arg1": 10, "arg2": "hello"}}
                ]
            },
            {
                "tool": {
                    "name": "test_tool",
                    "args": {"arg1": 10, "arg2": "hello"},
                }
            },
            1.0,
        ),
    ],
)
def test_is_correct(run_outputs: dict, example_outputs: dict, score: float):
    run = Run(
        outputs=run_outputs,
        id=uuid.uuid4(),
        name="test",
        start_time=datetime.datetime.now(),
        run_type="llm",
    )
    example = Example(outputs=example_outputs, id=uuid.uuid4())

    result = is_correct(run, example)

    assert result.key == "is_correct"
    assert result.score == score
