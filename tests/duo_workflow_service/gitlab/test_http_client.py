from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contract import contract_pb2
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, checkpoint_decoder


@pytest.fixture
def mock_execute_action():
    return AsyncMock()


@pytest.fixture
def client():
    return GitlabHttpClient({})


@pytest.fixture
def monkeypatch_execute_action(monkeypatch, mock_execute_action):
    monkeypatch.setattr(
        "duo_workflow_service.gitlab.http_client._execute_action",
        mock_execute_action,
    )
    return mock_execute_action


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, body, parse_json, mock_return_value, expected_result",
    [
        ("GET", "/api/v4/projects/1", None, True, '{"key": "value"}', {"key": "value"}),
        (
            "GET",
            "/api/v4/projects/1/jobs/102/trace",
            None,
            False,
            "Non-JSON response",
            "Non-JSON response",
        ),
        (
            "POST",
            "/api/v4/test",
            '{ "test": 1 }',
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "POST",
            "/api/v4/test",
            '{ "test": 1 }',
            False,
            "Non-JSON response",
            "Non-JSON response",
        ),
        (
            "PUT",
            "/api/v4/test",
            '{ "test": 1 }',
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "PUT",
            "/api/v4/test",
            '{ "test": 1 }',
            False,
            "Non-JSON response",
            "Non-JSON response",
        ),
        (
            "GET",
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints",
            None,
            True,
            """
            {
              "id": 1,
              "checkpoint": {
                "v": 1,
                "id": "checkpoint_id",
                "channel_values": {
                  "conversation_history": {
                    "planner": [{
                      "type": "SystemMessage",
                      "content": "You are an AI planner.",
                      "additional_kwargs": {},
                      "response_metadata": {},
                      "name": null,
                      "id": null
                    }]
                  },
                  "status": "Completed"
                }
              },
              "metadata": {}
            }
            """,
            {
                "id": 1,
                "checkpoint": {
                    "v": 1,
                    "id": "checkpoint_id",
                    "channel_values": {
                        "conversation_history": {
                            "planner": [
                                {
                                    "type": "SystemMessage",
                                    "content": "You are an AI planner.",
                                    "additional_kwargs": {},
                                    "response_metadata": {},
                                    "name": None,
                                    "id": None,
                                }
                            ],
                        },
                        "status": "Completed",
                    },
                },
                "metadata": {},
            },
        ),
        (
            "PATCH",
            "/api/v4/test",
            '{ "test": 1 }',
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
    ],
)
async def test_gitlab_http_client_methods(
    client,
    monkeypatch_execute_action,
    method,
    path,
    body,
    parse_json,
    mock_return_value,
    expected_result,
):
    monkeypatch_execute_action.return_value = mock_return_value

    if method == "GET":
        result = await client.aget(path, parse_json=parse_json)
    elif method == "POST":
        result = await client.apost(path, body, parse_json=parse_json)
    elif method == "PUT":
        result = await client.aput(path, body, parse_json=parse_json)
    elif method == "PATCH":
        result = await client.apatch(path, body, parse_json=parse_json)
    else:
        pytest.fail(f"Unexpected HTTP method: {method}")
        result = None

    expected_action = contract_pb2.Action(
        runHTTPRequest=contract_pb2.RunHTTPRequest(path=path, method=method, body=body)
    )
    monkeypatch_execute_action.assert_called_once_with({}, expected_action)

    assert result == expected_result


@pytest.mark.asyncio
async def test_gitlab_http_client_checkpoint_decoding(
    client,
    monkeypatch_execute_action,
):
    json_checkpoint = """
      {
        "id": 1,
        "checkpoint": {
          "v": 1,
          "id": "checkpoint_id",
          "channel_values": {
            "conversation_history": {
              "planner": [{
                "type": "SystemMessage",
                "content": "You are an AI planner.",
                "additional_kwargs": {},
                "response_metadata": {},
                "name": null,
                "id": null
              }, {
                "type": "HumanMessage",
                "content": "Human question.",
                "additional_kwargs": {},
                "response_metadata": {},
                "name": null,
                "id": null
              }, {
                "type": "AIMessage",
                "content": "AI response.",
                "additional_kwargs": {},
                "response_metadata": {},
                "name": null,
                "id": null,
                "tool_calls": [
                  {
                    "id": "toolu_01GmisLdYSy7LTP1fB7nm5HH",
                    "args": {
                      "pattern": "(teh|alos|wiht|reciev|seperat|accomodat)",
                      "ignore_case": true,
                      "search_directory": "doc"
                    },
                    "name": "grep_files",
                    "type": "tool_call"
                  }
                ]
              }, {
                "id": null,
                "name": null,
                "type": "ToolMessage",
                "status": "success",
                "content": "Error running tool: exit status 1",
                "tool_call_id": "toolu_01GmisLdYSy7LTP1fB7nm5HH",
                "additional_kwargs": {},
                "response_metadata": {}
              }, {
                "type": "unknown",
                "content": "unknown type"
              }]
            },
            "status": "Completed"
          }
        },
        "metadata": {}
      }
    """
    expected_result = {
        "id": 1,
        "checkpoint": {
            "v": 1,
            "id": "checkpoint_id",
            "channel_values": {
                "conversation_history": {
                    "planner": [
                        SystemMessage(content="You are an AI planner."),
                        HumanMessage(content="Human question."),
                        AIMessage(
                            content="AI response.",
                            tool_calls=[
                                {
                                    "name": "grep_files",
                                    "args": {
                                        "pattern": "(teh|alos|wiht|reciev|seperat|accomodat)",
                                        "ignore_case": True,
                                        "search_directory": "doc",
                                    },
                                    "id": "toolu_01GmisLdYSy7LTP1fB7nm5HH",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        ToolMessage(
                            content="Error running tool: exit status 1",
                            tool_call_id="toolu_01GmisLdYSy7LTP1fB7nm5HH",
                        ),
                        {"type": "unknown", "content": "unknown type"},
                    ],
                },
                "status": "Completed",
            },
        },
        "metadata": {},
    }
    monkeypatch_execute_action.return_value = json_checkpoint

    result = await client.aget(
        "/api/v4/ai/duo_workflows/workflows/1/checkpoints",
        parse_json=True,
        object_hook=checkpoint_decoder,
    )
    assert result == expected_result
