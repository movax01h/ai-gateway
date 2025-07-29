import json
from datetime import datetime
from typing import AsyncIterator, Type
from unittest.mock import Mock, PropertyMock, call, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from pydantic import AnyUrl
from starlette.testclient import TestClient

from ai_gateway.api.v2 import api_router
from ai_gateway.api.v2.chat.typing import (
    AgentRequest,
    AgentRequestOptions,
    ReActAgentScratchpad,
)
from ai_gateway.chat.agents import (
    AdditionalContext,
    AgentBaseEvent,
    AgentStep,
    AgentToolAction,
    CurrentFile,
    Message,
    ReActAgentInputs,
)
from ai_gateway.chat.agents.typing import AgentFinalAnswer, TypeAgentEvent
from ai_gateway.chat.context.current_page import Context, MergeRequestContext
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)
from ai_gateway.models.base_chat import Role
from ai_gateway.prompts.typing import Model


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(scopes=["duo_chat", "amazon_q_integration"]),
    )


@pytest.fixture(name="mock_create_event_stream")
def mock_create_event_stream_fixture():
    async def _dummy_create_event_stream(*args, **kwargs):
        react_inputs = ReActAgentInputs(
            messages=(
                kwargs.get("agent_request").messages
                if kwargs.get("agent_request")
                else []
            ),
            agent_scratchpad=kwargs.get("agent_scratchpad", []),
            unavailable_resources=(
                kwargs.get("agent_request").unavailable_resources
                if kwargs.get("agent_request")
                else []
            ),
            current_date=kwargs.get(
                "current_date", datetime.now().strftime("%A, %B %d, %Y")
            ),
        )

        async def _dummy_stream_events():
            if hasattr(mock_create_event_stream_mock, "stream_return_values"):
                for event in mock_create_event_stream_mock.stream_return_values:
                    yield event
            else:
                yield AgentFinalAnswer(text="mocked_answer")

        return react_inputs, _dummy_stream_events()

    with patch(
        "ai_gateway.api.v2.chat.agent.create_event_stream",
        side_effect=_dummy_create_event_stream,
    ) as mock:
        mock_create_event_stream_mock = mock

        yield mock


@pytest.fixture(name="mock_model")
def mock_model_fixture(model: Model):
    with patch("ai_gateway.prompts.Prompt._build_model", return_value=model) as mock:
        yield mock


@pytest.fixture(name="mocked_tools")
def mocked_tools_fixture():
    with patch(
        "ai_gateway.chat.executor.GLAgentRemoteExecutor.tools",
        new_callable=PropertyMock,
        return_value=[],
    ) as mock:
        yield mock


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {"custom_models": {"enabled": True}}


def chunk_to_model(chunk: str, klass: Type[AgentBaseEvent]) -> AgentBaseEvent:
    res = json.loads(chunk)
    data = res.pop("data")
    return klass.model_validate({**res, **data})


class TestReActAgentStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("agent_request", "model_metadata", "model_response", "expected_events"),
        [
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="How can I write hello world in python?",
                        ),
                    ]
                ),
                None,
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="How can I write hello world in python?",
                        ),
                    ],
                ),
                ModelMetadata(
                    name="vertex",
                    provider="claude-3-5-haiku-20241022",
                    endpoint=AnyUrl("http://localhost:4000"),
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="How can I write hello world in python?",
                        ),
                    ],
                ),
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="chat history",
                        ),
                        Message(role=Role.ASSISTANT, content="chat history"),
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                            context=Context(type="issue", content="issue content"),
                            current_file=CurrentFile(
                                file_path="main.py",
                                data="def main()",
                                selected_code=True,
                            ),
                        ),
                    ],
                    options=AgentRequestOptions(
                        agent_scratchpad=ReActAgentScratchpad(
                            agent_type="react",
                            steps=[
                                ReActAgentScratchpad.AgentStep(
                                    thought="thought",
                                    tool="tool",
                                    tool_input="tool_input",
                                    observation="observation",
                                )
                            ],
                        ),
                    ),
                    unavailable_resources=["Mystery Resource 1", "Mystery Resource 2"],
                ),
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="chat history",
                        ),
                        Message(role=Role.ASSISTANT, content="chat history"),
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                            context=Context(type="issue", content="issue content"),
                            current_file=CurrentFile(
                                file_path="main.py",
                                data="def main()",
                                selected_code=True,
                            ),
                        ),
                        Message(
                            role=Role.ASSISTANT,
                            content=None,
                            agent_scratchpad=[
                                AgentStep(
                                    action=AgentToolAction(
                                        thought="thought",
                                        tool="tool",
                                        tool_input="tool_input",
                                    ),
                                    observation="observation",
                                )
                            ],
                        ),
                    ],
                    unavailable_resources=["Mystery Resource 1", "Mystery Resource 2"],
                ),
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
            (
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What this MR changing?",
                            context=MergeRequestContext(
                                type="merge_request", title="Fixing database"
                            ),
                        ),
                        Message(
                            role=Role.ASSISTANT,
                            content=None,
                            agent_scratchpad=[],
                        ),
                    ],
                ),
                None,
                "thought\nFinal Answer: answer\n",
                [AgentFinalAnswer(text=c) for c in "answer"],
            ),
        ],
    )
    async def test_success(
        self,
        mock_client: TestClient,
        mock_model: Mock,
        mocked_tools: Mock,
        mock_create_event_stream: Mock,
        agent_request: AgentRequest,
        model_metadata: TypeModelMetadata,
        expected_events: list[TypeAgentEvent],
    ):

        async def _stream_gen():
            for event in expected_events:
                yield event

        side_effect = mock_create_event_stream.side_effect

        async def dynamic_side_effect(*args, **kwargs):
            inputs_part, _ = await side_effect(*args, **kwargs)

            return inputs_part, _stream_gen()

        mock_create_event_stream.side_effect = dynamic_side_effect

        json_params = agent_request.model_dump(mode="json")
        if model_metadata:
            json_params["model_metadata"] = model_metadata.model_dump(mode="json")

        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=json_params,
        )

        actual_events = [
            chunk_to_model(chunk, AgentFinalAnswer)
            for chunk in response.text.strip().split("\n")
        ]

        assert response.status_code == 200
        assert actual_events == expected_events
        mock_create_event_stream.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "messages",
            "actions",
            "model_metadata",
            "expected_actions",
            "unavailable_resources",
        ),
        [
            (
                [
                    Message(
                        role=Role.USER,
                        content="chat history",
                    ),
                    Message(role=Role.ASSISTANT, content="chat history"),
                    Message(
                        role=Role.USER,
                        content="What's the title of this issue?",
                        context=Context(type="issue", content="issue content"),
                        current_file=CurrentFile(
                            file_path="main.py",
                            data="def main()",
                            selected_code=True,
                        ),
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=None,
                        agent_scratchpad=[
                            AgentStep(
                                action=AgentToolAction(
                                    thought="thought",
                                    tool="tool",
                                    tool_input="tool_input",
                                ),
                                observation="observation",
                            )
                        ],
                    ),
                ],
                [
                    AgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                    )
                ],
                ModelMetadata(
                    name="mistral",
                    provider="litellm",
                    endpoint=AnyUrl("http://localhost:4000"),
                    api_key="token",
                ),
                [
                    AgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                    )
                ],
                ["Mystery Resource 1", "Mystery Resource 2"],
            ),
        ],
    )
    async def test_legacy_success(
        self,
        mock_client: TestClient,
        mock_create_event_stream: Mock,
        mock_track_internal_event,
        messages: list[Message],
        actions: list[TypeAgentEvent],
        expected_actions: list[TypeAgentEvent],
        model_metadata: ModelMetadata,
        unavailable_resources: list[str],
    ):
        async def _agent_stream(*_args, **_kwargs) -> AsyncIterator[TypeAgentEvent]:
            for action in actions:
                yield action

        side_effect = mock_create_event_stream.side_effect

        async def dynamic_side_effect(*args, **kwargs):
            inputs_part, _ = await side_effect(*args, **kwargs)

            return inputs_part, _agent_stream()

        mock_create_event_stream.side_effect = dynamic_side_effect

        mock_now = datetime(2024, 12, 25)
        expected_date_string = "Wednesday, December 25, 2024"

        with patch("ai_gateway.api.v2.chat.agent.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now

            response = mock_client.post(
                "/chat/agent",
                headers={
                    "Authorization": "Bearer 12345",
                    "X-Gitlab-Authentication-Type": "oidc",
                },
                json={
                    "messages": [m.model_dump(mode="json") for m in messages],
                    "model_metadata": model_metadata.model_dump(mode="json"),
                    "unavailable_resources": unavailable_resources,
                },
            )

        actual_actions = [
            chunk_to_model(chunk, AgentToolAction)
            for chunk in response.text.strip().split("\n")
        ]

        agent_inputs = ReActAgentInputs(
            messages=messages,
            agent_scratchpad=[
                AgentStep(
                    action=AgentToolAction(
                        thought="thought",
                        tool="tool",
                        tool_input="tool_input",
                    ),
                    observation="observation",
                )
            ],
            unavailable_resources=unavailable_resources,
            current_date=expected_date_string,
        )

        assert response.status_code == 200
        assert actual_actions == expected_actions

        mock_create_event_stream.assert_called_once()

        mock_track_internal_event.assert_called_once_with(
            "request_duo_chat",
            category="ai_gateway.api.v2.chat.agent",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "auth_user,agent_request,expected_status_code,expected_error,expected_internal_events,model_metadata",
        [
            (
                CloudConnectorUser(
                    authenticated=True, claims=UserClaims(scopes=["duo_chat"])
                ),
                AgentRequest(messages=[Message(role=Role.USER, content="Hi")]),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True, claims=UserClaims(scopes="wrong_scope")
                ),
                AgentRequest(messages=[Message(role=Role.USER, content="Hi")]),
                403,
                '{"detail":"Unauthorized to access duo chat"}',
                [],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(scopes=["duo_chat", "include_file_context"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[AdditionalContext(category="file")],
                        )
                    ]
                ),
                200,
                "",
                [
                    call(
                        "request_include_file_context",
                        category="ai_gateway.api.v2.chat.agent",
                    ),
                ],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(scopes=["duo_chat", "include_terminal_context"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Explain this terminal content",
                            additional_context=[
                                AdditionalContext(
                                    category="terminal",
                                    content="zsh: command not found: fakecommand",
                                )
                            ],
                        ),
                        Message(role=Role.ASSISTANT, content=None, agent_scratchpad=[]),
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(
                        scopes=["duo_chat", "include_repository_context"]
                    ),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Some prompt",
                            additional_context=[
                                AdditionalContext(
                                    category="repository",
                                    content="some repository context",
                                )
                            ],
                        ),
                        Message(role=Role.ASSISTANT, content=None, agent_scratchpad=[]),
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(scopes=["duo_chat", "include_directory_context"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Some prompt",
                            additional_context=[
                                AdditionalContext(
                                    category="directory",
                                    content="some directory context",
                                )
                            ],
                        ),
                        Message(role=Role.ASSISTANT, content=None, agent_scratchpad=[]),
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True, claims=UserClaims(scopes=["duo_chat"])
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[],
                        )
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True, claims=UserClaims(scopes=["duo_chat"])
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=None,
                        )
                    ]
                ),
                200,
                "",
                [call("request_duo_chat", category="ai_gateway.api.v2.chat.agent")],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(scopes=["amazon_q_integration"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=None,
                        )
                    ]
                ),
                403,
                '{"detail":"Unauthorized to access duo chat"}',
                [],
                None,
            ),
            (
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(scopes=["amazon_q_integration"]),
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=None,
                        )
                    ],
                ),
                200,
                "",
                [
                    call(
                        "request_amazon_q_integration",
                        category="ai_gateway.api.v2.chat.agent",
                    )
                ],
                AmazonQModelMetadata(
                    name="amazon_q",
                    provider="amazon_q",
                    role_arn="role-arn",
                ),
            ),
            (
                CloudConnectorUser(
                    authenticated=True, claims=UserClaims(scopes=["duo_chat"])
                ),
                AgentRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi",
                            additional_context=[AdditionalContext(category="file")],
                        )
                    ]
                ),
                403,
                '{"detail":"Unauthorized to access include_file_context"}',
                [],
                None,
            ),
        ],
    )
    async def test_authorization(
        self,
        auth_user: CloudConnectorUser,
        agent_request: AgentRequest,
        mock_client: TestClient,
        mock_model: Mock,
        expected_status_code: int,
        expected_error: str,
        expected_internal_events,
        mock_track_internal_event: Mock,
        model_metadata,
    ):
        json_params = agent_request.model_dump(mode="json")

        if model_metadata:
            json_params["model_metadata"] = model_metadata.model_dump(mode="json")

        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json=json_params,
        )

        assert response.status_code == expected_status_code

        if expected_error:
            assert response.text == expected_error
        else:
            mock_track_internal_event.assert_has_calls(expected_internal_events)


class TestChatAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "messages",
            "model_response",
            "expected_actions",
        ),
        [
            (
                [
                    Message(role=Role.USER, content="Basic request"),
                    Message(
                        role=Role.ASSISTANT,
                        content=None,
                        agent_scratchpad=[],
                    ),
                ],
                "thought\nFinal Answer: answer\n",
                [
                    AgentFinalAnswer(
                        text=c,
                    )
                    for c in "answer"
                ],
            ),
            (
                [
                    Message(
                        role=Role.USER,
                        content="Request with '{}' in the input",
                        current_file=CurrentFile(
                            file_path="main.c",
                            data="int main() {}",
                            selected_code=True,
                        ),
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=None,
                        agent_scratchpad=[],
                    ),
                ],
                "thought\nFinal Answer: answer\n",
                [
                    AgentFinalAnswer(
                        text=c,
                    )
                    for c in "answer"
                ],
            ),
        ],
    )
    async def test_request(
        self,
        mock_client: TestClient,
        mock_model: Mock,
        mocked_tools: Mock,
        mock_track_internal_event,
        messages: list[Message],
        expected_actions: list[TypeAgentEvent],
    ):
        response = mock_client.post(
            "/chat/agent",
            headers={
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
            },
            json={
                "messages": [m.model_dump(mode="json") for m in messages],
            },
        )

        actual_actions = [
            chunk_to_model(chunk, AgentFinalAnswer)
            for chunk in response.text.strip().split("\n")
        ]

        assert response.status_code == 200
        assert actual_actions == expected_actions

        mock_track_internal_event.assert_called_once_with(
            "request_duo_chat",
            category="ai_gateway.api.v2.chat.agent",
        )
