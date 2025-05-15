import json
from datetime import datetime
from typing import AsyncIterator
from unittest.mock import Mock, PropertyMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from starlette.testclient import TestClient

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.chat.agent import convert_v1_to_v2_inputs
from ai_gateway.api.v1.chat.typing import ChatRequest, PromptPayload
from ai_gateway.api.v2.chat.typing import AgentRequest
from ai_gateway.chat.agents import AgentStep, AgentToolAction, Message, ReActAgentInputs
from ai_gateway.chat.agents.typing import AgentFinalAnswer, TypeAgentEvent
from ai_gateway.feature_flags import (
    FeatureFlag,
    current_feature_flag_context,
    is_feature_enabled,
)
from ai_gateway.models import KindAnthropicModel
from ai_gateway.models.base_chat import Role


@pytest.fixture
def auth_user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(scopes=["duo_chat", "amazon_q_integration"]),
    )


@pytest.fixture(scope="class")
def fast_api_router():
    return api_router


@pytest.fixture
def text_content():
    return "\n\nHuman: hello, what is your name?\n\nAssistant:"


@pytest.fixture
def chat_messages():
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant",
            "context": None,
            "current_file": None,
            "additional_context": None,
            "agent_scratchpad": None,
        },
        {
            "role": "user",
            "content": "Hi",
            "context": None,
            "current_file": None,
            "additional_context": None,
            "agent_scratchpad": None,
        },
    ]


@pytest.fixture
def chat_response():
    return [
        {
            "role": "user",
            "content": "You are a helpful assistant\\n\\nHi",
            "context": None,
            "current_file": None,
            "additional_context": None,
            "agent_scratchpad": None,
        },
    ]


@pytest.fixture
def request_body(request, content_fixture):
    return {
        "prompt_components": [
            {
                "type": "prompt",
                "metadata": {
                    "source": "gitlab-rails-sm",
                    "version": "17.0.0-ee",
                },
                "payload": {
                    "content": request.getfixturevalue(content_fixture),
                    "provider": "anthropic",
                    "model": KindAnthropicModel.CLAUDE_2_1.value,
                    "model_endpoint": "https://api.example.com",
                },
            }
        ],
        "stream": False,
    }


@pytest.fixture
def default_headers():
    return {
        "Authorization": "Bearer 12345",
        "X-Gitlab-Authentication-Type": "oidc",
        "X-GitLab-Instance-Id": "1234",
        "X-GitLab-Realm": "self-managed",
    }


@pytest.fixture
def mock_create_event_stream():
    async def _dummy_create_event_stream(*args, **kwargs):

        async def _dummy_stream_events():
            if hasattr(mock_create_event_stream_mock, "stream_return_values"):
                for event in mock_create_event_stream_mock.stream_return_values:
                    yield event
            else:
                yield AgentFinalAnswer(text="mocked_answer")

        return None, _dummy_stream_events()

    with patch(
        "ai_gateway.api.v1.chat.agent.create_event_stream",
        side_effect=_dummy_create_event_stream,
    ) as mock:
        mock_create_event_stream_mock = mock

        yield mock


@pytest.fixture
def stub_get_agent():
    fake_agent = Mock(name="ReActAgent")
    with patch("ai_gateway.api.v2.chat.agent.get_agent", return_value=fake_agent):
        yield fake_agent


@pytest.fixture
def stub_executor_factory():
    class _StubExecutor:
        def on_behalf(self, *_, **__):
            pass

        stream = Mock()

    with patch(
        "ai_gateway.api.v1.chat.agent.get_gl_agent_remote_executor_factory",
        return_value=lambda agent: _StubExecutor(),
    ):
        yield


class TestConvertV1ToV2Inputs:
    @pytest.mark.parametrize("content_fixture", ["text_content", "chat_messages"])
    def test_convert_v1_to_v2_inputs(self, content_fixture, request_body, request):
        current_feature_flag_context.set({FeatureFlag.CHAT_V1_REDIRECT})
        chat_request = ChatRequest(**request_body)
        expected_content = request.getfixturevalue(content_fixture)

        agent_request = convert_v1_to_v2_inputs(chat_request)

        if isinstance(expected_content, str):
            assert [m.model_dump() for m in agent_request.messages] == [
                Message(role=Role.USER, content=expected_content).model_dump()
            ]
        else:
            assert [
                m.model_dump() for m in agent_request.messages
            ] == request.getfixturevalue("chat_response")


class TestRedirectedV1ChatEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("content_fixture", ["text_content", "chat_messages"])
    @pytest.mark.parametrize("stream", [False, True])
    async def test_success_response(
        self,
        content_fixture,
        mock_client: TestClient,
        mock_create_event_stream: Mock,
        request_body,
        default_headers,
        stream,
    ):
        async def _dummy_stream(*_, **__):
            for ch in "answer":
                yield AgentFinalAnswer(text=ch)

        side_effect = mock_create_event_stream.side_effect

        async def dynamic_side_effect(*args, **kwargs):
            inputs_part, _ = await side_effect(*args, **kwargs)

            return None, _dummy_stream()

        mock_create_event_stream.side_effect = dynamic_side_effect

        current_feature_flag_context.set({FeatureFlag.CHAT_V1_REDIRECT})

        payload = dict(request_body)
        payload["stream"] = stream

        response = mock_client.post(
            "/chat/agent",
            headers=default_headers,
            json=payload,
        )

        assert response.status_code == 200
        if stream:
            assert response.text == "answer"
            assert response.headers["content-type"].startswith("text/event-stream")
        else:
            assert response.json()["response"] == "answer"

        mock_create_event_stream.assert_called_once()
