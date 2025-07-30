import json
from datetime import datetime
from typing import AsyncIterator
from unittest.mock import Mock, PropertyMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from starlette.testclient import TestClient

from ai_gateway.api.v1 import api_router
from ai_gateway.api.v1.chat.typing import ChatRequest, PromptPayload
from ai_gateway.api.v2.chat.typing import AgentRequest
from ai_gateway.chat.agents import AgentStep, AgentToolAction, Message, ReActAgentInputs
from ai_gateway.chat.agents.typing import AgentFinalAnswer, TypeAgentEvent
from ai_gateway.models import KindAnthropicModel
from ai_gateway.models.base_chat import Role
from lib.feature_flags import (
    FeatureFlag,
    current_feature_flag_context,
    is_feature_enabled,
)


@pytest.fixture(name="auth_user")
def auth_user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(scopes=["duo_chat", "amazon_q_integration"]),
    )


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="text_content")
def text_content_fixture():
    return "\n\nHuman: hello, what is your name?\n\nAssistant:"


@pytest.fixture(name="chat_messages")
def chat_messages_fixture():
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


@pytest.fixture(name="chat_response")
def chat_response_fixture():
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


@pytest.fixture(name="request_body")
def request_body_fixture(request, content_fixture):
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


@pytest.fixture(name="default_headers")
def default_headers_fixture():
    return {
        "Authorization": "Bearer 12345",
        "X-Gitlab-Authentication-Type": "oidc",
        "X-GitLab-Instance-Id": "1234",
        "X-GitLab-Realm": "self-managed",
    }


@pytest.fixture(name="mock_create_event_stream")
def mock_create_event_stream_fixture():
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


@pytest.fixture(name="stub_get_agent")
def stub_get_agent_fixture():
    fake_agent = Mock(name="ReActAgent")
    with patch("ai_gateway.api.v2.chat.agent.get_agent", return_value=fake_agent):
        yield fake_agent


@pytest.fixture(name="stub_executor_factory")
def stub_executor_factory_fixture():
    class _StubExecutor:
        def on_behalf(self, *_, **__):
            pass

        stream = Mock()

    with patch(
        "ai_gateway.api.v1.chat.agent.get_gl_agent_remote_executor_factory",
        return_value=lambda agent: _StubExecutor(),
    ):
        yield
