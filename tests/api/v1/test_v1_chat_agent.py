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
