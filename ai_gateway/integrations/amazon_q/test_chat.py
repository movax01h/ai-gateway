from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory


@pytest.fixture
def mock_q_client_factory():
    return mock.MagicMock(AmazonQClientFactory)


class TestChatAmazonQ:
    @pytest.fixture
    def chat_amazon_q(self, mock_q_client_factory):
        return ChatAmazonQ(amazon_q_client_factory=mock_q_client_factory)

    @pytest.fixture
    def sample_messages(self):
        return [
            SystemMessage(content="system message", role="user"),
            HumanMessage(content="user message", role="user"),
            AIMessage(content="assistant message", role="user"),
            HumanMessage(content="latest user message", role="user"),
            AIMessage(content="latest assistant message", role="user"),
        ]

    @pytest.fixture
    def mock_q_client(self, mock_q_client_factory):
        mock_stream = mock.MagicMock()
        mock_stream.close = mock.MagicMock()
        mock_stream.__iter__.return_value = [
            {"assistantResponseEvent": {"content": "Streamed response"}}
        ]
        mock_response = {"responseStream": mock_stream}

        q_client = mock.MagicMock()
        q_client.send_message.return_value = mock_response
        mock_q_client_factory.get_client.return_value = q_client

        return q_client

    def test_generate_response(self, chat_amazon_q, sample_messages):
        result = chat_amazon_q.invoke(sample_messages)

        assert result.content == "Amazon Q"

    def test_stream(self, chat_amazon_q, mock_q_client, mock_q_client_factory):
        mock_user = mock.MagicMock()
        role_arn = "role-arn"

        messages = [
            SystemMessage(content="system message", role="user"),
            HumanMessage(content="user message", role="user"),
        ]

        stream = chat_amazon_q._stream(messages, user=mock_user, role_arn=role_arn)

        chunk = next(stream)
        assert isinstance(chunk, ChatGenerationChunk)
        assert chunk.message.content == "Streamed response"
        mock_q_client_factory.get_client.assert_called_once_with(
            current_user=mock_user, role_arn=role_arn
        )
        mock_q_client.send_message.assert_called_once_with(
            message={"content": "system message user message"},
            history=[],
        )

    def test_stream_history(
        self, chat_amazon_q, mock_q_client, mock_q_client_factory, sample_messages
    ):
        mock_user = mock.MagicMock()
        role_arn = "role-arn"

        stream = chat_amazon_q._stream(
            sample_messages, user=mock_user, role_arn=role_arn
        )

        chunk = next(stream)
        assert isinstance(chunk, ChatGenerationChunk)
        assert chunk.message.content == "Streamed response"
        mock_q_client_factory.get_client.assert_called_once_with(
            current_user=mock_user, role_arn=role_arn
        )
        mock_q_client.send_message.assert_called_once_with(
            message={
                "content": "system message latest user message latest assistant message"
            },
            history=[
                {"userInputMessage": {"content": "user message"}},
                {"assistantResponseMessage": {"content": "assistant message"}},
            ],
        )

    def test_identifying_params(self, chat_amazon_q):
        params = chat_amazon_q._identifying_params
        assert params == {"model": "amazon_q"}

    def test_llm_type(self, chat_amazon_q):
        assert chat_amazon_q._llm_type == "amazon_q"
