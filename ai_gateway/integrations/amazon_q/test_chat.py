from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from ai_gateway.integrations.amazon_q.chat import ChatAmazonQ, Reference, ReferenceSpan
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory


class TestChatAmazonQ:
    @pytest.fixture
    def mock_q_client_factory(self):
        return mock.MagicMock(AmazonQClientFactory)

    @pytest.fixture
    def chat_amazon_q(self, mock_q_client_factory):
        return ChatAmazonQ(amazon_q_client_factory=mock_q_client_factory)

    @pytest.fixture
    def messages(self):
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

    @pytest.fixture
    def mock_user(self):
        return mock.MagicMock()

    def assert_chunk_content(self, chunks, expected_content):
        """Helper method to assert chunk content"""
        chunk_list = list(chunks)
        if not expected_content:
            assert len(chunk_list) == 1
            assert chunk_list[0].message.content == ""
        else:
            assert len(chunk_list) == 1
            assert chunk_list[0].message.content == expected_content

    def test_process_complete_reference(self, chat_amazon_q):
        """Test processing a complete reference with all fields present"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": {"shape": "MIT"},
                        "url": {"shape": "https://github.com/aws/aws-sdk"},
                        "recommendationContentSpan": {"shape": "lines 10-20"},
                    }
                ]
            }
        }
        expected = "aws-sdk [MIT]: https://github.com/aws/aws-sdk (lines 10-20)"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_multiple_references(self, chat_amazon_q):
        """Test processing multiple references"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": {"shape": "MIT"},
                        "url": {"shape": "https://github.com/aws/aws-sdk"},
                        "recommendationContentSpan": {"shape": "lines 10-20"},
                    },
                    {
                        "repository": {"shape": "boto3"},
                        "licenseName": {"shape": "Apache-2.0"},
                        "url": {"shape": "https://github.com/boto/boto3"},
                        "recommendationContentSpan": {"shape": "lines 5-15"},
                    },
                ]
            }
        }
        expected = (
            "aws-sdk [MIT]: https://github.com/aws/aws-sdk (lines 10-20)\n"
            "boto3 [Apache-2.0]: https://github.com/boto/boto3 (lines 5-15)"
        )
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_missing_optional_fields(self, chat_amazon_q):
        """Test processing reference with missing optional fields"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": {"shape": "MIT"},
                    }
                ]
            }
        }
        expected = "aws-sdk [MIT]"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_direct_string_values(self, chat_amazon_q):
        """Test processing reference with direct string values instead of shape objects"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": "aws-sdk",
                        "licenseName": "MIT",
                        "url": "https://github.com/aws/aws-sdk",
                        "recommendationContentSpan": "lines 10-20",
                    }
                ]
            }
        }
        expected = "aws-sdk [MIT]: https://github.com/aws/aws-sdk (lines 10-20)"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_empty_references(self, chat_amazon_q):
        """Test processing empty references list"""
        event = {"codeReferenceEvent": {"references": []}}
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, "")

    def test_process_mixed_shape_and_direct_values(self, chat_amazon_q):
        """Test processing reference with mixed shape objects and direct values"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": "MIT",
                        "url": {"shape": "https://github.com/aws/aws-sdk"},
                        "recommendationContentSpan": "lines 10-20",
                    }
                ]
            }
        }
        expected = "aws-sdk [MIT]: https://github.com/aws/aws-sdk (lines 10-20)"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_null_values(self, chat_amazon_q):
        """Test processing reference with null values"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": None,
                        "url": None,
                        "recommendationContentSpan": None,
                    }
                ]
            }
        }
        expected = "aws-sdk"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_invalid_event_structure(self, chat_amazon_q):
        """Test processing invalid event structure"""
        invalid_events = [
            {},  # Empty event
            {"wrongKey": {}},  # Wrong key
            {"codeReferenceEvent": {}},  # Missing references
            {"codeReferenceEvent": {"references": None}},  # Null references
            None,  # None event
        ]
        for event in invalid_events:
            chunks = chat_amazon_q._process_code_reference_event(event)
            self.assert_chunk_content(chunks, "")

    def test_process_invalid_reference_data(self, chat_amazon_q):
        """Test processing invalid reference data"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {
                            "wrong_key": "aws-sdk"
                        },  # Invalid shape structure
                        "licenseName": 123,  # Invalid type
                        "url": [],  # Invalid type
                    }
                ]
            }
        }
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, "")

    def test_process_repository_only(self, chat_amazon_q):
        """Test processing reference with only repository field"""
        event = {
            "codeReferenceEvent": {"references": [{"repository": {"shape": "aws-sdk"}}]}
        }
        expected = "aws-sdk"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_process_multiple_mixed_references(self, chat_amazon_q):
        """Test processing multiple references with mixed valid and invalid data"""
        event = {
            "codeReferenceEvent": {
                "references": [
                    {
                        "repository": {"shape": "aws-sdk"},
                        "licenseName": {"shape": "MIT"},
                    },
                    {"invalid": "data"},
                    {
                        "repository": "boto3",
                        "url": "https://github.com/boto/boto3",
                    },
                ]
            }
        }
        expected = "aws-sdk [MIT]\nboto3: https://github.com/boto/boto3"
        chunks = chat_amazon_q._process_code_reference_event(event)
        self.assert_chunk_content(chunks, expected)

    def test_reference_span_model(self):
        """Test ReferenceSpan model creation and validation"""
        # Test valid shape
        span = ReferenceSpan(shape="test-shape")
        assert span.shape == "test-shape"

        # Test empty shape
        with pytest.raises(ValueError):
            ReferenceSpan(shape="")

        # Test missing shape
        with pytest.raises(ValueError):
            ReferenceSpan()

    def test_reference_model_with_span_objects(self):
        """Test Reference model with ReferenceSpan objects"""
        reference = Reference(
            repository=ReferenceSpan(shape="repo-name"),
            licenseName=ReferenceSpan(shape="MIT"),
            url=ReferenceSpan(shape="https://example.com"),
            recommendationContentSpan=ReferenceSpan(shape="lines 1-10"),
        )

        assert reference.get_repository() == "repo-name"
        assert reference.get_license_name() == "MIT"
        assert reference.get_url() == "https://example.com"
        assert reference.get_span() == "lines 1-10"

    def test_reference_model_with_direct_strings(self):
        """Test Reference model with direct string values"""
        reference = Reference(
            repository="repo-name",
            licenseName="MIT",
            url="https://example.com",
            recommendationContentSpan="lines 1-10",
        )

        assert reference.get_repository() == "repo-name"
        assert reference.get_license_name() == "MIT"
        assert reference.get_url() == "https://example.com"
        assert reference.get_span() == "lines 1-10"

    def test_reference_model_with_mixed_values(self):
        """Test Reference model with mixed ReferenceSpan and string values"""
        reference = Reference(
            repository=ReferenceSpan(shape="repo-name"),
            licenseName="MIT",
            url=ReferenceSpan(shape="https://example.com"),
            recommendationContentSpan="lines 1-10",
        )

        assert reference.get_repository() == "repo-name"
        assert reference.get_license_name() == "MIT"
        assert reference.get_url() == "https://example.com"
        assert reference.get_span() == "lines 1-10"

    def test_reference_model_with_none_values(self):
        """Test Reference model with None values"""
        reference = Reference(
            repository=None, licenseName=None, url=None, recommendationContentSpan=None
        )

        assert reference.get_repository() is None
        assert reference.get_license_name() is None
        assert reference.get_url() is None
        assert reference.get_span() is None

    def test_reference_format_all_fields(self):
        """Test format_reference with all fields present"""
        reference = Reference(
            repository="repo-name",
            licenseName="MIT",
            url="https://example.com",
            recommendationContentSpan="lines 1-10",
        )

        expected = "repo-name [MIT]: https://example.com (lines 1-10)"
        assert reference.format_reference() == expected

    def test_reference_format_partial_fields(self):
        """Test format_reference with partial fields"""
        test_cases = [
            {"input": {"repository": "repo-name"}, "expected": "repo-name"},
            {
                "input": {"repository": "repo-name", "licenseName": "MIT"},
                "expected": "repo-name [MIT]",
            },
            {
                "input": {"repository": "repo-name", "url": "https://example.com"},
                "expected": "repo-name: https://example.com",
            },
            {
                "input": {
                    "repository": "repo-name",
                    "recommendationContentSpan": "lines 1-10",
                },
                "expected": "repo-name (lines 1-10)",
            },
            {
                "input": {"licenseName": "MIT", "url": "https://example.com"},
                "expected": "[MIT]: https://example.com",
            },
        ]

        for case in test_cases:
            reference = Reference(**case["input"])
            assert reference.format_reference() == case["expected"]

    def test_reference_model_validation(self):
        """Test Reference model validation"""
        # Test with invalid ReferenceSpan
        with pytest.raises(ValueError):
            Reference(repository=ReferenceSpan(shape=""))

        # Test with invalid types
        with pytest.raises(ValueError):
            Reference(repository=123)  # type: ignore

        with pytest.raises(ValueError):
            Reference(url=["invalid"])  # type: ignore

    def test_reference_format_empty(self):
        """Test format_reference with empty Reference"""
        reference = Reference()
        assert reference.format_reference() == ""

    def test_model_validate_method(self):
        """Test model_validate method with various inputs"""
        # Valid input with shape objects
        input_data = {
            "repository": {"shape": "repo-name"},
            "licenseName": {"shape": "MIT"},
            "url": {"shape": "https://example.com"},
            "recommendationContentSpan": {"shape": "lines 1-10"},
        }
        reference = Reference.model_validate(input_data)
        assert reference.get_repository() == "repo-name"

        # Valid input with direct strings
        input_data = {
            "repository": "repo-name",
            "licenseName": "MIT",
            "url": "https://example.com",
            "recommendationContentSpan": "lines 1-10",
        }
        reference = Reference.model_validate(input_data)
        assert reference.get_repository() == "repo-name"

        # Invalid input
        with pytest.raises(ValueError):
            Reference.model_validate({"repository": {"invalid": "value"}})

    def test_generate_response(
        self,
        chat_amazon_q,
        messages,
        mock_user,
        mock_q_client,
        mock_q_client_factory,
    ):
        role_arn = "role-arn"
        result = chat_amazon_q.invoke(messages, user=mock_user, role_arn=role_arn)

        assert result.content == "Streamed response"
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

    def test_stream(
        self, chat_amazon_q, mock_user, mock_q_client, mock_q_client_factory
    ):
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
        self,
        chat_amazon_q,
        messages,
        mock_user,
        mock_q_client,
        mock_q_client_factory,
    ):
        role_arn = "role-arn"

        stream = chat_amazon_q._stream(messages, user=mock_user, role_arn=role_arn)

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
