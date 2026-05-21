# pylint: disable=file-naming-for-tests,unused-argument
from contextlib import contextmanager
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.code_suggestions import CodeGenerations, ModelProvider
from ai_gateway.code_suggestions.processing import TokenStrategyBase
from ai_gateway.code_suggestions.processing.post.generations import (
    PostProcessor,
    PostProcessorAnthropic,
)
from ai_gateway.code_suggestions.processing.pre import PromptBuilderBase
from ai_gateway.code_suggestions.processing.typing import (
    MetadataCodeContent,
    MetadataPromptBuilder,
    Prompt,
)
from ai_gateway.models.amazon_q import AmazonQModel
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes
from lib.billing_events import BillingEvent, BillingEventService, ExecutionEnvironment
from lib.billing_events.client import BillingEventsClient


@pytest.fixture(name="llm_ops_from_context")
def llm_ops_from_context_fixture():
    """Raw llm_operations dicts as populated by the request-context contextvar."""
    return [
        {
            "model_id": "context-model",
            "model_engine": "openai",
            "model_provider": "openai",
            "token_count": 999,
            "prompt_tokens": 500,
            "completion_tokens": 499,
        }
    ]


@pytest.fixture(name="expected_llm_ops")
def expected_llm_ops_fixture(llm_ops_from_context):
    """Same dicts after BillingEventService validates them through LLMOperation."""
    return [
        {
            **llm_ops_from_context[0],
            "agent_name": None,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "operation_type": "standard",
        }
    ]


@pytest.fixture(name="expected_billing_metadata")
def expected_billing_metadata_fixture(expected_llm_ops):
    """Full metadata dict the service forwards to the client."""
    return {
        "feature_qualified_name": "code_suggestions",
        "feature_ai_catalog_item": False,
        "execution_environment": ExecutionEnvironment.CODE_GENERATIONS.value,
        "llm_operations": expected_llm_ops,
        "tool_names": [],
        "orbit_called": False,
    }


@pytest.fixture(name="mock_get_llm_operations")
def mock_get_llm_operations_fixture(llm_ops_from_context):
    with patch("lib.billing_events.service.get_llm_operations") as mock:
        mock.return_value = llm_ops_from_context
        yield mock


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeGeneration:
    @pytest.fixture(name="mock_user")
    def mock_user_fixture(self):
        return Mock(spec=CloudConnectorUser)

    @pytest.fixture(name="mock_billing_client")
    def mock_billing_client_fixture(self):
        return Mock(spec=BillingEventsClient)

    @pytest.fixture(name="billing_service")
    def billing_service_fixture(self, mock_billing_client):
        return BillingEventService(client=mock_billing_client)

    @pytest.fixture(name="use_case", scope="class")
    def use_case_fixture(self):
        model = Mock(spec=TextGenModelBase)
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        tokenization_strategy_mock = Mock(spec=TokenStrategyBase)
        tokenization_strategy_mock.estimate_length = Mock(return_value=[1, 2])
        prompt_builder_mock = Mock(spec=PromptBuilderBase)
        prompt = Prompt(
            prefix="prompt",
            metadata=MetadataPromptBuilder(
                components={
                    "prompt": MetadataCodeContent(
                        length=len("prompt"),
                        length_tokens=1,
                    ),
                }
            ),
        )
        prompt_builder_mock.build = Mock(return_value=prompt)

        use_case = CodeGenerations(
            model, tokenization_strategy_mock, Mock(spec=BillingEventService)
        )
        use_case.prompt_builder = prompt_builder_mock

        yield use_case

    @pytest.fixture(name="use_case_q", scope="class")
    def use_case_q_fixture(self):
        model = Mock(spec=AmazonQModel)
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        tokenization_strategy_mock = Mock(spec=TokenStrategyBase)
        tokenization_strategy_mock.estimate_length = Mock(return_value=[1, 2])
        prompt_builder_mock = Mock(spec=PromptBuilderBase)
        prompt = Prompt(
            prefix="prompt",
            metadata=MetadataPromptBuilder(
                components={
                    "prompt": MetadataCodeContent(
                        length=len("prompt"),
                        length_tokens=1,
                    ),
                }
            ),
        )
        prompt_builder_mock.build = Mock(return_value=prompt)

        use_case = CodeGenerations(
            model, tokenization_strategy_mock, Mock(spec=BillingEventService)
        )
        use_case.prompt_builder = prompt_builder_mock

        yield use_case

    @pytest.mark.parametrize(
        ("model_provider", "expected_post_processor"),
        [
            (ModelProvider.VERTEX_AI, PostProcessor),
            (ModelProvider.ANTHROPIC, PostProcessorAnthropic),
        ],
    )
    async def test_execute_with_prompt_version(
        self, use_case: CodeGenerations, model_provider, expected_post_processor
    ):
        with patch.object(
            use_case.model, "generate", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = TextGenModelOutput(
                text="output", score=0, safety_attributes=SafetyAttributes()
            )

            with (
                patch.object(expected_post_processor, "process") as mock_post_processor,
                patch(
                    "ai_gateway.code_suggestions.generations.init_llm_operations"
                ) as mock_init_llm_ops,
            ):
                _ = await use_case.execute(
                    "prefix",
                    "test.py",
                    editor_lang="Python",
                    raw_prompt="test prompt",
                    model_provider=model_provider,
                )
                mock_init_llm_ops.assert_called()
                mock_generate.assert_called()
                mock_post_processor.assert_called()

    @pytest.mark.parametrize(
        (
            "model_chunks",
            "expected_chunks",
        ),
        [
            (
                [
                    TextGenModelChunk(text="hello\n"),
                    TextGenModelChunk(text="world!\n"),
                ],
                [
                    "hello\n",
                    "world!\n",
                ],
            ),
        ],
    )
    async def test_execute_stream(
        self,
        use_case: CodeGenerations,
        model_chunks: list[TextGenModelChunk],
        expected_chunks: list[str],
    ):
        async def _stream_generator(
            prefix: str, suffix: str, stream: bool
        ) -> AsyncIterator[TextGenModelChunk]:
            for chunk in model_chunks:
                yield chunk

        with patch.object(
            use_case.model, "generate", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.side_effect = _stream_generator

            actual = await use_case.execute(
                prefix="any",
                file_name="bar.py",
                editor_lang="Python",
                model_provider=ModelProvider.ANTHROPIC,
                stream=True,
            )

            chunks: List[str] = []
            if isinstance(actual, AsyncIterator):
                async for content in actual:
                    chunks.append(content.text)
            else:
                chunks.append(actual.text)

            assert chunks == expected_chunks

            mock_generate.assert_called_with(
                use_case.prompt_builder.build().prefix,
                "",
                stream=True,
            )

    @pytest.mark.parametrize(
        ("model_chunks"),
        [
            # Bug case: leading fence + lang tag split across chunks, trailing fence alone
            [
                TextGenModelChunk(text="```"),
                TextGenModelChunk(text="cpp"),
                TextGenModelChunk(text="\n"),
                TextGenModelChunk(text="void"),
                TextGenModelChunk(text=" f() {}\n"),
                TextGenModelChunk(text="```"),
            ],
            # Leading fence in single chunk
            [
                TextGenModelChunk(text="```cpp\n"),
                TextGenModelChunk(text="int main() { return 0; }\n"),
                TextGenModelChunk(text="```"),
            ],
            # Both fences inline in a single chunk
            [
                TextGenModelChunk(text="```\nint x = 1;\n```"),
            ],
            # Leading fence with no language identifier
            [
                TextGenModelChunk(text="```\n"),
                TextGenModelChunk(text="code\n"),
                TextGenModelChunk(text="```"),
            ],
        ],
    )
    async def test_execute_stream_strips_markdown_fences(
        self,
        use_case: CodeGenerations,
        model_chunks: list[TextGenModelChunk],
    ):
        async def _stream_generator(
            prefix: str, suffix: str, stream: bool
        ) -> AsyncIterator[TextGenModelChunk]:
            for chunk in model_chunks:
                yield chunk

        with patch.object(
            use_case.model, "generate", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.side_effect = _stream_generator

            actual = await use_case.execute(
                prefix="any",
                file_name="foo.cpp",
                editor_lang="C++",
                model_provider=ModelProvider.ANTHROPIC,
                stream=True,
            )

            chunks: List[str] = []
            assert isinstance(actual, AsyncIterator)
            async for content in actual:
                chunks.append(content.text)

            full_output = "".join(chunks)

            assert (
                "```" not in full_output
            ), f"Streamed generation leaked markdown fences: {full_output!r}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
            "expected_lang",
            "expected_output",
            "additional_kwargs",
        ),
        [
            # Basic case with editor_lang and suffix
            (
                "def test():",
                "# test function",
                "test.py",
                "python",
                False,
                "python",
                "\ngenerated code",
                {},
            ),
            # Empty suffix case
            (
                "def test():",
                "",
                "test.py",
                "python",
                False,
                "python",
                "\ngenerated code",
                {},
            ),
            # None suffix case
            (
                "def test():",
                None,
                "test.py",
                "python",
                False,
                "python",
                "\ngenerated code",
                {},
            ),
            # Language resolution from filename
            (
                "function test() {",
                "// test function",
                "script.js",
                None,
                False,
                "javascript",
                "\ngenerated code",
                {},
            ),
            # With additional kwargs
            (
                "def test",
                "():",
                "test.py",
                "python",
                False,
                "python",
                "\ngenerated code",
                {"temperature": 0.7, "max_tokens": 100},
            ),
            # With streaming
            (
                "def test",
                "():",
                "test.py",
                "python",
                True,
                "python",
                "\ngenerated code",
                {"temperature": 0.7, "max_tokens": 100},
            ),
        ],
    )
    async def test_amazon_q_model_generation(
        self,
        use_case_q: CodeGenerations,
        prefix: str,
        suffix: str | None,
        file_name: str,
        stream: bool,
        editor_lang: str | None,
        expected_lang: str,
        expected_output: str,
        additional_kwargs: dict,
    ):
        """Test generation with AmazonQModel with various input combinations."""

        async def _stream_generator(
            prefix: str,
            suffix: str,
            file_name: str,
            editor_lang: str,
            stream: bool,
            **kwargs: Any,
        ) -> AsyncIterator[TextGenModelChunk]:
            yield TextGenModelChunk(text=expected_output)

        with patch.object(
            use_case_q.model, "generate", new_callable=AsyncMock
        ) as mock_generate:
            if stream:
                mock_generate.side_effect = _stream_generator
            else:
                mock_generate.return_value = TextGenModelOutput(
                    text=expected_output, score=0, safety_attributes=SafetyAttributes()
                )

            result = await use_case_q.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
                stream=stream,
                **additional_kwargs,
            )

            assert result is not None
            if isinstance(result, AsyncIterator):
                chunks: List[str] = []
                async for res in result:
                    chunks.append(res.text)
                assert "".join(chunks) == expected_output
            else:
                assert result.text == expected_output

            mock_generate.assert_called_once_with(
                prefix, suffix, file_name, expected_lang, stream, **additional_kwargs
            )

    @pytest.fixture(name="use_case_with_billing")
    def use_case_with_billing_fixture(self, billing_service):
        model = Mock(spec=TextGenModelBase)
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        model.metadata = Mock(
            name="test-model", engine="test-engine", identifier="test-model-id"
        )
        model.metadata.name = "test-model"
        model.metadata.engine = "test-engine"
        model.metadata.identifier = "test-model-id"

        tokenization_strategy_mock = Mock(spec=TokenStrategyBase)
        tokenization_strategy_mock.estimate_length = Mock(return_value=[25, 30])

        use_case = CodeGenerations(
            model=model,
            tokenization_strategy=tokenization_strategy_mock,
            billing_event_service=billing_service,
        )

        prompt_builder_mock = Mock(spec=PromptBuilderBase)
        prompt = Prompt(
            prefix="prompt",
            metadata=MetadataPromptBuilder(
                components={
                    "prompt": MetadataCodeContent(
                        length=len("prompt"),
                        length_tokens=1,
                    ),
                }
            ),
        )
        prompt_builder_mock.build = Mock(return_value=prompt)
        use_case.prompt_builder = prompt_builder_mock

        return use_case

    async def test_billing_event_tracked_on_successful_generation(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing event is tracked when code generation is successful."""
        expected_output = "generated code"

        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0.8,
                safety_attributes=SafetyAttributes(),
            )
        )

        with patch.object(PostProcessor, "process", return_value=expected_output):
            await use_case_with_billing.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                user=mock_user,
            )

        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()

    async def test_no_billing_event_when_no_user_provided(
        self, use_case_with_billing, mock_billing_client
    ):
        """Test that no billing event is tracked when user is not provided."""
        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="generated code",
                score=0.8,
                safety_attributes=SafetyAttributes(),
            )
        )

        with patch.object(PostProcessor, "process", return_value="generated code"):
            await use_case_with_billing.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                # No user provided
            )

        mock_billing_client.track_billing_event.assert_not_called()

    async def test_billing_event_exception_handling(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing event exceptions are handled gracefully."""
        expected_output = "generated code"

        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0.8,
                safety_attributes=SafetyAttributes(),
            )
        )

        # Make billing service raise an exception
        mock_billing_client.track_billing_event.side_effect = Exception(
            "Billing service unavailable"
        )

        # This should not raise an exception - billing errors should be handled gracefully
        with patch.object(PostProcessor, "process", return_value=expected_output):
            result = await use_case_with_billing.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                user=mock_user,
            )

        assert result.text == expected_output
        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()

    async def test_billing_event_exception_handling_streaming(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing event exceptions are handled gracefully for streaming."""

        async def _stream_generator():
            yield TextGenModelChunk(text="hello ")
            yield TextGenModelChunk(text="world!")

        use_case_with_billing.model.generate = AsyncMock(
            side_effect=lambda *args, **kwargs: _stream_generator()
        )

        # Make billing service raise an exception
        mock_billing_client.track_billing_event.side_effect = Exception(
            "Billing service unavailable"
        )

        result = await use_case_with_billing.execute(
            prefix="def hello",
            file_name="test.py",
            editor_lang="python",
            stream=True,
            user=mock_user,
        )

        # Consume the stream - this should not raise an exception
        chunks = []
        async for chunk in result:
            chunks.append(chunk.text)

        assert "".join(chunks) == "hello world!"
        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()

    async def test_billing_event_with_zero_tokens(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing event is tracked even when output tokens is zero."""
        use_case_with_billing.tokenization_strategy.estimate_length = Mock(
            return_value=[0, 0]
        )

        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="",
                score=0.0,
                safety_attributes=SafetyAttributes(),
            )
        )

        with patch.object(PostProcessor, "process", return_value=""):
            await use_case_with_billing.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                user=mock_user,
            )

        # Should still track billing event even with 0 tokens for consistency
        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()

    async def test_billing_event_with_anthropic_provider(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing event is tracked correctly with Anthropic provider."""
        expected_output = "generated code"

        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0.8,
                safety_attributes=SafetyAttributes(),
            )
        )

        with patch.object(
            PostProcessorAnthropic, "process", return_value=expected_output
        ):
            await use_case_with_billing.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                model_provider=ModelProvider.ANTHROPIC,
                user=mock_user,
            )

        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()

    async def test_billing_event_tracked_for_streaming(
        self,
        use_case_with_billing,
        mock_user,
        mock_billing_client,
        mock_get_llm_operations,
        expected_billing_metadata,
    ):
        """Test that billing events are tracked for streaming responses."""

        async def _stream_generator():
            yield TextGenModelChunk(text="hello ")
            yield TextGenModelChunk(text="world!")

        use_case_with_billing.model.generate = AsyncMock(
            side_effect=lambda *args, **kwargs: _stream_generator()
        )

        result = await use_case_with_billing.execute(
            prefix="def hello",
            file_name="test.py",
            editor_lang="python",
            stream=True,
            user=mock_user,
        )

        # Consume the stream
        chunks = []
        async for chunk in result:
            chunks.append(chunk.text)

        assert "".join(chunks) == "hello world!"

        mock_billing_client.track_billing_event.assert_called_once_with(
            mock_user,
            BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            "CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata=expected_billing_metadata,
        )
        mock_get_llm_operations.assert_called()
