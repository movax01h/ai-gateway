from contextlib import contextmanager
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, call, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from snowplow_tracker import Snowplow

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
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models.amazon_q import AmazonQModel
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent, SnowplowEventContext
from lib.billing_events import BillingEvent, BillingEventsClient


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeGeneration:
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @pytest.fixture(name="mock_user")
    def mock_user_fixture(self):
        return Mock(spec=CloudConnectorUser)

    @pytest.fixture(name="mock_billing_client")
    def mock_billing_client_fixture(self):
        return Mock(spec=BillingEventsClient)

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
            model, tokenization_strategy_mock, Mock(spec=SnowplowInstrumentator)
        )
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
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
            model, tokenization_strategy_mock, Mock(spec=SnowplowInstrumentator)
        )
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
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

            with patch.object(
                expected_post_processor, "process"
            ) as mock_post_processor:
                _ = await use_case.execute(
                    "prefix",
                    "test.py",
                    editor_lang="Python",
                    raw_prompt="test prompt",
                    model_provider=model_provider,
                )
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
                    TextGenModelChunk(text="hello "),
                    TextGenModelChunk(text="world!"),
                ],
                [
                    "hello ",
                    "world!",
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
        ("stream", "response_token_length"),
        [
            (True, 9),
            (False, 4),
        ],
    )
    async def test_snowplow_instrumentation(
        self,
        use_case: CodeGenerations,
        stream: bool,
        response_token_length: int,
    ):
        async def _stream_generator(
            prefix: str, suffix: str, stream: bool
        ) -> AsyncIterator[TextGenModelChunk]:
            model_chunks = [
                TextGenModelChunk(text="hello "),
                TextGenModelChunk(text="world!"),
            ]

            for chunk in model_chunks:
                yield chunk

        snowplow_event_context = MagicMock(spec=SnowplowEventContext)
        expected_event_1 = SnowplowEvent(
            context=snowplow_event_context,
            action="tokens_per_user_request_prompt",
            label="code_generation",
            value=1,
        )
        expected_event_2 = SnowplowEvent(
            context=snowplow_event_context,
            action="tokens_per_user_request_response",
            label="code_generation",
            value=response_token_length,
        )

        with (
            patch.object(use_case, "tokenization_strategy") as mock,
            patch.object(use_case, "snowplow_instrumentator") as snowplow_mock,
        ):
            mock.estimate_length = Mock(return_value=[4, 5])

            if stream:
                with patch.object(
                    use_case.model, "generate", new_callable=AsyncMock
                ) as mock_generate:
                    mock_generate.side_effect = _stream_generator

                    actual = await use_case.execute(
                        prefix="any",
                        file_name="bar.py",
                        editor_lang="Python",
                        model_provider=ModelProvider.ANTHROPIC,
                        stream=stream,
                        snowplow_event_context=snowplow_event_context,
                    )

                    if isinstance(actual, AsyncIterator):
                        async for _ in actual:
                            pass

            else:
                with patch.object(
                    use_case.model, "generate", new_callable=AsyncMock
                ) as mock_generate:
                    mock_generate.return_value = TextGenModelOutput(
                        text="output", score=0, safety_attributes=SafetyAttributes()
                    )

                    actual = await use_case.execute(
                        prefix="any",
                        file_name="bar.py",
                        editor_lang="Python",
                        model_provider=ModelProvider.ANTHROPIC,
                        stream=stream,
                        snowplow_event_context=snowplow_event_context,
                    )

            mock.estimate_length.assert_called()

            snowplow_mock.watch.assert_has_calls(
                [call(expected_event_1), call(expected_event_2)]
            )

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
                async for res in result:
                    assert res.text == expected_output
            else:
                assert result.text == expected_output

            mock_generate.assert_called_once_with(
                prefix, suffix, file_name, expected_lang, stream, **additional_kwargs
            )

    @pytest.fixture(name="use_case_with_billing")
    def use_case_with_billing_fixture(self, mock_billing_client):
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
            snowplow_instrumentator=Mock(spec=SnowplowInstrumentator),
            billing_event_client=mock_billing_client,
        )
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)

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
        self, use_case_with_billing, mock_user, mock_billing_client
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

        # Verify billing event was tracked with correct parameters
        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            category="CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_generations",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 25}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )

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

    async def test_no_billing_event_when_no_billing_client(self, mock_user):
        """Test that no billing event is tracked when billing client is not provided."""
        model = Mock(spec=TextGenModelBase)
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        model.metadata = Mock(name="test-model", engine="test-engine")
        model.metadata.name = "test-model"
        model.metadata.engine = "test-engine"

        tokenization_strategy_mock = Mock(spec=TokenStrategyBase)
        tokenization_strategy_mock.estimate_length = Mock(return_value=[25, 30])

        use_case = CodeGenerations(
            model=model,
            tokenization_strategy=tokenization_strategy_mock,
            snowplow_instrumentator=Mock(spec=SnowplowInstrumentator),
            # No billing client provided
        )
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)

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

        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="generated code",
                score=0.8,
                safety_attributes=SafetyAttributes(),
            )
        )

        # This should not raise an exception
        with patch.object(PostProcessor, "process", return_value="generated code"):
            result = await use_case.execute(
                prefix="def hello",
                file_name="test.py",
                editor_lang="python",
                user=mock_user,
            )

        assert result.text == "generated code"

    async def test_billing_event_exception_handling(
        self, use_case_with_billing, mock_user, mock_billing_client
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

        # Make billing client raise an exception
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

        # Verify the generation still works despite billing error
        assert result.text == expected_output
        mock_billing_client.track_billing_event.assert_called_once()

    async def test_billing_event_exception_handling_streaming(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event exceptions are handled gracefully for streaming."""

        async def _stream_generator():
            yield TextGenModelChunk(text="hello ")
            yield TextGenModelChunk(text="world!")

        use_case_with_billing.model.generate = AsyncMock(
            side_effect=lambda *args, **kwargs: _stream_generator()
        )

        # Make billing client raise an exception
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

        assert chunks == ["hello ", "world!"]
        mock_billing_client.track_billing_event.assert_called_once()

    async def test_billing_event_with_zero_tokens(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event is tracked even when output tokens is zero."""
        # Mock tokenization strategy to return 0 tokens
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
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            category="CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_generations",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 0}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )

    async def test_billing_event_with_anthropic_provider(
        self, use_case_with_billing, mock_user, mock_billing_client
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

        # Verify billing event was tracked
        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            category="CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_generations",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 25}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )

    async def test_billing_event_tracked_for_streaming(
        self, use_case_with_billing, mock_user, mock_billing_client
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

        assert chunks == ["hello ", "world!"]

        # Billing events should now be tracked for streaming responses
        # The billing event is tracked in the finally block of _handle_stream
        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
            category="CodeGenerations",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_generations",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 55}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )
