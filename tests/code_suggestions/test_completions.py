from contextlib import contextmanager
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, call, patch

import pytest

from ai_gateway.code_suggestions import CodeCompletions, CodeCompletionsLegacy
from ai_gateway.code_suggestions.processing import (
    ModelEngineCompletions,
    ModelEngineOutput,
)
from ai_gateway.code_suggestions.processing.post.completions import PostProcessor
from ai_gateway.code_suggestions.processing.pre import PromptBuilderPrefixBased
from ai_gateway.code_suggestions.processing.typing import (
    LanguageId,
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
    Prompt,
    TokenStrategyBase,
)
from ai_gateway.instrumentators import KnownMetrics, TextGenModelInstrumentator
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    ModelAPIError,
    ModelMetadata,
    PalmCodeGeckoModel,
)
from ai_gateway.models.amazon_q import AmazonQModel
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from ai_gateway.tracking.snowplow import SnowplowEvent, SnowplowEventContext


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeCompletionsLegacy:
    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "engine_response_text",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                "a good suggestion",
                LanguageId.PYTHON,
                "a wonderful suggestion",
            ),
        ],
    )
    async def test_execute(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        engine_response_text: str,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        engine_response = [
            ModelEngineOutput(
                text=engine_response_text,
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=expected_language_id,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=1, output_tokens=2
                ),
            )
        ]
        engine = Mock(spec=ModelEngineCompletions)
        engine.generate = AsyncMock(return_value=engine_response)
        engine.model = PalmCodeGeckoModel(Mock(), "gl", "us-central-1")

        post_processor = Mock(spec=PostProcessor)
        post_processor.process.return_value = expected_output
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        with patch(
            "ai_gateway.code_suggestions.completions.benchmark"
        ) as mock_benchmark:
            use_case = CodeCompletionsLegacy(
                engine=engine,
                post_processor=post_processor_factory,
                snowplow_instrumentator=Mock(spec=SnowplowInstrumentator),
            )
            actual = await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
            )

        assert expected_output == actual[0].text
        assert expected_language_id == actual[0].lang_id

        engine.generate.assert_called_with(prefix, suffix, file_name, editor_lang)
        mock_benchmark.assert_called_with(
            metric_key=KnownMetrics.POST_PROCESSING_DURATION,
            labels={"model_engine": "vertex-ai", "model_name": "code-gecko@002"},
        )
        post_processor_factory.assert_called_with(
            prefix, suffix=suffix, lang_id=expected_language_id
        )
        post_processor.process.assert_called_with(engine_response_text)

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "engine_response_text",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                "",
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute_without_post_processing(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        engine_response_text: str,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        engine_response = [
            ModelEngineOutput(
                text=engine_response_text,
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=expected_language_id,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=1, output_tokens=2
                ),
            )
        ]
        engine = Mock(spec=ModelEngineCompletions)
        engine.generate = AsyncMock(return_value=engine_response)
        engine.model = PalmCodeGeckoModel(
            Mock(),
            "gl",
            "us-central-1",
        )

        post_processor = Mock(spec=PostProcessor)
        post_processor.process.return_value = expected_output
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        with patch(
            "ai_gateway.code_suggestions.completions.benchmark"
        ) as mock_benchmark:
            use_case = CodeCompletionsLegacy(
                engine=engine,
                post_processor=post_processor_factory,
                snowplow_instrumentator=Mock(spec=SnowplowInstrumentator),
            )
            await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
            )

        mock_benchmark.assert_not_called()
        post_processor.process.assert_not_called()

    async def test_snowplow_instrumentation(
        self,
    ):
        snowplow_instrumentator_mock = Mock(spec=SnowplowInstrumentator)
        snowplow_event_context = MagicMock(spec=SnowplowEventContext)
        expected_event_1 = SnowplowEvent(
            context=snowplow_event_context,
            action="tokens_per_user_request_prompt",
            label="code_completion",
            value=1,
        )

        expected_event_2 = SnowplowEvent(
            context=snowplow_event_context,
            action="tokens_per_user_request_response",
            label="code_completion",
            value=2,
        )

        engine_response = [
            ModelEngineOutput(
                text="",
                score=0,
                model=ModelMetadata(name="code-gecko", engine="vertex-ai"),
                lang_id=LanguageId.PYTHON,
                metadata=MetadataPromptBuilder(
                    components={
                        "prefix": MetadataCodeContent(length=10, length_tokens=2),
                        "suffix": MetadataCodeContent(length=10, length_tokens=2),
                    },
                ),
                tokens_consumption_metadata=TokensConsumptionMetadata(
                    input_tokens=1, output_tokens=2
                ),
            )
        ]
        engine = Mock(spec=ModelEngineCompletions)
        engine.generate = AsyncMock(return_value=engine_response)
        engine.model = PalmCodeGeckoModel(
            Mock(),
            "gl",
            "us-central-1",
        )

        post_processor = Mock(spec=PostProcessor)
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        with patch(
            "ai_gateway.code_suggestions.completions.benchmark"
        ) as mock_benchmark:
            use_case = CodeCompletionsLegacy(
                engine=engine,
                post_processor=post_processor_factory,
                snowplow_instrumentator=snowplow_instrumentator_mock,
            )
            _ = await use_case.execute(
                prefix="random_prefix",
                suffix="random_suffix",
                file_name="file_name",
                editor_lang="python",
                snowplow_event_context=snowplow_event_context,
            )

        mock_benchmark.assert_not_called()
        snowplow_instrumentator_mock.watch.assert_has_calls(
            [call(expected_event_1), call(expected_event_2)]
        )


@pytest.mark.asyncio
class TestCodeCompletions:
    @pytest.fixture(scope="class")
    def use_case(self):
        model = Mock(spec=TextGenModelBase)
        type(model).input_token_limit = PropertyMock(return_value=2_048)

        use_case = CodeCompletions(model, Mock(spec=TokenStrategyBase))
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
        use_case.prompt_builder = Mock(spec=PromptBuilderPrefixBased)

        yield use_case

    @pytest.fixture(scope="class")
    def completions_with_post_processing(self):
        model = Mock(spec=TextGenModelBase)
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="Unprocessed completion output",
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10),
            )
        )

        post_processor = Mock(spec=PostProcessor)
        post_processor.process.return_value = "Post-processed completion output"
        post_processor_factory = Mock()
        post_processor_factory.return_value = post_processor

        prompt_builder = Mock(spec=PromptBuilderPrefixBased)
        prompt_builder.build.return_value = Prompt(
            prefix="test_prefix",
            suffix="test_suffix",
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
                code_context=MetadataExtraInfo(
                    name="code_context",
                    pre=MetadataCodeContent(length=20, length_tokens=4),
                    post=MetadataCodeContent(length=15, length_tokens=3),
                ),
            ),
        )

        completions = CodeCompletions(
            model,
            tokenization_strategy=Mock(spec=TokenStrategyBase),
            post_processor=post_processor_factory,
        )
        completions.prompt_builder = prompt_builder
        completions.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)

        yield completions

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
            "context_max_percent",
            "code_context",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                False,
                1.0,
                None,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                None,
                False,
                1.0,
                None,
                None,
                "random_suggestion",
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name.py",
                None,
                False,
                1.0,
                None,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name.py",
                None,
                False,
                0.5,
                ["some context"],
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute(
        self,
        use_case: CodeCompletions,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        stream: bool,
        context_max_percent: float,
        code_context: list,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10),
            )
        )

        use_case.prompt_builder.build.return_value = Prompt(
            prefix="test_prefix",
            suffix="test_suffix",
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                },
                code_context=MetadataExtraInfo(
                    name="code_context",
                    pre=MetadataCodeContent(length=20, length_tokens=4),
                    post=MetadataCodeContent(length=15, length_tokens=3),
                ),
            ),
        )

        actual = await use_case.execute(
            prefix=prefix,
            suffix=suffix,
            file_name=file_name,
            editor_lang=editor_lang,
            stream=stream,
            code_context=code_context,
            context_max_percent=context_max_percent,
        )

        use_case.prompt_builder.add_content.assert_called_with(
            prefix,
            suffix=suffix,
            suffix_reserved_percent=CodeCompletions.SUFFIX_RESERVED_PERCENT,
            context_max_percent=context_max_percent,
            code_context=code_context,
        )

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id
        assert isinstance(
            actual.metadata.tokens_consumption_metadata, TokensConsumptionMetadata
        )
        assert actual.metadata.tokens_consumption_metadata.input_tokens == 4
        assert actual.metadata.tokens_consumption_metadata.output_tokens == 10
        assert actual.metadata.tokens_consumption_metadata.context_tokens_sent == 4
        assert actual.metadata.tokens_consumption_metadata.context_tokens_used == 3

        use_case.model.generate.assert_called_with(
            use_case.prompt_builder.build().prefix,
            use_case.prompt_builder.build().suffix,
            stream,
        )

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
        use_case: CodeCompletions,
        model_chunks: list[TextGenModelChunk],
        expected_chunks: list[str],
    ):
        async def _stream_generator(_prefix, _suffix, _stream):
            for chunk in model_chunks:
                yield chunk

        use_case.model.generate = AsyncMock(side_effect=_stream_generator)

        actual = await use_case.execute(
            prefix="any",
            suffix="how",
            file_name="bar.py",
            editor_lang=LanguageId.PYTHON,
            stream=True,
        )

        chunks = []
        async for content in actual:
            chunks += content

        assert chunks == expected_chunks

        use_case.model.generate.assert_called_with(
            use_case.prompt_builder.build().prefix,
            use_case.prompt_builder.build().suffix,
            True,
        )

    @pytest.mark.parametrize(
        (
            "prompt",
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
            "expected_language_id",
            "expected_output",
        ),
        [
            (
                "prompt_random_prefix",
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                False,
                LanguageId.PYTHON,
                "random_suggestion",
            ),
        ],
    )
    async def test_execute_with_prompt_prepared(
        self,
        use_case: CodeCompletions,
        prompt: str,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        stream: bool,
        expected_language_id: LanguageId,
        expected_output: str,
    ):
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10),
            )
        )

        mock_prompt = Mock(spec=Prompt)
        mock_prompt.prefix = "test_prefix"
        mock_prompt.suffix = "test_suffix"
        mock_prompt.metadata = Mock(spec=MetadataPromptBuilder)
        mock_prompt.metadata.components = {
            "prefix": Mock(spec=MetadataCodeContent, length_tokens=2),
            "suffix": Mock(spec=MetadataCodeContent, length_tokens=2),
        }
        mock_prompt.metadata.code_context = Mock(
            spec=MetadataExtraInfo,
            pre=Mock(spec=MetadataCodeContent, length_tokens=4),
            post=Mock(spec=MetadataCodeContent, length_tokens=3),
        )

        use_case.prompt_builder.wrap.return_value = mock_prompt

        actual = await use_case.execute(
            prefix, suffix, file_name, editor_lang=editor_lang, raw_prompt=prompt
        )

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id
        assert isinstance(
            actual.metadata.tokens_consumption_metadata, TokensConsumptionMetadata
        )
        assert actual.metadata.tokens_consumption_metadata.input_tokens == 4
        assert actual.metadata.tokens_consumption_metadata.output_tokens == 10
        assert actual.metadata.tokens_consumption_metadata.context_tokens_sent == 4
        assert actual.metadata.tokens_consumption_metadata.context_tokens_used == 3

        use_case.model.generate.assert_called_with(
            mock_prompt.prefix,
            mock_prompt.suffix,
            stream,
        )

        use_case.prompt_builder.wrap.assert_called_with(prompt)

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "model_exception_type",
        ),
        [
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                AnthropicAPIStatusError,
            ),
            (
                "random_prefix",
                "random_suffix",
                "file_name",
                "python",
                AnthropicAPIConnectionError,
            ),
        ],
    )
    async def test_execute_processed_exception(
        self,
        use_case: CodeCompletions,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        model_exception_type: Type[ModelAPIError],
    ):
        if issubclass(model_exception_type, AnthropicAPIStatusError):
            model_exception_type.code = 404
        exception = model_exception_type("exception message")

        def _side_effect(*_args, **_kwargs):
            raise exception

        use_case.model.generate = AsyncMock(side_effect=_side_effect)

        with pytest.raises(model_exception_type):
            _ = await use_case.execute(prefix, suffix, file_name, editor_lang)

        code = (
            model_exception_type.code if hasattr(model_exception_type, "code") else -1
        )

        use_case.instrumentator.watcher.register_model_exception.assert_called_with(
            str(exception), code
        )

    async def test_execute_with_post_processor(
        self, completions_with_post_processing: Mock
    ):
        prefix = "def foo"
        suffix = ""
        file_name = "foo.py"
        stream = False
        editor_lang = "python"

        actual = await completions_with_post_processing.execute(
            prefix=prefix,
            suffix=suffix,
            file_name=file_name,
            editor_lang=editor_lang,
            stream=stream,
        )

        mock_post_process = (
            completions_with_post_processing.post_processor.return_value.process
        )
        mock_post_process.assert_called_with("Unprocessed completion output", score=0)

        assert actual.text == "Post-processed completion output"

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "expected_output",
            "expected_language_id",
            "expected_language",
        ),
        [
            # Test with editor_lang provided
            (
                "def hello",
                ":",
                "test.py",
                "python",
                "world()",
                LanguageId.PYTHON,
                "python",
            ),
            # Test with language resolved from filename
            (
                "function test",
                "{",
                "script.js",
                None,
                "return true;",
                LanguageId.JS,
                "javascript",
            ),
            # Test with no language identifiable
            ("some code", "", "noextension", None, None, None, None),
        ],
    )
    async def test_execute_amazon_q_model(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        expected_output: str,
        expected_language_id: LanguageId,
        expected_language: str,
    ):
        # Mock AmazonQModel
        model = Mock(spec=AmazonQModel)
        model.input_token_limit = 16
        if expected_output:
            model.generate = AsyncMock(
                return_value=TextGenModelOutput(
                    text=expected_output,
                    score=0,
                    safety_attributes=SafetyAttributes(),
                    metadata=Mock(output_tokens=10),
                )
            )

        # Create use case with AmazonQModel
        use_case = CodeCompletions(model, Mock(spec=TokenStrategyBase))
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)

        # Mock prompt builder
        use_case.prompt_builder = Mock(spec=PromptBuilderPrefixBased)
        use_case.prompt_builder.build.return_value = Prompt(
            prefix=prefix,
            suffix=suffix,
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                }
            ),
        )

        # Execute the code completion
        actual = await use_case.execute(
            prefix=prefix,
            suffix=suffix,
            file_name=file_name,
            editor_lang=editor_lang,
        )

        # Verify results
        if expected_output:
            assert actual.text == expected_output
            assert actual.lang_id == expected_language_id

            # Verify model.generate was called with correct parameters
            model.generate.assert_called_once_with(
                prefix,
                suffix,
                file_name,
                expected_language.lower() if expected_language else None,
            )
        else:
            assert actual.text == ""
