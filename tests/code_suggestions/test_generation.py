from contextlib import contextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, call, patch

import pytest
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

    @pytest.fixture(scope="class")
    def use_case(self):
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

    @pytest.fixture(scope="class")
    def use_case_q(self):
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
        use_case.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="output", score=0, safety_attributes=SafetyAttributes()
            )
        )
        with patch.object(expected_post_processor, "process") as mock:
            _ = await use_case.execute(
                "prefix",
                "test.py",
                editor_lang="Python",
                raw_prompt="test prompt",
                model_provider=model_provider,
            )

            mock.assert_called()

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

        use_case.model.generate = AsyncMock(side_effect=_stream_generator)

        actual = await use_case.execute(
            prefix="any",
            file_name="bar.py",
            editor_lang="Python",
            model_provider=ModelProvider.ANTHROPIC,
            stream=True,
        )

        chunks = []
        async for content in actual:
            chunks += content

        assert chunks == expected_chunks

        use_case.model.generate.assert_called_with(
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

        with patch.object(use_case, "tokenization_strategy") as mock, patch.object(
            use_case, "snowplow_instrumentator"
        ) as snowplow_mock:
            mock.estimate_length = Mock(return_value=[4, 5])

            if stream:
                use_case.model.generate = AsyncMock(side_effect=_stream_generator)
            else:
                use_case.model.generate = AsyncMock(
                    return_value=TextGenModelOutput(
                        text="output", score=0, safety_attributes=SafetyAttributes()
                    )
                )

            actual = await use_case.execute(
                prefix="any",
                file_name="bar.py",
                editor_lang="Python",
                model_provider=ModelProvider.ANTHROPIC,
                stream=stream,
                snowplow_event_context=snowplow_event_context,
            )

            if stream:
                async for _ in actual:
                    pass

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
        editor_lang: str | None,
        expected_lang: str,
        expected_output: str,
        additional_kwargs: dict,
    ):
        """Test generation with AmazonQModel with various input combinations."""
        use_case_q.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output, score=0, safety_attributes=SafetyAttributes()
            )
        )

        result = await use_case_q.execute(
            prefix=prefix,
            suffix=suffix,
            file_name=file_name,
            editor_lang=editor_lang,
            **additional_kwargs,
        )

        assert result is not None
        assert result.text == expected_output

        # Verify model.generate was called with correct parameters
        use_case_q.model.generate.assert_called_once_with(
            prefix, suffix, file_name, expected_lang, **additional_kwargs
        )
