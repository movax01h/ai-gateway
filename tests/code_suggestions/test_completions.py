# pylint: disable=too-many-lines

from contextlib import contextmanager
from typing import Any, AsyncIterator, Type, cast
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.code_suggestions import CodeCompletions
from ai_gateway.code_suggestions.base import CodeSuggestionsChunk, CodeSuggestionsOutput
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
from ai_gateway.instrumentators import TextGenModelInstrumentator
from ai_gateway.models import (
    AnthropicAPIConnectionError,
    AnthropicAPIStatusError,
    ModelAPIError,
)
from ai_gateway.models.agent_model import AgentModel
from ai_gateway.models.amazon_q import AmazonQModel
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.models.base_text import (
    TextGenModelBase,
    TextGenModelChunk,
    TextGenModelOutput,
)
from ai_gateway.safety_attributes import SafetyAttributes
from lib.billing_events import BillingEvent, BillingEventsClient


class InstrumentorMock(Mock):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.watcher = Mock()

    @contextmanager
    def watch(self, _prompt: str, **_kwargs: Any):
        yield self.watcher


@pytest.mark.asyncio
class TestCodeCompletions:
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

        tokenization_strategy = Mock(spec=TokenStrategyBase)
        tokenization_strategy.estimate_length = Mock(
            return_value=[10, 0]
        )  # Return subscriptable list

        use_case = CodeCompletions(model, tokenization_strategy)
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
        use_case.prompt_builder = Mock(spec=PromptBuilderPrefixBased)

        yield use_case

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

        use_case = CodeCompletions(
            model=model,
            tokenization_strategy=Mock(spec=TokenStrategyBase),
            billing_event_client=mock_billing_client,
        )
        use_case.instrumentator = InstrumentorMock(spec=TextGenModelInstrumentator)
        use_case.prompt_builder = Mock()
        use_case.prompt_builder.build.return_value = Prompt(
            prefix="test_prefix",
            suffix="test_suffix",
            metadata=MetadataPromptBuilder(
                components={
                    "prefix": MetadataCodeContent(length=10, length_tokens=2),
                    "suffix": MetadataCodeContent(length=10, length_tokens=2),
                }
            ),
        )

        return use_case

    @pytest.fixture(name="completions_with_post_processing", scope="class")
    def completions_with_post_processing_fixture(self):
        model = Mock(
            spec=TextGenModelBase,
            metadata=Mock(name="text-completion-openai/test-model"),
        )
        type(model).input_token_limit = PropertyMock(return_value=2_048)
        model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="Unprocessed completion output",
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10, spec_set=["output_tokens"]),
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
        mock_generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10, max_output_tokens_used=True),
            )
        )

        mock_prompt = Prompt(
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
        mock_build = Mock(return_value=mock_prompt)
        mock_add_content = Mock()

        with (
            patch.object(use_case.model, "generate", mock_generate),
            patch.object(use_case.prompt_builder, "build", mock_build),
            patch.object(use_case.prompt_builder, "add_content", mock_add_content),
        ):
            actual = await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
                stream=stream,
                code_context=code_context,
                context_max_percent=context_max_percent,
            )

        mock_add_content.assert_called_with(
            prefix,
            suffix=suffix,
            suffix_reserved_percent=CodeCompletions.SUFFIX_RESERVED_PERCENT,
            context_max_percent=context_max_percent,
            code_context=code_context,
        )

        actual = cast(CodeSuggestionsOutput, actual)

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id
        assert actual.metadata is not None
        assert isinstance(
            actual.metadata.tokens_consumption_metadata, TokensConsumptionMetadata
        )
        assert actual.metadata.tokens_consumption_metadata.input_tokens == 4
        assert actual.metadata.tokens_consumption_metadata.output_tokens == 10
        assert (
            actual.metadata.tokens_consumption_metadata.max_output_tokens_used is True
        )
        assert actual.metadata.tokens_consumption_metadata.context_tokens_sent == 4
        assert actual.metadata.tokens_consumption_metadata.context_tokens_used == 3

        mock_generate.assert_called_with(
            mock_prompt.prefix,
            mock_prompt.suffix,
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

        mock_generate = AsyncMock(side_effect=_stream_generator)

        with patch.object(use_case.model, "generate", mock_generate):
            actual = await use_case.execute(
                prefix="any",
                suffix="how",
                file_name="bar.py",
                editor_lang="python",
                stream=True,
            )

            chunks = []
            actual_stream = cast(AsyncIterator[CodeSuggestionsChunk], actual)
            async for content in actual_stream:
                chunks.append(content.text)

            assert chunks == expected_chunks

            mock_generate.assert_called_with(
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
        mock_generate = AsyncMock(
            return_value=TextGenModelOutput(
                text=expected_output,
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10, spec_set=["output_tokens"]),
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

        mock_wrap = Mock(return_value=mock_prompt)

        with (
            patch.object(use_case.model, "generate", mock_generate),
            patch.object(use_case.prompt_builder, "wrap", mock_wrap),
        ):

            actual = await use_case.execute(
                prefix, suffix, file_name, editor_lang=editor_lang, raw_prompt=prompt
            )

        # Since stream=False, actual should be CodeSuggestionsOutput, not AsyncIterator
        actual = cast(CodeSuggestionsOutput, actual)

        assert expected_output == actual.text
        assert expected_language_id == actual.lang_id
        assert actual.metadata is not None
        assert isinstance(
            actual.metadata.tokens_consumption_metadata, TokensConsumptionMetadata
        )
        assert actual.metadata.tokens_consumption_metadata.input_tokens == 4
        assert actual.metadata.tokens_consumption_metadata.output_tokens == 10
        assert (
            actual.metadata.tokens_consumption_metadata.max_output_tokens_used is False
        )
        assert actual.metadata.tokens_consumption_metadata.context_tokens_sent == 4
        assert actual.metadata.tokens_consumption_metadata.context_tokens_used == 3

        mock_generate.assert_called_with(
            mock_prompt.prefix,
            mock_prompt.suffix,
            stream,
        )

        mock_wrap.assert_called_with(prompt)

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

        mock_generate = AsyncMock(side_effect=_side_effect)

        mock_context_manager = Mock()
        mock_context_manager.register_model_exception = Mock()
        mock_enter = Mock(return_value=mock_context_manager)
        mock_exit = Mock(return_value=None)
        mock_watch = Mock()
        mock_watch.return_value.__enter__ = mock_enter
        mock_watch.return_value.__exit__ = mock_exit

        with (
            patch.object(use_case.model, "generate", mock_generate),
            patch.object(use_case.instrumentator, "watch", mock_watch),
        ):

            with pytest.raises(model_exception_type):
                _ = await use_case.execute(prefix, suffix, file_name, editor_lang)

            code = (
                model_exception_type.code
                if hasattr(model_exception_type, "code")
                else -1
            )

            mock_context_manager.register_model_exception.assert_called_with(
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
        mock_post_process.assert_called_with(
            "Unprocessed completion output",
            score=0,
            max_output_tokens_used=False,
            model_name=completions_with_post_processing.model.metadata.name,
        )

        assert actual.text == "Post-processed completion output"

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "file_name",
            "editor_lang",
            "stream",
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
                False,
                "world()",
                LanguageId.PYTHON,
                "python",
            ),
            # Test with streaming
            (
                "def hello",
                ":",
                "test.py",
                "python",
                True,
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
                False,
                "return true;",
                LanguageId.JS,
                "javascript",
            ),
            # Test with no language identifiable
            ("some code", "", "noextension", False, None, None, None, None),
        ],
    )
    async def test_execute_amazon_q_model(
        self,
        prefix: str,
        suffix: str,
        file_name: str,
        editor_lang: str,
        stream: bool,
        expected_output: str,
        expected_language_id: LanguageId,
        expected_language: str,
    ):
        # Mock AmazonQModel
        model = Mock(spec=AmazonQModel)
        model.input_token_limit = 16
        if expected_output:
            if stream:

                async def _stream_generator(*_args, **_kwargs):
                    yield TextGenModelChunk(text=expected_output)

                model.generate = AsyncMock(side_effect=_stream_generator)
            else:
                model.generate = AsyncMock(
                    return_value=TextGenModelOutput(
                        text=expected_output,
                        score=0,
                        safety_attributes=SafetyAttributes(),
                        metadata=Mock(output_tokens=10, spec_set=["output_tokens"]),
                    )
                )

        # Create use case with AmazonQModel
        tokenization_strategy = Mock(spec=TokenStrategyBase)
        tokenization_strategy.estimate_length = Mock(return_value=[10, 0])
        use_case = CodeCompletions(model, tokenization_strategy)
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
            stream=stream,
        )

        # Verify results
        if expected_output:
            if stream:
                chunks = []
                actual_stream = cast(AsyncIterator[CodeSuggestionsChunk], actual)
                async for content in actual_stream:
                    chunks.append(content.text)

                assert chunks == [expected_output]
            else:
                actual = cast(CodeSuggestionsOutput, actual)
                assert actual.text == expected_output
                assert actual.lang_id == expected_language_id

            # Verify model.generate was called with correct parameters
            model.generate.assert_called_once_with(
                prefix,
                suffix,
                file_name,
                expected_language.lower() if expected_language else None,
                stream,
            )
        else:
            actual = cast(CodeSuggestionsOutput, actual)
            assert actual.text == ""

    async def test_execute_agent_model(self):
        prefix = "def hello"
        suffix = ":"
        file_name = "test.py"
        editor_lang = "python"
        stream = False

        agent_model = Mock(spec=AgentModel)
        agent_model.input_token_limit = 16
        agent_model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="world()",
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=10, spec_set=["output_tokens"]),
            )
        )

        use_case = CodeCompletions(agent_model, Mock(spec=TokenStrategyBase))
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
            stream=stream,
        )

        actual = cast(CodeSuggestionsOutput, actual)
        assert actual.text == "world()"
        assert actual.lang_id == LanguageId.PYTHON

        # Verify model.generate was called with correct parameters
        agent_model.generate.assert_called_once_with(
            {
                "prefix": "def hello",
                "suffix": ":",
                "file_name": "test.py",
                "language": "python",
            },
            False,
        )

    async def test_billing_event_tracked_on_successful_completion(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event is tracked when code completion is successful."""
        expected_output_tokens = 25

        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="generated code",
                score=0.8,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(
                    output_tokens=expected_output_tokens, max_output_tokens_used=False
                ),
            )
        )

        await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
            file_name="test.py",
            editor_lang="python",
            user=mock_user,
        )

        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
            category="CodeCompletions",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_completions",
                "llm_operations": [
                    {
                        "model_id": "test-model-id",
                        "completion_tokens": expected_output_tokens,
                    }
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
                metadata=Mock(output_tokens=25, max_output_tokens_used=False),
            )
        )

        await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
            file_name="test.py",
            editor_lang="python",
            # No user provided
        )

        mock_billing_client.track_billing_event.assert_not_called()

    async def test_billing_event_exception_handling(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event exceptions are handled gracefully."""
        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="generated code",
                score=0.8,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=25, max_output_tokens_used=False),
            )
        )

        # Make billing client raise an exception
        mock_billing_client.track_billing_event.side_effect = Exception(
            "Billing service unavailable"
        )

        # This should not raise an exception - billing errors should be handled gracefully
        result = await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
            file_name="test.py",
            editor_lang="python",
            user=mock_user,
        )

        # Verify the completion still works despite billing error
        assert result.text == "generated code"
        mock_billing_client.track_billing_event.assert_called_once()

    async def test_billing_event_with_zero_tokens(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event is tracked when output tokens is zero."""
        use_case_with_billing.model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="",
                score=0.0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=0, max_output_tokens_used=False),
            )
        )

        await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
            file_name="test.py",
            editor_lang="python",
            user=mock_user,
        )

        # Should still track billing event even with 0 tokens for consistency
        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
            category="CodeCompletions",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_completions",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 0}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )

    async def test_billing_event_tracked_for_streaming(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing events are tracked for streaming responses."""

        async def _stream_generator(_prefix, _suffix, _stream):
            yield TextGenModelChunk(text="hello ")
            yield TextGenModelChunk(text="world!")

        use_case_with_billing.model.generate = AsyncMock(side_effect=_stream_generator)

        # Mock tokenization strategy to return expected token count
        use_case_with_billing.tokenization_strategy.estimate_length = Mock(
            return_value=[12, 0]  # 12 tokens for "hello world!"
        )

        result = await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
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

        # Billing events should be tracked for streaming responses
        mock_billing_client.track_billing_event.assert_called_once_with(
            user=mock_user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
            category="CodeCompletions",
            unit_of_measure="request",
            quantity=1,
            metadata={
                "execution_environment": "code_completions",
                "llm_operations": [
                    {"model_id": "test-model-id", "completion_tokens": 12}
                ],
                "feature_qualified_name": "code_suggestions",
                "feature_ai_catalog_item": False,
            },
        )

    async def test_billing_event_exception_handling_streaming(
        self, use_case_with_billing, mock_user, mock_billing_client
    ):
        """Test that billing event exceptions are handled gracefully for streaming."""

        async def _stream_generator(_prefix, _suffix, _stream):
            yield TextGenModelChunk(text="hello ")
            yield TextGenModelChunk(text="world!")

        use_case_with_billing.model.generate = AsyncMock(side_effect=_stream_generator)

        # Mock tokenization strategy
        use_case_with_billing.tokenization_strategy.estimate_length = Mock(
            return_value=[12, 0]
        )

        # Make billing client raise an exception
        mock_billing_client.track_billing_event.side_effect = Exception(
            "Billing service unavailable"
        )

        result = await use_case_with_billing.execute(
            prefix="def hello",
            suffix=":",
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

    async def test_execute_agent_model_with_unknown_language_should_not_crash(self):
        """Test that AgentModel handles the case where both editor_lang is None and resolve_lang_name returns None
        without crashing.

        This test reproduces the production error:
        AttributeError: 'NoneType' object has no attribute 'lower'
        """
        prefix = "some code"
        suffix = ""
        file_name = "unknown_file_without_extension"  # File with no extension
        editor_lang = None  # No editor language provided
        stream = False

        agent_model = Mock(spec=AgentModel)
        agent_model.input_token_limit = 16
        agent_model.generate = AsyncMock(
            return_value=TextGenModelOutput(
                text="completion",
                score=0,
                safety_attributes=SafetyAttributes(),
                metadata=Mock(output_tokens=5, spec_set=["output_tokens"]),
            )
        )

        use_case = CodeCompletions(agent_model, Mock(spec=TokenStrategyBase))
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

        # Mock resolve_lang_name to return None (unknown file extension)
        with patch(
            "ai_gateway.code_suggestions.completions.resolve_lang_name",
            return_value=None,
        ):
            # This should not crash with AttributeError: 'NoneType' object has no attribute 'lower'
            # Instead, it should handle the None case gracefully
            actual = await use_case.execute(
                prefix=prefix,
                suffix=suffix,
                file_name=file_name,
                editor_lang=editor_lang,
                stream=stream,
            )

            # Should return empty result when language cannot be determined
            actual = cast(CodeSuggestionsOutput, actual)

            assert actual.text == ""
            assert actual.score == 0
            assert actual.lang_id is None

            # Model.generate should not be called when language is unknown
            agent_model.generate.assert_not_called()
