from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ai_gateway.model_selection.models import CompletionType
from ai_gateway.models.v2.completion_litellm import MODEL_STOP_TOKENS, CompletionLiteLLM


@pytest.fixture(name="message_content")
def message_content_fixture() -> dict:
    return {"prefix": "test", "suffix": ""}


@pytest.fixture(name="messages")
def messages_fixture(message_content: dict) -> list[BaseMessage]:
    return [HumanMessage(content=[message_content])]


@pytest.fixture(name="mock_response_text")
def mock_response_text_fixture() -> str:
    return "test"


@pytest.fixture(name="mock_response")
def mock_response_fixture(mock_response_text: str) -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock(text=mock_response_text)]
    mock.usage = None
    mock.model = "test-model"
    return mock


class TestCompletionLiteLLMInit:
    @pytest.mark.parametrize(
        ("completion_type", "fim_format", "expects_error"),
        [
            (CompletionType.FIM, None, True),
            (
                CompletionType.FIM,
                "</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
                False,
            ),
            (CompletionType.TEXT, None, False),
        ],
    )
    def test_init_completion_types(self, completion_type, fim_format, expects_error):
        if expects_error:
            with pytest.raises(ValueError, match="fim_format is required"):
                CompletionLiteLLM(
                    model="codestral-2501",
                    completion_type=completion_type,
                    fim_format=fim_format,
                )
            return

        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=completion_type,
            fim_format=fim_format,
        )
        assert model.completion_type == completion_type
        assert model.fim_format == fim_format

    def test_init_ignores_extra_kwargs(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            model_keys="ignored",
            client="ignored",
            streaming="ignored",
            model_kwargs="ignored",
        )
        assert model.model == "codestral-2501"


class TestCompletionLiteLLMProperties:
    @pytest.fixture
    def fim_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            temperature=0.32,
            max_tokens=64,
            custom_llm_provider="fireworks_ai",
        )

    def test_default_params(self, fim_model):
        params = fim_model._default_params
        assert params["temperature"] == 0.32
        assert params["max_tokens"] == 64
        assert params["custom_llm_provider"] == "fireworks_ai"
        assert "model" not in params

    def test_default_params_excludes_none(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        params = model._default_params
        assert "temperature" not in params
        assert "max_tokens" not in params

    def test_identifying_params(self, fim_model):
        params = fim_model._identifying_params
        assert params["model"] == "codestral-2501"
        assert params["completion_type"] == "fim"
        assert params["temperature"] == 0.32
        assert params["max_tokens"] == 64

    def test_llm_type(self, fim_model):
        assert fim_model._llm_type == "litellm-completion"


class TestStopTokens:
    @pytest.mark.parametrize(
        ("model_name", "lookup_name", "expected_tokens"),
        [
            (
                "codestral-2501",
                "codestral-2501",
                MODEL_STOP_TOKENS["codestral-2501"],
            ),
            (
                "qwen2p5-coder-7b",
                "qwen2p5-coder-7b",
                MODEL_STOP_TOKENS["qwen2p5-coder-7b"],
            ),
            ("unknown-model", "unknown-model", []),
        ],
    )
    def test_get_stop_tokens(self, model_name, lookup_name, expected_tokens):
        model = CompletionLiteLLM(
            model=model_name,
            completion_type=CompletionType.TEXT,
        )
        tokens = model._get_stop_tokens(lookup_name)
        assert tokens == expected_tokens


class TestFormatFimPrompt:
    @pytest.mark.parametrize(
        ("prefix", "suffix", "expected"),
        [
            (
                "def hello():",
                "\nreturn 42",
                "</s>[SUFFIX]\nreturn 42[PREFIX]def hello():[MIDDLE]",
            ),
            (
                "def hello():",
                "",
                "</s>[SUFFIX][PREFIX]def hello():[MIDDLE]",
            ),
        ],
    )
    def test_format_fim_prompt(self, prefix, suffix, expected):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
        )
        assert model._format_fim_prompt(prefix, suffix) == expected

    def test_format_fim_prompt_without_fim_format(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        with pytest.raises(ValueError, match="fim_format is required"):
            model._format_fim_prompt("prefix", "suffix")


class TestBuildCompletionArgs:
    @pytest.fixture
    def fim_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            temperature=0.32,
            max_tokens=64,
            custom_llm_provider="fireworks_ai",
        )

    @pytest.fixture
    def text_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            temperature=0.32,
            max_tokens=64,
            custom_llm_provider="vertex_ai",
        )

    def test_fim_completion_args(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="formatted prompt",
            suffix=None,
            stop=None,
            stream=False,
        )
        assert args["prompt"] == "formatted prompt"
        assert args["model"] == "codestral-2501"
        assert args["stream"] is False
        assert args["temperature"] == 0.32
        assert args["max_tokens"] == 64
        assert "messages" not in args
        assert "suffix" not in args

    def test_text_completion_args(self, text_model):
        args = text_model._build_completion_args(
            prompt="def hello():",
            suffix="\nreturn 42",
            stop=None,
            stream=False,
        )
        assert args["prompt"] == "def hello():"
        assert args["suffix"] == "\nreturn 42"
        assert "messages" not in args

    def test_text_completion_no_suffix(self, text_model):
        args = text_model._build_completion_args(
            prompt="def hello():",
            suffix=None,
            stop=None,
            stream=False,
        )
        assert "suffix" not in args

    def test_api_base_and_key_from_kwargs(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
            api_base="https://api.fireworks.ai",
            api_key="test-key",
        )
        assert args["api_base"] == "https://api.fireworks.ai"
        assert args["api_key"] == "test-key"

    def test_model_override_from_kwargs(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
            model="accounts/fireworks/models/codestral-2508",
        )
        assert args["model"] == "accounts/fireworks/models/codestral-2508"

    def test_stop_tokens_merged(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=["custom_stop"],
            stream=False,
        )
        assert "custom_stop" in args["stop"]
        assert "\n\n" in args["stop"]

    def test_fireworks_logprobs(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
        )
        assert args["logprobs"] == 1

    def test_fireworks_cache_disabled(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
            using_cache="false",
        )
        assert args["prompt_cache_max_len"] == 0

    def test_fireworks_session_affinity(self, fim_model):
        args = fim_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
            session_id="test-session-123",
        )
        assert args["extra_headers"]["x-session-affinity"] == "test-session-123"

    def test_vertex_ai_location(self, text_model):
        args = text_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
            vertex_ai_location="us-east1",
        )
        assert args["vertex_ai_location"] == "us-east1"

    def test_vertex_ai_default_location(self, text_model):
        args = text_model._build_completion_args(
            prompt="test",
            suffix=None,
            stop=None,
            stream=False,
        )
        assert args["vertex_ai_location"] == "us-central1"


class TestGenerate:
    def test_sync_generate_not_implemented(self, messages: list[BaseMessage]):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        with pytest.raises(
            NotImplementedError, match="Sync generation not implemented"
        ):
            model._generate(messages)


class TestAGenerate:
    @pytest.fixture
    def fim_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            custom_llm_provider="fireworks_ai",
        )

    @pytest.fixture
    def text_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            custom_llm_provider="vertex_ai",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fixture_name", "message_content", "expected_prompt", "expected_suffix"),
        [
            (
                "fim_model",
                {"prefix": "def hello():", "suffix": ""},
                "</s>[SUFFIX][PREFIX]def hello():[MIDDLE]",
                None,
            ),
            (
                "text_model",
                {"prefix": "def hello():", "suffix": "\nreturn 1"},
                "def hello():",
                "\nreturn 1",
            ),
        ],
    )
    async def test_agenerate(
        self,
        request,
        fixture_name,
        messages,
        mock_response,
        mock_response_text,
        expected_prompt,
        expected_suffix,
    ):
        model = request.getfixturevalue(fixture_name)

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response)
        ) as mock_acompletion:
            result = await model._agenerate(messages)

            assert result == ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=mock_response_text,
                            response_metadata={"model_name": "test-model"},
                        )
                    )
                ]
            )

            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["prompt"] == expected_prompt
            if expected_suffix is None:
                assert "suffix" not in call_kwargs
            else:
                assert call_kwargs["suffix"] == expected_suffix


class TestStreamSync:
    def test_sync_stream_not_implemented(self, messages: list[BaseMessage]):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        with pytest.raises(NotImplementedError, match="Sync streaming not implemented"):
            list(model._stream(messages))


class TestAStreamAsync:
    @pytest.fixture
    def fim_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            custom_llm_provider="fireworks_ai",
        )

    @pytest.fixture
    def text_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            custom_llm_provider="vertex_ai",
        )

    @pytest.mark.asyncio
    async def test_astream_fim(self, fim_model, messages: list[BaseMessage]):
        async def mock_response():
            chunks = [
                MagicMock(
                    choices=[MagicMock(text="print")], usage=None, model="test-model"
                ),
                MagicMock(
                    choices=[MagicMock(text="('hello')")],
                    usage=None,
                    model="test-model",
                ),
            ]
            for chunk in chunks:
                yield chunk

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response())
        ):
            chunks = []
            async for chunk in fim_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="print",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="('hello')",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]

    @pytest.mark.asyncio
    async def test_astream_text(self, text_model, messages: list[BaseMessage]):
        async def mock_response():
            chunks = [
                MagicMock(
                    choices=[MagicMock(text="print")], usage=None, model="test-model"
                ),
                MagicMock(
                    choices=[MagicMock(text="('hello')")],
                    usage=None,
                    model="test-model",
                ),
            ]
            for chunk in chunks:
                yield chunk

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response())
        ):
            chunks = []
            async for chunk in text_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="print",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="('hello')",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]

    @pytest.mark.asyncio
    async def test_astream_disabled(
        self,
        fim_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        fim_model.disable_streaming = True

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response)
        ):
            chunks = []
            async for chunk in fim_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=mock_response_text,
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]

    @pytest.mark.asyncio
    async def test_astream_empty_chunks(self, fim_model, messages: list[BaseMessage]):
        async def mock_response():
            chunks = [
                MagicMock(choices=[], usage=None, model="test-model"),
                MagicMock(choices=[MagicMock(text="")], usage=None, model="test-model"),
                MagicMock(
                    choices=[MagicMock(text="valid")], usage=None, model="test-model"
                ),
            ]
            for chunk in chunks:
                yield chunk

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response())
        ):
            chunks = []
            async for chunk in fim_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="valid",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]


class TestExtractText:
    @pytest.mark.parametrize(
        ("completion_type", "fim_format"),
        [
            (CompletionType.FIM, "test"),
            (CompletionType.TEXT, None),
        ],
    )
    def test_extract_text(self, completion_type, fim_format):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=completion_type,
            fim_format=fim_format,
        )
        response = MagicMock()
        response.choices = [MagicMock(text="completion text")]
        assert model._extract_text(response) == "completion text"


class TestExtractChunkText:
    @pytest.mark.parametrize(
        ("completion_type", "fim_format"),
        [
            (CompletionType.FIM, "test"),
            (CompletionType.TEXT, None),
        ],
    )
    def test_extract_chunk_text(self, completion_type, fim_format):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=completion_type,
            fim_format=fim_format,
        )
        chunk = MagicMock()
        chunk.choices = [MagicMock(text="chunk text")]
        assert model._extract_chunk_text(chunk) == "chunk text"

    def test_extract_chunk_text_empty_choices(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        chunk = MagicMock()
        chunk.choices = []
        assert model._extract_chunk_text(chunk) == ""

    def test_extract_chunk_text_no_choices_attr(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
        )
        chunk = MagicMock(spec=[])
        assert model._extract_chunk_text(chunk) == ""

    def test_extract_chunk_text_none_text(self):
        model = CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="test",
        )
        chunk = MagicMock()
        chunk.choices = [MagicMock(text=None)]
        assert model._extract_chunk_text(chunk) == ""


class TestUsageMetadata:
    @pytest.fixture
    def fim_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            custom_llm_provider="fireworks_ai",
        )

    @pytest.fixture
    def text_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            custom_llm_provider="vertex_ai",
        )

    @pytest.mark.asyncio
    async def test_agenerate_with_usage_metadata(
        self,
        fim_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        """Verify usage metadata is extracted and included in agenerate response."""
        mock_response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response)
        ):
            result = await fim_model._agenerate(messages)

            assert result == ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=mock_response_text,
                            response_metadata={"model_name": "test-model"},
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        )
                    )
                ]
            )

    @pytest.mark.asyncio
    async def test_astream_with_usage_metadata(
        self, text_model, messages: list[BaseMessage]
    ):
        """Verify usage metadata is extracted in streaming responses."""

        async def mock_response():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock(text="hello")]
            chunk1.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            }
            chunk1.model = "test-model"
            yield chunk1

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock(text=" world")]
            chunk2.usage = None
            chunk2.model = "test-model"
            yield chunk2

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response())
        ):
            chunks = []
            async for chunk in text_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="hello",
                        response_metadata={"model_name": "test-model"},
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "total_tokens": 11,
                        },
                    )
                ),
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=" world",
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]

    @pytest.mark.asyncio
    async def test_astream_disabled_with_usage_metadata(
        self,
        fim_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        """Verify usage metadata is preserved when streaming is disabled."""
        fim_model.disable_streaming = True

        mock_response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        with patch(
            "litellm.atext_completion", new=AsyncMock(return_value=mock_response)
        ):
            chunks = []
            async for chunk in fim_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=mock_response_text,
                        response_metadata={"model_name": "test-model"},
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                ),
            ]


class TestFireworksRetry:
    @pytest.fixture
    def fireworks_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
            custom_llm_provider="fireworks_ai",
        )

    @pytest.fixture
    def vertex_model(self):
        return CompletionLiteLLM(
            model="codestral-2501",
            completion_type=CompletionType.TEXT,
            custom_llm_provider="vertex_ai",
        )

    @pytest.mark.asyncio
    async def test_fireworks_503_retries(
        self,
        fireworks_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        """Verify 503 errors trigger retries for Fireworks."""
        call_count = 0

        async def mock_acompletion(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise litellm.ServiceUnavailableError(
                    message="Service temporarily unavailable",
                    llm_provider="fireworks_ai",
                    model="codestral-2501",
                )
            return mock_response

        with patch(
            "litellm.atext_completion", new=AsyncMock(side_effect=mock_acompletion)
        ):
            result = await fireworks_model._agenerate(messages)

            assert call_count == 3
            assert result == ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=mock_response_text,
                            response_metadata={"model_name": "test-model"},
                        )
                    )
                ]
            )

    @pytest.mark.asyncio
    async def test_non_fireworks_503_fails_immediately(
        self, vertex_model, messages: list[BaseMessage]
    ):
        """Verify 503 errors do NOT trigger retries for non-Fireworks providers."""
        call_count = 0

        async def mock_acompletion(**_kwargs):
            nonlocal call_count
            call_count += 1
            raise litellm.ServiceUnavailableError(
                message="Service temporarily unavailable",
                llm_provider="vertex_ai",
                model="codestral-2501",
            )

        with patch(
            "litellm.atext_completion", new=AsyncMock(side_effect=mock_acompletion)
        ):
            with pytest.raises(litellm.ServiceUnavailableError):
                await vertex_model._agenerate(messages)

            assert call_count == 1

    @pytest.mark.asyncio
    async def test_fireworks_stream_503_retries(
        self,
        fireworks_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        """Verify 503 errors trigger retries for Fireworks streaming."""
        call_count = 0

        async def mock_acompletion(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise litellm.ServiceUnavailableError(
                    message="Service temporarily unavailable",
                    llm_provider="fireworks_ai",
                    model="codestral-2501",
                )

            async def mock_stream():
                yield mock_response

            return mock_stream()

        with patch(
            "litellm.atext_completion", new=AsyncMock(side_effect=mock_acompletion)
        ):
            chunks = []
            async for chunk in fireworks_model._astream(messages):
                chunks.append(chunk)

            assert chunks == [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=mock_response_text,
                        response_metadata={"model_name": "test-model"},
                    )
                ),
            ]

    @pytest.mark.asyncio
    async def test_fireworks_rate_limit_retries(
        self,
        fireworks_model,
        messages: list[BaseMessage],
        mock_response: MagicMock,
        mock_response_text: str,
    ):
        """Verify rate limit errors trigger retries for Fireworks."""
        call_count = 0

        async def mock_acompletion(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise litellm.RateLimitError(
                    message="Rate limit exceeded",
                    llm_provider="fireworks_ai",
                    model="codestral-2501",
                )
            return mock_response

        with patch(
            "litellm.atext_completion", new=AsyncMock(side_effect=mock_acompletion)
        ):
            result = await fireworks_model._agenerate(messages)

            assert call_count == 2
            assert result == ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=mock_response_text,
                            response_metadata={"model_name": "test-model"},
                        )
                    )
                ]
            )
