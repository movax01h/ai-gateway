# pylint: disable=too-many-lines
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from litellm.exceptions import APIConnectionError, InternalServerError

from ai_gateway.models import KindLiteLlmModel, LiteLlmChatModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.models.base_text import TextGenModelOutput
from ai_gateway.models.litellm import (
    LiteLlmAPIConnectionError,
    LiteLlmInternalServerError,
    LiteLlmTextGenModel,
)
from ai_gateway.tracking import SnowplowEventContext


class TestKindLiteLlmModel:
    def test_chat_model(self):
        assert KindLiteLlmModel.MISTRAL.chat_model() == "custom_openai/mistral"
        assert KindLiteLlmModel.CODESTRAL.chat_model() == "custom_openai/codestral"
        assert KindLiteLlmModel.CODEGEMMA.chat_model() == "custom_openai/codegemma"
        assert KindLiteLlmModel.CODELLAMA.chat_model() == "custom_openai/codellama"
        assert (
            KindLiteLlmModel.DEEPSEEKCODER.chat_model() == "custom_openai/deepseekcoder"
        )
        assert (
            KindLiteLlmModel.CODESTRAL.chat_model(provider=KindModelProvider.MISTRALAI)
            == "codestral/codestral"
        )

    def test_text_model(self):
        assert (
            KindLiteLlmModel.CODEGEMMA.text_model()
            == "text-completion-custom_openai/codegemma"
        )
        assert (
            KindLiteLlmModel.CODESTRAL.text_model()
            == "text-completion-custom_openai/codestral"
        )
        assert (
            KindLiteLlmModel.CODESTRAL.text_model(provider=KindModelProvider.MISTRALAI)
            == "text-completion-codestral/codestral"
        )


class TestLiteLlmChatModel:
    @pytest.fixture
    def endpoint(self):
        return "http://127.0.0.1:1111/v1"

    @pytest.fixture
    def api_key(self):
        return "specified-api-key"

    @pytest.fixture
    def identifier(self):
        return "provider/some-cool-model"

    @pytest.fixture
    def lite_llm_chat_model(self, endpoint, api_key, identifier):
        return LiteLlmChatModel.from_model_name(
            name="mistral",
            endpoint=endpoint,
            api_key=api_key,
            identifier=identifier,
            custom_models_enabled=True,
            disable_streaming=False,
        )

    @pytest.mark.parametrize(
        ("model_name", "expected_limit"),
        [
            (KindLiteLlmModel.CODEGEMMA, 8_192),
            (KindLiteLlmModel.CODELLAMA, 16_384),
            (KindLiteLlmModel.CODESTRAL, 32_768),
        ],
    )
    def test_max_model_len(self, model_name: str, expected_limit: int):
        model = LiteLlmChatModel.from_model_name(name=model_name)
        assert model.input_token_limit == expected_limit

    @pytest.mark.parametrize(
        (
            "model_name",
            "api_key",
            "identifier",
            "provider",
            "custom_models_enabled",
            "provider_keys",
            "provider_endpoints",
            "expected_name",
            "expected_api_key",
            "expected_engine",
            "expected_endpoint",
            "expected_identifier",
        ),
        [
            (
                "mistral",
                "",
                "provider/some-cool-model",
                KindModelProvider.LITELLM,
                True,
                {},
                {},
                "custom_openai/mistral",
                "stubbed-api-key",
                "litellm",
                "http://127.0.0.1:1111/v1",
                "provider/some-cool-model",
            ),
            (
                "codestral",
                "",
                "",
                KindModelProvider.MISTRALAI,
                True,
                {},
                {},
                "codestral/codestral",
                "stubbed-api-key",
                "codestral",
                "http://127.0.0.1:1111/v1",
                "",
            ),
            (
                "codestral",
                None,
                "",
                KindModelProvider.MISTRALAI,
                True,
                {"mistral_api_key": "stubbed-api-key"},
                {},
                "codestral/codestral",
                "stubbed-api-key",
                "codestral",
                "http://127.0.0.1:1111/v1",
                "",
            ),
            (
                "qwen2p5-coder-7b",
                "",
                None,
                KindModelProvider.FIREWORKS,
                True,
                {"fireworks_api_key": "stubbed-api-key"},
                {
                    "fireworks_current_region_endpoint": {
                        "endpoint": "https://fireworks.endpoint",
                        "identifier": "provider/some-cool-model#deployment_id",
                    }
                },
                "fireworks_ai/qwen2p5-coder-7b",
                "stubbed-api-key",
                "fireworks_ai",
                "https://fireworks.endpoint",
                "fireworks_ai/provider/some-cool-model#deployment_id",
            ),
        ],
    )
    def test_from_model_name(
        self,
        model_name: str,
        api_key: Optional[str],
        identifier: Optional[str],
        provider: KindModelProvider,
        custom_models_enabled: bool,
        provider_keys: dict,
        provider_endpoints: dict,
        expected_name: str,
        expected_api_key: str,
        expected_engine: str,
        expected_endpoint: str,
        expected_identifier: str,
        endpoint,
    ):
        model = LiteLlmChatModel.from_model_name(
            name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            custom_models_enabled=custom_models_enabled,
            provider=provider,
            identifier=identifier,
            provider_keys=provider_keys,
            provider_endpoints=provider_endpoints,
        )

        assert model.metadata.name == expected_name
        assert model.metadata.endpoint == expected_endpoint
        assert model.metadata.api_key == expected_api_key
        assert model.metadata.engine == expected_engine
        assert model.metadata.identifier == expected_identifier

        model = LiteLlmChatModel.from_model_name(name=model_name, api_key=None)

        assert model.metadata.endpoint is None

        if provider == KindModelProvider.LITELLM:
            with pytest.raises(ValueError) as exc:
                LiteLlmChatModel.from_model_name(name=model_name, endpoint=endpoint)
            assert str(exc.value) == "specifying custom models endpoint is disabled"

            with pytest.raises(ValueError) as exc:
                LiteLlmChatModel.from_model_name(name=model_name, api_key="api-key")
            assert str(exc.value) == "specifying custom models endpoint is disabled"

    @pytest.mark.asyncio
    async def test_generate(self, lite_llm_chat_model):
        expected_messages = [{"role": "user", "content": "Test message"}]

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))],
                usage=AsyncMock(completion_tokens=999),
            )
            messages = [Message(content="Test message", role="user")]
            output = await lite_llm_chat_model.generate(messages)
            assert isinstance(output, TextGenModelOutput)
            assert output.text == "Test response"
            assert output.metadata.output_tokens == 999

            mock_acompletion.assert_called_with(
                messages=expected_messages,
                stream=False,
                temperature=0.2,
                top_p=0.95,
                max_tokens=2048,
                timeout=30.0,
                stop=["</new_code>"],
                api_base="http://127.0.0.1:1111/v1",
                api_key="specified-api-key",
                model="some-cool-model",
                custom_llm_provider="provider",
            )

    @pytest.mark.asyncio
    async def test_override_stream(self, endpoint, api_key, identifier):
        chat_model = LiteLlmChatModel.from_model_name(
            name="mistral",
            endpoint=endpoint,
            api_key=api_key,
            identifier=identifier,
            custom_models_enabled=True,
            disable_streaming=True,
        )

        expected_messages = [{"role": "user", "content": "Test message"}]

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))],
                usage=AsyncMock(completion_tokens=999),
            )

            messages = [Message(content="Test message", role=Role.USER)]
            output = await chat_model.generate(
                messages=messages,
                stream=True,
            )

            mock_acompletion.assert_called_with(
                messages=expected_messages,
                stream=False,
                temperature=0.2,
                top_p=0.95,
                max_tokens=2048,
                timeout=30.0,
                stop=["</new_code>"],
                api_base="http://127.0.0.1:1111/v1",
                api_key="specified-api-key",
                model="some-cool-model",
                custom_llm_provider="provider",
            )

            assert isinstance(output, TextGenModelOutput)
            assert output.text == "Test response"

    @pytest.mark.asyncio
    async def test_generate_stream(self, lite_llm_chat_model):
        expected_messages = [{"role": "user", "content": "Test message"}]

        streamed_response = AsyncMock()
        streamed_response.__aiter__.return_value = iter(
            [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                )
            ]
        )

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion, patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            mock_acompletion.return_value = streamed_response

            messages = [Message(content="Test message", role="user")]
            response = await lite_llm_chat_model.generate(
                messages=messages,
                stream=True,
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=1024,
            )

            content = []
            async for chunk in response:
                content.append(chunk.text)
            assert content == ["Streamed content"]

            mock_acompletion.assert_called_with(
                messages=expected_messages,
                stream=True,
                temperature=0.3,
                top_p=0.9,
                max_tokens=1024,
                timeout=30.0,
                stop=["</new_code>"],
                api_base="http://127.0.0.1:1111/v1",
                api_key="specified-api-key",
                model="some-cool-model",
                custom_llm_provider="provider",
            )

            mock_watch.assert_called_once_with(stream=True)
            watcher.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_instrumented(self, lite_llm_chat_model):
        async def mock_stream(*args, **kwargs):
            completions = [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                ),
                "break here",
            ]
            for item in completions:
                if item == "break here":
                    raise ValueError("broken")
                yield item

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion, patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            mock_acompletion.side_effect = AsyncMock(side_effect=mock_stream)

            messages = [Message(content="Test message", role="user")]
            response = await lite_llm_chat_model.generate(
                messages=messages, stream=True
            )

            watcher.finish.assert_not_called()

            with pytest.raises(ValueError):
                _ = [item async for item in response]

            mock_watch.assert_called_once_with(stream=True)
            watcher.register_error.assert_called_once()
            watcher.finish.assert_called_once()


class TestLiteLlmTextGenModel:
    @pytest.fixture
    def endpoint(self):
        return "http://127.0.0.1:4000"

    @pytest.fixture
    def api_key(self):
        return "specified-api-key"

    @pytest.fixture
    def provider_keys(self):
        return {
            "mistral_api_key": "codestral-api-key",
            "fireworks_api_key": "fireworks-api-key",
        }

    @pytest.fixture
    def provider_endpoints(self):
        return {
            "fireworks_current_region_endpoint": {
                "endpoint": "https://fireworks.endpoint",
                "identifier": "provider/some-cool-model#deployment_id",
            }
        }

    @pytest.fixture
    def lite_llm_text_model(self, endpoint, api_key):
        return LiteLlmTextGenModel.from_model_name(
            name="codegemma",
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=True,
        )

    @pytest.mark.parametrize(
        ("model_name", "expected_limit"),
        [
            (KindLiteLlmModel.CODEGEMMA, 8_192),
            (KindLiteLlmModel.CODELLAMA, 16_384),
            (KindLiteLlmModel.CODESTRAL, 32_768),
        ],
    )
    def test_max_model_len(self, model_name: str, expected_limit: int):
        model = LiteLlmTextGenModel.from_model_name(name=model_name)
        assert model.input_token_limit == expected_limit

    @pytest.mark.parametrize(
        (
            "model_name",
            "api_key",
            "identifier",
            "provider",
            "custom_models_enabled",
            "provider_keys",
            "provider_endpoints",
            "expected_name",
            "expected_api_key",
            "expected_engine",
            "expected_endpoint",
            "expected_identifier",
        ),
        [
            (
                "codegemma",
                "a-key",
                "provider/some-cool-model",
                KindModelProvider.LITELLM,
                True,
                {},
                {},
                "text-completion-custom_openai/codegemma",
                "a-key",
                "litellm",
                "http://127.0.0.1:4000",
                "provider/some-cool-model",
            ),
            (
                "codegemma",
                None,
                None,
                KindModelProvider.LITELLM,
                True,
                {},
                {},
                "text-completion-custom_openai/codegemma",
                "stubbed-api-key",
                "litellm",
                "http://127.0.0.1:4000",
                None,
            ),
            (
                "codestral",
                None,
                None,
                KindModelProvider.MISTRALAI,
                True,
                {},
                {},
                "text-completion-codestral/codestral",
                "stubbed-api-key",
                "codestral",
                "http://127.0.0.1:4000",
                None,
            ),
            (
                "codestral",
                "",
                None,
                KindModelProvider.MISTRALAI,
                True,
                {"mistral_api_key": "stubbed-api-key"},
                {},
                "text-completion-codestral/codestral",
                "stubbed-api-key",
                "codestral",
                "http://127.0.0.1:4000",
                None,
            ),
            (
                "qwen2p5-coder-7b",
                "",
                None,
                KindModelProvider.FIREWORKS,
                True,
                {"fireworks_api_key": "stubbed-api-key"},
                {
                    "fireworks_current_region_endpoint": {
                        "endpoint": "https://fireworks.endpoint",
                        "identifier": "provider/some-cool-model#deployment_id",
                    }
                },
                "text-completion-fireworks_ai/qwen2p5-coder-7b",
                "stubbed-api-key",
                "fireworks_ai",
                "https://fireworks.endpoint",
                "text-completion-openai/provider/some-cool-model#deployment_id",
            ),
        ],
    )
    def test_from_model_name(
        self,
        model_name: str,
        api_key: Optional[str],
        identifier: Optional[str],
        provider: KindModelProvider,
        custom_models_enabled: bool,
        provider_keys: dict,
        provider_endpoints: dict,
        expected_name: str,
        expected_api_key: str,
        expected_engine: str,
        expected_endpoint: str,
        expected_identifier: str,
        endpoint,
    ):
        model = LiteLlmTextGenModel.from_model_name(
            name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            custom_models_enabled=custom_models_enabled,
            provider=provider,
            identifier=identifier,
            provider_keys=provider_keys,
            provider_endpoints=provider_endpoints,
        )

        assert model.metadata.name == expected_name
        assert model.metadata.endpoint == expected_endpoint
        assert model.metadata.api_key == expected_api_key
        assert model.metadata.engine == expected_engine
        assert model.metadata.identifier == expected_identifier

        model = LiteLlmTextGenModel.from_model_name(name=model_name, api_key=None)

        assert model.metadata.endpoint is None

        if provider == KindModelProvider.LITELLM:
            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(name=model_name, endpoint=endpoint)
            assert str(exc.value) == "specifying custom models endpoint is disabled"

            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(name=model_name, api_key="api-key")
            assert str(exc.value) == "specifying custom models endpoint is disabled"

        if provider == KindModelProvider.FIREWORKS:
            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(
                    provider=provider,
                    name=model_name,
                    provider_keys={},
                )
            assert str(exc.value) == "Fireworks API key is missing from configuration."

            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(
                    provider=provider,
                    name=model_name,
                    provider_keys={"fireworks_api_key": "stubbed-api-key"},
                    provider_endpoints={
                        "fireworks_current_region_endpoint": {"invalid": "config"}
                    },
                )
            assert (
                str(exc.value)
                == "Fireworks endpoint or identifier missing in region config."
            )

            with pytest.raises(ValueError) as exc:
                LiteLlmTextGenModel.from_model_name(
                    provider=provider,
                    name=model_name,
                    provider_keys={"fireworks_api_key": "stubbed-api-key"},
                    provider_endpoints={},
                )
            assert (
                str(exc.value)
                == "Fireworks regional endpoints configuration is missing."
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "model_name",
            "provider",
            "custom_models_enabled",
            "model_completion_args",
            "prefix",
            "suffix",
            "expect_async_client",
            "expected_score",
        ),
        [
            (
                "codegemma",
                KindModelProvider.LITELLM,
                True,
                {
                    "model": "text-completion-custom_openai/codegemma",
                    "stop": [
                        "<|fim_prefix|>",
                        "<|fim_suffix|>",
                        "<|fim_middle|>",
                        "<|file_separator|>",
                    ],
                },
                "def hello_world():",
                "def goodbye_world():",
                False,
                10**5,
            ),
            (
                "codestral",
                KindModelProvider.MISTRALAI,
                True,
                {
                    "model": "text-completion-codestral/codestral",
                    "stop": [],
                    "api_key": "codestral-api-key",
                },
                "def hello_world():",
                "def goodbye_world():",
                False,
                10**5,
            ),
            (
                "qwen2p5-coder-7b",
                KindModelProvider.FIREWORKS,
                True,
                {
                    "model": "provider/some-cool-model#deployment_id",
                    "stop": [
                        "<|fim_prefix|>",
                        "<|fim_suffix|>",
                        "<|fim_middle|>",
                        "<|fim_pad|>",
                        "<|repo_name|>",
                        "<|file_sep|>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "\n\n",
                        "def goodbye_world():",
                    ],
                    "api_key": "fireworks-api-key",
                    "messages": [
                        {
                            "content": "<|fim_prefix|>def hello_world():<|fim_suffix|>def goodbye_world():<|fim_middle|>",
                            "role": Role.USER,
                        }
                    ],
                    "timeout": 60,
                    "api_base": "https://fireworks.endpoint",
                    "custom_llm_provider": "text-completion-openai",
                    "extra_headers": {"x-session-affinity": "test"},
                    "prompt_cache_max_len": 0,
                    "logprobs": True,
                },
                "def hello_world():",
                "def goodbye_world():",
                True,
                999.0,
            ),
            (
                "qwen2p5-coder-7b",
                KindModelProvider.FIREWORKS,
                True,
                {
                    "model": "provider/some-cool-model#deployment_id",
                    "stop": [
                        "<|fim_prefix|>",
                        "<|fim_suffix|>",
                        "<|fim_middle|>",
                        "<|fim_pad|>",
                        "<|repo_name|>",
                        "<|file_sep|>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "\n\n",
                    ],
                    "api_key": "fireworks-api-key",
                    "messages": [
                        {
                            "content": "<|fim_prefix|>def hello_world():<|fim_suffix|><|fim_middle|>",
                            "role": Role.USER,
                        }
                    ],
                    "timeout": 60,
                    "api_base": "https://fireworks.endpoint",
                    "custom_llm_provider": "text-completion-openai",
                    "extra_headers": {"x-session-affinity": "test"},
                    "prompt_cache_max_len": 0,
                    "logprobs": True,
                },
                "def hello_world():",
                None,
                False,
                999.0,
            ),
            (
                "qwen2p5-coder-7b",
                KindModelProvider.FIREWORKS,
                True,
                {
                    "model": "provider/some-cool-model#deployment_id",
                    "stop": [
                        "<|fim_prefix|>",
                        "<|fim_suffix|>",
                        "<|fim_middle|>",
                        "<|fim_pad|>",
                        "<|repo_name|>",
                        "<|file_sep|>",
                        "<|im_start|>",
                        "<|im_end|>",
                        "\n\n",
                        "def goodbye_world():",
                    ],
                    "api_key": "fireworks-api-key",
                    "messages": [
                        {
                            "content": "<|fim_prefix|>def hello_world():<|fim_suffix|>def goodbye_world():\n//another line<|fim_middle|>",
                            "role": Role.USER,
                        }
                    ],
                    "timeout": 60,
                    "api_base": "https://fireworks.endpoint",
                    "custom_llm_provider": "text-completion-openai",
                    "extra_headers": {"x-session-affinity": "test"},
                    "prompt_cache_max_len": 0,
                    "logprobs": 1,
                },
                "def hello_world():",
                "def goodbye_world():\n//another line",
                True,
                999.0,
            ),
        ],
    )
    async def test_generate(
        self,
        model_name,
        provider,
        custom_models_enabled,
        model_completion_args,
        endpoint,
        api_key,
        provider_keys,
        provider_endpoints,
        expect_async_client,
        mock_litellm_acompletion: Mock,
        prefix,
        suffix,
        expected_score,
    ):
        async_fireworks_client = Mock() if expect_async_client else None

        litellm_model = LiteLlmTextGenModel.from_model_name(
            name=model_name,
            provider=provider,
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=custom_models_enabled,
            provider_keys=provider_keys,
            provider_endpoints=provider_endpoints,
            async_fireworks_client=async_fireworks_client,
        )

        output = await litellm_model.generate(
            prefix=prefix,
            suffix=suffix,
            snowplow_event_context=MagicMock(
                spec=SnowplowEventContext, gitlab_global_user_id="test"
            ),
        )

        expected_completion_args = {
            "max_tokens": 16,
            "temperature": 0.95,
            "top_p": 0.95,
            "stream": False,
            "timeout": 30.0,
            "api_key": api_key,
            "api_base": endpoint,
            "messages": [{"content": prefix, "role": Role.USER}],
        }
        expected_completion_args.update(model_completion_args)

        if provider == KindModelProvider.FIREWORKS:
            expected_completion_args["client"] = (
                async_fireworks_client if expect_async_client else None
            )

        mock_litellm_acompletion.assert_called_with(**expected_completion_args)

        assert isinstance(output, TextGenModelOutput)
        assert output.text == "Test response"
        assert output.score == expected_score
        if output.metadata:
            assert output.metadata.output_tokens == 999

    @pytest.mark.asyncio
    async def test_override_stream(self, endpoint, api_key):
        generation_model = LiteLlmTextGenModel.from_model_name(
            name="mistral",
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=True,
            disable_streaming=True,
        )

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))],
                usage=AsyncMock(completion_tokens=999),
            )

            output = await generation_model.generate(
                prefix="def hello_world():",
                suffix=None,
                stream=True,
            )

            mock_acompletion.assert_called_with(
                messages=[{"content": "def hello_world():", "role": Role.USER}],
                max_tokens=16,
                temperature=0.95,
                top_p=0.95,
                stream=False,
                timeout=30.0,
                stop=["</new_code>"],
                api_base="http://127.0.0.1:4000",
                api_key="specified-api-key",
                model="text-completion-custom_openai/mistral",
            )

            assert isinstance(output, TextGenModelOutput)
            assert output.text == "Test response"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "model_name",
            "provider",
            "custom_models_enabled",
        ),
        [
            (
                "codegemma",
                KindModelProvider.LITELLM,
                True,
            ),
            (
                "codestral",
                KindModelProvider.MISTRALAI,
                True,
            ),
        ],
    )
    async def test_generate_internal_server_error(
        self,
        model_name,
        provider,
        custom_models_enabled,
        endpoint,
        api_key,
        provider_keys,
        mock_litellm_acompletion: Mock,
    ):
        litellm_model = LiteLlmTextGenModel.from_model_name(
            name=model_name,
            provider=provider,
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=custom_models_enabled,
            provider_keys=provider_keys,
        )

        mock_litellm_acompletion.side_effect = InternalServerError(
            message="Test internal server error",
            llm_provider=provider,
            model=model_name,
        )

        with pytest.raises(LiteLlmInternalServerError) as ex:
            await litellm_model.generate(
                prefix="def hello_world():",
            )

        expected_error_message = (
            "litellm.InternalServerError: Test internal server error"
        )

        assert str(ex.value) == expected_error_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "model_name",
            "provider",
            "custom_models_enabled",
        ),
        [
            (
                "codegemma",
                KindModelProvider.LITELLM,
                True,
            ),
            (
                "codestral",
                KindModelProvider.MISTRALAI,
                True,
            ),
        ],
    )
    async def test_generate_api_connection_error(
        self,
        model_name,
        provider,
        custom_models_enabled,
        endpoint,
        api_key,
        provider_keys,
        mock_litellm_acompletion: Mock,
    ):
        litellm_model = LiteLlmTextGenModel.from_model_name(
            name=model_name,
            provider=provider,
            endpoint=endpoint,
            api_key=api_key,
            custom_models_enabled=custom_models_enabled,
            provider_keys=provider_keys,
        )

        mock_litellm_acompletion.side_effect = APIConnectionError(
            message="Test API connection error", llm_provider=provider, model=model_name
        )

        with pytest.raises(LiteLlmAPIConnectionError) as ex:
            await litellm_model.generate(
                prefix="def hello_world():",
            )

        expected_error_message = "litellm.APIConnectionError: Test API connection error"

        assert str(ex.value) == expected_error_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "model_name",
            "provider",
            "custom_models_enabled",
            "endpoint",
            "api_key",
        ),
        [
            (
                "codegemma",
                KindModelProvider.LITELLM,
                True,
                "http:://codegemma.local",
                "api-key",
            ),
            (
                "codestral",
                KindModelProvider.MISTRALAI,
                True,
                "http://codestral.local",
                None,
            ),
            (
                "claude_3.5",
                KindModelProvider.LITELLM,
                True,
                None,
                None,
            ),
        ],
    )
    async def test_generate_stream(
        self,
        model_name,
        provider,
        custom_models_enabled,
        endpoint,
        api_key,
        provider_keys,
        mock_litellm_acompletion_streamed: Mock,
    ):
        with patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            litellm_model = LiteLlmTextGenModel.from_model_name(
                name=model_name,
                provider=provider,
                endpoint=endpoint,
                api_key=api_key,
                custom_models_enabled=custom_models_enabled,
                provider_keys=provider_keys,
            )

            response = await litellm_model.generate(
                prefix="Test message",
                stream=True,
            )

            _args, kwargs = mock_litellm_acompletion_streamed.call_args
            assert kwargs["stream"] is True

            content = []
            async for chunk in response:
                content.append(chunk.text)
            assert content == ["Streamed content"]

            mock_watch.assert_called_once_with(stream=True)
            watcher.finish.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_instrumented(self, lite_llm_text_model):
        async def mock_stream(*args, **kwargs):
            completions = [
                AsyncMock(
                    choices=[AsyncMock(delta=AsyncMock(content="Streamed content"))]
                ),
                "break here",
            ]
            for item in completions:
                if item == "break here":
                    raise ValueError("broken")
                yield item

        with patch("ai_gateway.models.litellm.acompletion") as mock_acompletion, patch(
            "ai_gateway.instrumentators.model_requests.ModelRequestInstrumentator.watch"
        ) as mock_watch:
            watcher = Mock()
            mock_watch.return_value.__enter__.return_value = watcher

            mock_acompletion.side_effect = AsyncMock(side_effect=mock_stream)

            response = await lite_llm_text_model.generate(
                prefix="Test message", stream=True
            )

            watcher.finish.assert_not_called()

            with pytest.raises(ValueError):
                _ = [item async for item in response]

            mock_watch.assert_called_once_with(stream=True)
            watcher.register_error.assert_called_once()
            watcher.finish.assert_called_once()
