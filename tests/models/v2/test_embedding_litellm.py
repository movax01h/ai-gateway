from unittest.mock import AsyncMock

import litellm
import pytest
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from litellm.types.utils import (
    CacheCreationTokenDetails,
    PromptTokensDetailsWrapper,
    Usage,
)

from ai_gateway.models.v2.embedding_litellm import (
    EmbeddingAuthenticationError,
    EmbeddingBadRequestError,
    EmbeddingLiteLLM,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
)


class TestEmbeddingLiteLLMProperties:
    def test_properties(self):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        assert model.model == "test-embedding-model"
        assert model.custom_llm_provider == "openai"
        assert model._llm_type == "litellm-embedding"
        assert model._identifying_params == {"model": "test-embedding-model"}
        assert model.disable_streaming is False


class TestEmbeddingLiteLLMNotImplementedCalls:
    def test_sync_invoke_not_implemented(self):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )
        with pytest.raises(
            NotImplementedError, match="Sync invocation not implemented. Use ainvoke."
        ):
            model.invoke(input={"contents": ["test"]})

    def test_sync_stream_not_implemented(self):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )
        with pytest.raises(
            NotImplementedError,
            match="Sync or async streaming not implemented. Use ainvoke.",
        ):
            model.stream(input={"contents": ["test"]})

    @pytest.mark.asyncio
    async def test_async_stream_not_implemented(self):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )
        with pytest.raises(
            NotImplementedError,
            match="Sync or async streaming not implemented. Use ainvoke.",
        ):
            await model.astream(input={"contents": ["test"]})


class TestEmbeddingLiteLLMAsyncInvoke:
    @pytest.mark.asyncio
    async def test_async_invoke(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(input={"contents": ["test text 1", "test text 2"]})

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        llm_response_usage = mock_litellm_aembedding_response.usage
        assert result.usage_metadata == {
            "input_tokens": llm_response_usage.prompt_tokens,
            "output_tokens": llm_response_usage.completion_tokens,
            "total_tokens": llm_response_usage.total_tokens,
            "input_token_details": {"cache_read": 4},
        }

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["model"] == "test-embedding-model"
        assert call_kwargs["custom_llm_provider"] == "openai"
        assert "dimensions" not in call_kwargs

    @pytest.mark.asyncio
    async def test_async_invoke_with_cache_creation_details(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        mock_litellm_aembedding_response.usage = Usage(
            prompt_tokens=12,
            completion_tokens=0,
            total_tokens=12,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=4,
                text_tokens=8,
                cache_creation_tokens=6,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=2,
                    ephemeral_1h_input_tokens=4,
                ),
            ),
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(input={"contents": ["test text 1"]})

        assert result.usage_metadata == {
            "input_tokens": 12,
            "output_tokens": 0,
            "total_tokens": 12,
            "input_token_details": {
                "cache_read": 4,
                "cache_creation": 6,
                "ephemeral_5m_input_tokens": 2,
                "ephemeral_1h_input_tokens": 4,
            },
        }

    @pytest.mark.asyncio
    async def test_async_invoke_no_prompt_tokens_details(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        mock_litellm_aembedding_response.usage = Usage(
            prompt_tokens=12,
            completion_tokens=0,
            total_tokens=12,
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(input={"contents": ["test text 1"]})

        assert result.usage_metadata == {
            "input_tokens": 12,
            "output_tokens": 0,
            "total_tokens": 12,
        }

    @pytest.mark.asyncio
    async def test_async_invoke_no_usage_data(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        mock_litellm_aembedding_response.usage = None

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(input={"contents": ["test text 1", "test text 2"]})

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        assert result.usage_metadata == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_request_timeout_reaches_litellm_aembedding(
        self, mock_litellm_aembedding
    ):
        """`litellm.aembedding` reads the per-attempt bound from `timeout`; an omitted key leaves its own 600s
        default."""
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="openai",
            request_timeout=42.0,
        )

        await model.ainvoke(input={"contents": ["test text"]})

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["timeout"] == 42.0

    @pytest.mark.asyncio
    async def test_max_retries_reaches_litellm_aembedding(
        self, mock_litellm_aembedding
    ):
        """An omitted `max_retries` leaves the provider client on its own default, multiplying the wall bound."""
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="openai",
            max_retries=3,
        )

        await model.ainvoke(input={"contents": ["test text"]})

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_async_invoke_with_dimensions(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(
            input={
                "contents": ["test text 1", "test text 2"],
                "dimensions": 768,
            }
        )

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["dimensions"] == 768
        assert call_kwargs["model"] == "test-embedding-model"
        assert call_kwargs["custom_llm_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_async_invoke_with_drop_params(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        result = await model.ainvoke(
            input={
                "contents": ["test text 1", "test text 2"],
                "dimensions": 768,
                "drop_params": True,
            }
        )

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["dimensions"] == 768
        assert call_kwargs["model"] == "test-embedding-model"
        assert call_kwargs["custom_llm_provider"] == "openai"
        assert call_kwargs["drop_params"] is True

    @pytest.mark.asyncio
    async def test_async_invoke_vertex(
        self, mock_litellm_aembedding, mock_litellm_aembedding_response
    ):
        model = EmbeddingLiteLLM(
            model="text-embedding",
            custom_llm_provider="vertex_ai",
        )

        result = await model.ainvoke(
            input={"contents": ["test text 1", "test text 2"]},
            vertex_location="europe-west4",
        )

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["model"] == "text-embedding"
        assert call_kwargs["custom_llm_provider"] == "vertex_ai"
        assert call_kwargs["vertex_ai_location"] == "europe-west4"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "default_api_base",
            "default_api_key",
            "override_api_base",
            "override_api_key",
        ),
        [
            (
                "http://localhost/url/base",
                "default-key",
                None,
                None,
            ),
            (
                "http://localhost/url/base",
                "default-key",
                "http://localhost/url/model/override",
                "model-override-key",
            ),
        ],
    )
    async def test_async_invoke_with_api_endpoint(
        self,
        default_api_base,
        default_api_key,
        override_api_base,
        override_api_key,
        mock_litellm_aembedding,
        mock_litellm_aembedding_response,
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="openai",
            api_base=default_api_base,
            api_key=default_api_key,
        )

        result = await model.ainvoke(
            input={"contents": ["test text 1", "test text 2"]},
            api_base=override_api_base,
            api_key=override_api_key,
        )

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["model"] == "test-embedding-model"
        assert call_kwargs["custom_llm_provider"] == "openai"
        assert call_kwargs["api_base"] == override_api_base or default_api_base
        assert call_kwargs["api_key"] == override_api_key or default_api_key

    @pytest.mark.asyncio
    async def test_async_invoke_with_model_override(
        self,
        mock_litellm_aembedding,
        mock_litellm_aembedding_response,
    ):
        model = EmbeddingLiteLLM(model="embedding", custom_llm_provider="openai")

        result = await model.ainvoke(
            input={"contents": ["test text 1", "test text 2"]},
            model="test-embedding-model-override",
        )

        assert isinstance(result, AIMessage)
        assert result.content == mock_litellm_aembedding_response.data

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["model"] == "test-embedding-model-override"
        assert call_kwargs["custom_llm_provider"] == "openai"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("mock_response_values"), [{"data": None}, {"data": []}])
    async def test_async_invoke_empty_response_data(
        self, mock_response_values, mock_litellm_aembedding
    ):
        mock_litellm_aembedding.return_value = AsyncMock(**mock_response_values)

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        error_message = "Unexpected response format: missing or empty response data"
        with pytest.raises(ValueError, match=error_message):
            await model.ainvoke(input={"contents": ["test text"]})

    @pytest.mark.asyncio
    async def test_async_invoke_bad_request_error(self, mock_litellm_aembedding):
        error_message = "Bad request error from litellm"

        mock_litellm_aembedding.side_effect = litellm.BadRequestError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        with pytest.raises(EmbeddingBadRequestError, match=error_message):
            await model.ainvoke(input={"contents": ["test text"]})

    @pytest.mark.asyncio
    async def test_async_invoke_rate_limit_error(self, mock_litellm_aembedding):
        error_message = "Resource exhausted, please try again later"

        mock_litellm_aembedding.side_effect = litellm.RateLimitError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        with pytest.raises(EmbeddingRateLimitError, match=error_message):
            await model.ainvoke(input={"contents": ["test text"]})

    @pytest.mark.asyncio
    async def test_async_invoke_authentication_error(self, mock_litellm_aembedding):
        error_message = "Authentication error"

        mock_litellm_aembedding.side_effect = litellm.AuthenticationError(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        with pytest.raises(EmbeddingAuthenticationError, match=error_message):
            await model.ainvoke(input={"contents": ["test text"]})

    @pytest.mark.asyncio
    async def test_async_invoke_timeout_error(self, mock_litellm_aembedding):
        error_message = "Request timed out"

        mock_litellm_aembedding.side_effect = litellm.Timeout(
            message=error_message, model="test-embedding-model", llm_provider="openai"
        )

        model = EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

        with pytest.raises(EmbeddingTimeoutError, match=error_message):
            await model.ainvoke(input={"contents": ["test text"]})


class TestEmbeddingLiteLLMBind:
    @pytest.mark.parametrize(
        (
            "custom_models_enabled",
            "override_api_base",
            "override_api_key",
        ),
        [
            (False, None, None),
            (True, "http://test", "test-api-key"),
        ],
    )
    def test_bind_successful(
        self, custom_models_enabled, override_api_base, override_api_key
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="custom_openai",
            custom_models_enabled=custom_models_enabled,
        )

        bound_model = model.bind(api_base=override_api_base, api_key=override_api_key)

        assert bound_model.model == "test-embedding-model"
        assert bound_model.custom_llm_provider == "custom_openai"

    @pytest.mark.parametrize(
        (
            "override_api_base",
            "override_api_key",
            "unexpected_field",
        ),
        [
            ("http://test", None, "api_base"),
            (None, "test-api-key", "api_key"),
        ],
    )
    def test_bind_failed_for_custom_models_disabled(
        self, override_api_base, override_api_key, unexpected_field
    ):
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="custom_openai",
        )

        with pytest.raises(
            ValueError,
            match=f"specifying custom models endpoint is disabled: {unexpected_field} is not allowed",
        ):
            model.bind(api_base=override_api_base, api_key=override_api_key)


class TestEmbeddingLiteLLMUserIdentityHeader:
    @pytest.fixture(name="model")
    def model_fixture(self):
        return EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="openai",
            custom_models_enabled=True,
            user_id_header="x-gitlab-user-id",
        )

    @pytest.mark.asyncio
    async def test_forwards_user_id(
        self, model, mock_litellm_aembedding, gitlab_user_id_in_context
    ):
        await model.ainvoke(input={"contents": ["test text"]})

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["extra_headers"] == {
            "x-gitlab-user-id": gitlab_user_id_in_context
        }

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_gitlab_user_id_in_context")
    async def test_omits_header_without_user_id(self, model, mock_litellm_aembedding):
        await model.ainvoke(input={"contents": ["test text"]})

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert "extra_headers" not in call_kwargs

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("gitlab_user_id_in_context")
    async def test_omits_header_when_not_configured(self, mock_litellm_aembedding):
        model = EmbeddingLiteLLM(
            model="test-embedding-model",
            custom_llm_provider="openai",
            custom_models_enabled=True,
        )

        await model.ainvoke(input={"contents": ["test text"]})

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert "extra_headers" not in call_kwargs


class TestEmbeddingLiteLLMCallbacks:
    @pytest.fixture(name="embedding_model")
    def embedding_model_fixture(self):
        return EmbeddingLiteLLM(
            model="test-embedding-model", custom_llm_provider="openai"
        )

    @pytest.fixture(name="mock_callback_handler")
    def mock_callback_handler_fixture(self):
        # LangChain skips an event when the handler's matching `ignore_` flag is truthy,
        # and every auto-created AsyncMock attribute is truthy
        return AsyncMock(ignore_llm=False, ignore_chat_model=False)

    @pytest.mark.asyncio
    async def test_llm_run(
        self,
        embedding_model,
        mock_callback_handler,
        mock_litellm_aembedding,
        mock_litellm_aembedding_response,
    ):
        ainvoke_result = await embedding_model.ainvoke(
            input={"contents": ["test text 1", "test text 2"]},
            config={"callbacks": [mock_callback_handler]},
        )

        mock_callback_handler.on_llm_start.assert_called_once()
        mock_callback_handler.on_chat_model_start.assert_not_called()

        serialized, prompts = mock_callback_handler.on_llm_start.call_args.args
        call_kwargs = mock_callback_handler.on_llm_start.call_args.kwargs
        assert serialized == {}
        assert prompts == [""]
        assert call_kwargs["invocation_params"] == {"model": "test-embedding-model"}
        assert call_kwargs["name"] == "litellm-embedding"

        mock_callback_handler.on_llm_end.assert_called_once()
        mock_callback_handler.on_llm_error.assert_not_called()

        (llm_result,) = mock_callback_handler.on_llm_end.call_args.args
        assert isinstance(llm_result, LLMResult)

        generation = llm_result.generations[0][0]
        assert isinstance(generation, ChatGeneration)
        assert generation.message is ainvoke_result
        assert generation.message.usage_metadata == {
            "input_tokens": mock_litellm_aembedding_response.usage.prompt_tokens,
            "output_tokens": mock_litellm_aembedding_response.usage.completion_tokens,
            "total_tokens": mock_litellm_aembedding_response.usage.total_tokens,
            "input_token_details": {"cache_read": 4},
        }

    @pytest.mark.asyncio
    async def test_usage_metadata_callback(
        self,
        embedding_model,
        mock_litellm_aembedding,
        mock_litellm_aembedding_response,
    ):
        usage_cb = UsageMetadataCallbackHandler()

        await embedding_model.ainvoke(
            input={"contents": ["test text 1"]},
            config={"callbacks": [usage_cb]},
        )

        assert usage_cb.usage_metadata == {
            mock_litellm_aembedding_response.model: {
                "input_tokens": 12,
                "output_tokens": 0,
                "total_tokens": 12,
                "input_token_details": {"cache_read": 4},
            }
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_attribute_missing", [False, True])
    async def test_usage_metadata_callback_without_model_in_response(
        self,
        model_attribute_missing,
        embedding_model,
        mock_litellm_aembedding,
        mock_litellm_aembedding_response,
    ):
        if model_attribute_missing:
            del mock_litellm_aembedding_response.model
        else:
            mock_litellm_aembedding_response.model = None

        usage_cb = UsageMetadataCallbackHandler()

        await embedding_model.ainvoke(
            input={"contents": ["test text 1"]},
            config={"callbacks": [usage_cb]},
        )

        assert list(usage_cb.usage_metadata) == [embedding_model.model]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("litellm_call_error", "expected_result_error"),
        [
            (
                litellm.BadRequestError(
                    message="Bad request error from litellm",
                    model="test-embedding-model",
                    llm_provider="openai",
                ),
                EmbeddingBadRequestError,
            ),
            (
                litellm.RateLimitError(
                    message="Resource exhausted, please try again later",
                    model="test-embedding-model",
                    llm_provider="openai",
                ),
                EmbeddingRateLimitError,
            ),
            (
                litellm.AuthenticationError(
                    message="Authentication error",
                    model="test-embedding-model",
                    llm_provider="openai",
                ),
                EmbeddingAuthenticationError,
            ),
            (
                litellm.Timeout(
                    message="Request timed out",
                    model="test-embedding-model",
                    llm_provider="openai",
                ),
                EmbeddingTimeoutError,
            ),
            (RuntimeError("Unexpected failure"), RuntimeError),
        ],
    )
    async def test_error_encountered(
        self,
        embedding_model,
        mock_callback_handler,
        mock_litellm_aembedding,
        litellm_call_error,
        expected_result_error,
    ):
        mock_litellm_aembedding.side_effect = litellm_call_error

        with pytest.raises(expected_result_error):
            await embedding_model.ainvoke(
                input={"contents": ["test text"]},
                config={"callbacks": [mock_callback_handler]},
            )

        mock_callback_handler.on_llm_start.assert_called_once()
        mock_callback_handler.on_llm_error.assert_called_once()
        mock_callback_handler.on_llm_end.assert_not_called()

        (reported_error,) = mock_callback_handler.on_llm_error.call_args.args
        assert reported_error is litellm_call_error

    @pytest.mark.asyncio
    async def test_reports_response_extraction_error(
        self, embedding_model, mock_callback_handler, mock_litellm_aembedding
    ):
        mock_litellm_aembedding.return_value = AsyncMock(data=[])

        with pytest.raises(ValueError):
            await embedding_model.ainvoke(
                input={"contents": ["test text"]},
                config={"callbacks": [mock_callback_handler]},
            )

        mock_callback_handler.on_llm_error.assert_called_once()
        mock_callback_handler.on_llm_end.assert_not_called()
