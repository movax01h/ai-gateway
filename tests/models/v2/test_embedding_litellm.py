from unittest.mock import AsyncMock

import litellm
import pytest
from langchain_core.messages import AIMessage

from ai_gateway.models.v2.embedding_litellm import (
    EmbeddingAuthenticationError,
    EmbeddingBadRequestError,
    EmbeddingLiteLLM,
    EmbeddingRateLimitError,
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

        call_kwargs = mock_litellm_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test text 1", "test text 2"]
        assert call_kwargs["model"] == "test-embedding-model"
        assert call_kwargs["custom_llm_provider"] == "openai"
        assert "dimensions" not in call_kwargs

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
