import time
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable

from ai_gateway.config import ConfigBedrockGuardrail
from ai_gateway.models.guardrails import BEDROCK_GUARDRAIL_PROVIDERS
from ai_gateway.models.v2._model_compat import PREVIOUS_ASSISTANT_CONTEXT_PREFIX
from ai_gateway.models.v2.chat_litellm import (
    ChatLiteLLM,
    _force_gpt_5_max_completion_tokens,
)
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM as _LChatLiteLLM
from ai_gateway.vendor.langchain_litellm.litellm import _create_usage_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "acompletion_response_fixture", ["acompletion_stream_response"]
)
@pytest.mark.usefixtures("mock_acompletion_with_retry")
async def test_astream_with_stream_options_and_stop_reason():
    """Test that stream_options is added correctly and finish_reason is extracted from stop_reason."""
    message = HumanMessage(content="Hello")

    chat = ChatLiteLLM(model="gpt-3.5-turbo")

    result = []
    async for item in chat._astream(messages=[message]):
        result.append(item)

    # Verify we got chunks
    assert len(result) == 3

    # Verify the last chunk has finish_reason in response_metadata
    last_chunk = result[-1]
    assert isinstance(last_chunk, ChatGenerationChunk)
    assert isinstance(last_chunk.message, AIMessageChunk)
    assert last_chunk.message.response_metadata.get("finish_reason") == "stop"


@pytest.mark.asyncio
async def test_fireworks_logprobs_in_streaming():
    """Test that Fireworks logprobs parameter is correctly set for streaming mode."""

    message = HumanMessage(content="def hello():")

    # Mock Fireworks streaming response with logprobs
    mock_chunks = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "    "},
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": [-0.5],
                        "tokens": ["    "],
                    },
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "delta": {"content": "print"},
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": [-1.2],
                        "tokens": ["print"],
                    },
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]

    async def mock_acompletion(*_args, **kwargs):
        # Verify that logprobs=1 was passed for Fireworks
        assert kwargs.get("logprobs") == 1
        for chunk in mock_chunks:
            yield chunk

    with patch(
        "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
        new=AsyncMock(side_effect=mock_acompletion),
    ):
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

        result = []
        async for item in chat._astream(messages=[message]):
            result.append(item)

        # Verify we got chunks
        assert len(result) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "acompletion_response_fixture", ["acompletion_non_stream_with_logprobs_response"]
)
@pytest.mark.usefixtures("mock_acompletion_with_retry")
async def test_fireworks_logprobs_in_non_streaming():
    """Test that Fireworks logprobs are correctly extracted and returned as score in non-streaming mode."""

    message = HumanMessage(content="def hello():")

    # Mock Fireworks non-streaming response with logprobs
    chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

    result = await chat._agenerate(messages=[message])

    # Verify the result has generations
    assert len(result.generations) == 1
    generation = result.generations[0]

    # Verify logprobs are in generation_info
    assert generation.generation_info is not None
    assert "logprobs" in generation.generation_info

    # Verify score was extracted from first token logprob
    assert "score" in generation.generation_info
    assert generation.generation_info["score"] == -0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "acompletion_response_fixture", ["acompletion_non_stream_response"]
)
async def test_fireworks_session_affinity_header(mock_acompletion_with_retry):
    """Test that Fireworks session affinity header is correctly set."""

    message = HumanMessage(content="def hello():")

    chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

    await chat._agenerate(messages=[message], session_id="test-session-123")

    # Verify the call was made with session affinity header
    call_kwargs = mock_acompletion_with_retry.call_args[1]
    assert "extra_headers" in call_kwargs
    assert call_kwargs["extra_headers"]["x-session-affinity"] == "test-session-123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "acompletion_response_fixture", ["acompletion_non_stream_response"]
)
async def test_fireworks_prompt_caching_disabled(mock_acompletion_with_retry):
    """Test that Fireworks prompt caching can be disabled."""

    message = HumanMessage(content="def hello():")

    chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

    await chat._agenerate(messages=[message], using_cache="false")

    # Verify the call was made with prompt_cache_max_len=0
    call_kwargs = mock_acompletion_with_retry.call_args[1]
    assert call_kwargs["prompt_cache_max_len"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_usage", "expected_usage_metadata"),
    [
        (
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
            },
            UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3),
        ),
        (
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 4,
            },
            UsageMetadata(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                input_token_details=InputTokenDetails(cache_creation=3, cache_read=4),
            ),
        ),
    ],
)
async def test_create_usage_metadata(token_usage, expected_usage_metadata):
    assert _create_usage_metadata(token_usage) == expected_usage_metadata


@pytest.mark.parametrize(
    ("bind_tools_params", "expected_tools"),
    [
        (
            {"web_search_options": {}},
            [{"type": "function", "function": {"name": "get_issue"}}],
        ),
        ({}, [{"type": "function", "function": {"name": "get_issue"}}]),
    ],
)
def test_bind_tools_with_web_search_options(bind_tools_params, expected_tools):
    """Test that web search tool is added when web_search_options is in bind_tools_params."""
    chat = ChatLiteLLM(model="gpt-3.5-turbo")

    existing_tools = [{"name": "get_issue"}]
    result = chat.bind_tools(
        tools=existing_tools,
        **bind_tools_params,
    )

    assert isinstance(result, Runnable)
    assert result.kwargs["tools"] == expected_tools


@pytest.mark.asyncio
async def test_fireworks_503_retries_with_exponential_backoff():
    """Verify 503 errors trigger retries with exponential backoff for Fireworks."""
    message = HumanMessage(content="test")

    # Mock to fail twice with 503, then succeed
    call_count = 0

    async def mock_acompletion(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise litellm.ServiceUnavailableError(
                message="Service temporarily unavailable",
                llm_provider="fireworks_ai",
                model="test-model",
            )
        return {
            "choices": [
                {
                    "message": {"content": "success", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    chat = ChatLiteLLM(
        model="test-model", custom_llm_provider="fireworks_ai", max_retries=3
    )

    with (
        patch("asyncio.sleep", new=AsyncMock()),
        patch.object(
            chat.client, "acompletion", new=AsyncMock(side_effect=mock_acompletion)
        ),
    ):
        result = await chat._agenerate(messages=[message])

        assert call_count == 3  # Failed twice, succeeded on third
        assert result.generations[0].message.content == "success"


@pytest.mark.asyncio
async def test_non_fireworks_503_fails_immediately():
    """Verify 503 errors do NOT trigger retries for non-Fireworks providers."""
    message = HumanMessage(content="test")

    call_count = 0

    async def mock_acompletion(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        raise litellm.ServiceUnavailableError(
            message="Service temporarily unavailable",
            llm_provider="anthropic",
            model="test-model",
        )

    chat = ChatLiteLLM(
        model="test-model", custom_llm_provider="anthropic", max_retries=3
    )

    with patch.object(
        chat.client, "acompletion", new=AsyncMock(side_effect=mock_acompletion)
    ):
        with pytest.raises(litellm.ServiceUnavailableError):
            await chat._agenerate(messages=[message])

        # Should fail on first attempt without retrying ServiceUnavailableError
        assert call_count == 1


@pytest.mark.asyncio
@pytest.mark.skip(reason="Slow test - takes ~120 seconds")
async def test_fireworks_retry_respects_120s_timeout():
    """Verify retries stop after 120 second timeout."""
    message = HumanMessage(content="test")

    start_time = time.time()

    async def mock_acompletion(*_args, **_kwargs):
        raise litellm.ServiceUnavailableError(
            message="Service temporarily unavailable",
            llm_provider="fireworks_ai",
            model="test-model",
        )

    chat = ChatLiteLLM(
        model="test-model",
        custom_llm_provider="fireworks_ai",
        max_retries=100,  # Set high to test timeout, not max_retries
    )

    with patch.object(
        chat.client, "acompletion", new=AsyncMock(side_effect=mock_acompletion)
    ):
        with pytest.raises(litellm.ServiceUnavailableError):
            await chat._agenerate(messages=[message])

        elapsed = time.time() - start_time

        # Should stop around 120 seconds (allow some tolerance)
        assert 115 <= elapsed <= 125, f"Expected ~120s timeout, got {elapsed}s"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Slow test - tests precise timing over 15+ seconds")
async def test_fireworks_exponential_backoff_timing():
    """Verify backoff follows 1s, 2s, 4s, 8s, 10s, 10s...

    pattern.
    """
    message = HumanMessage(content="test")

    call_times = []

    async def mock_acompletion(*_args, **_kwargs):
        call_times.append(time.time())
        raise litellm.ServiceUnavailableError(
            message="Service temporarily unavailable",
            llm_provider="fireworks_ai",
            model="test-model",
        )

    chat = ChatLiteLLM(
        model="test-model", custom_llm_provider="fireworks_ai", max_retries=5
    )

    with patch.object(
        chat.client, "acompletion", new=AsyncMock(side_effect=mock_acompletion)
    ):
        with pytest.raises(litellm.ServiceUnavailableError):
            await chat._agenerate(messages=[message])

        # Calculate delays between calls
        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        # Expected delays: ~1s, ~2s, ~4s, ~8s (with some tolerance)
        # Note: First 4 retries should follow exponential pattern
        if len(delays) >= 4:
            assert 0.8 <= delays[0] <= 1.5, (
                f"First delay should be ~1s, got {delays[0]}"
            )
            assert 1.8 <= delays[1] <= 2.5, (
                f"Second delay should be ~2s, got {delays[1]}"
            )
            assert 3.5 <= delays[2] <= 4.5, (
                f"Third delay should be ~4s, got {delays[2]}"
            )
            assert 7.0 <= delays[3] <= 9.0, (
                f"Fourth delay should be ~8s, got {delays[3]}"
            )


def test_fireworks_max_retries_set_via_params():
    """Verify that max_retries can be set via initialization params for Fireworks."""
    chat = ChatLiteLLM(
        model="test-model", custom_llm_provider="fireworks_ai", max_retries=10
    )
    assert chat.max_retries == 10


def test_non_fireworks_max_retries_uses_default():
    """Verify that non-Fireworks providers use default max_retries."""
    chat = ChatLiteLLM(model="test-model", custom_llm_provider="anthropic")
    assert chat.max_retries == 1


class TestMistralAIPrefixFormat:
    """Tests for automatic Mistral AI prefix format support based on custom_llm_provider."""

    @pytest.mark.parametrize(
        ("custom_llm_provider", "last_message_type", "expected_role", "expect_prefix"),
        [
            # Mistral AI provider: only assistant messages get prefix
            ("mistral", "assistant", "assistant", True),
            ("mistral", "user", "user", False),
            ("mistral", "system", "system", False),
            # Other providers: no prefix regardless of message type
            ("anthropic", "assistant", "assistant", False),
            ("vertex_ai", "assistant", "assistant", False),
            ("fireworks_ai", "assistant", "assistant", False),
            ("bedrock", "assistant", "assistant", False),
            (None, "assistant", "assistant", False),
        ],
    )
    def test_create_message_dicts_prefix_behavior(
        self, custom_llm_provider, last_message_type, expected_role, expect_prefix
    ):
        """Verify prefix is only added when using Mistral AI provider and last message is assistant."""
        chat = ChatLiteLLM(model="test-model", custom_llm_provider=custom_llm_provider)

        # Build messages based on last_message_type
        if last_message_type == "assistant":
            messages = [HumanMessage(content="Hello"), AIMessage(content="Thought:")]
        elif last_message_type == "user":
            messages = [HumanMessage(content="Hello")]
        else:  # system
            messages = [SystemMessage(content="You are a helpful assistant.")]

        message_dicts, _ = chat._create_message_dicts(messages, stop=None)

        assert message_dicts[-1]["role"] == expected_role
        if expect_prefix:
            assert message_dicts[-1]["prefix"] is True
        else:
            assert "prefix" not in message_dicts[-1]

    def test_create_message_dicts_empty_messages(self):
        """Verify no error when messages list is empty."""
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="mistral")
        message_dicts, _ = chat._create_message_dicts([], stop=None)
        assert message_dicts == []


class TestMistralContinueFinalMessage:
    """Tests for continue_final_message support for Mistral models on OpenAI-compatible endpoints (e.g. vLLM)."""

    @pytest.mark.parametrize(
        ("custom_llm_provider", "model", "last_message_type", "expect_continue"),
        [
            # custom_openai + Mistral model: only assistant messages get continue_final_message
            (
                "custom_openai",
                "mistralai/Ministral-3-3B-Instruct-2512",
                "assistant",
                True,
            ),
            ("hosted_vllm", "mistralai/Mistral-7B-Instruct-v0.3", "assistant", True),
            ("custom_openai", "mistralai/Ministral-3-3B-Instruct-2512", "user", False),
            (
                "custom_openai",
                "mistralai/Ministral-3-3B-Instruct-2512",
                "system",
                False,
            ),
            # custom_openai + non-Mistral model: no continue_final_message
            ("custom_openai", "meta-llama/Llama-3-8B-Instruct", "assistant", False),
            ("custom_openai", "Qwen/Qwen2.5-7B-Instruct", "assistant", False),
            # Other providers: no continue_final_message regardless of model/message type
            ("mistral", "mistralai/Ministral-3-3B-Instruct-2512", "assistant", False),
            (None, "mistralai/Ministral-3-3B-Instruct-2512", "assistant", True),
        ],
    )
    def test_create_message_dicts_continue_final_message_behavior(
        self, custom_llm_provider, model, last_message_type, expect_continue
    ):
        """Verify continue_final_message is added for Mistral models with trailing assistant message."""
        chat = ChatLiteLLM(model=model, custom_llm_provider=custom_llm_provider)

        if last_message_type == "assistant":
            messages = [HumanMessage(content="Hello"), AIMessage(content="Thought:")]
        elif last_message_type == "user":
            messages = [HumanMessage(content="Hello")]
        else:  # system
            messages = [SystemMessage(content="You are a helpful assistant.")]

        _, params = chat._create_message_dicts(messages, stop=None)

        if expect_continue:
            assert params.get("extra_body", {}).get("continue_final_message") is True
            assert params.get("extra_body", {}).get("add_generation_prompt") is False
        else:
            assert not params.get("extra_body", {}).get("continue_final_message")
            assert "add_generation_prompt" not in params.get("extra_body", {})

    def test_create_message_dicts_empty_messages(self):
        """Verify no error when messages list is empty."""
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="custom_openai")
        _, params = chat._create_message_dicts([], stop=None)
        assert not params.get("extra_body", {}).get("continue_final_message")


class TestClaude46PrefillCompat:
    @pytest.mark.parametrize(
        ("model", "expect_rewrite"),
        [
            # Claude 4.6+: prefill rejected, rewritten as user context
            pytest.param("claude-sonnet-4-6", True, id="4.6-sonnet-direct"),
            pytest.param("claude-opus-4-7", True, id="4.7-opus-direct"),
            pytest.param(
                "anthropic/claude-sonnet-4-6", True, id="4.6-sonnet-litellm-anthropic"
            ),
            pytest.param(
                "vertex_ai/claude-sonnet-4-6", True, id="4.6-sonnet-litellm-vertex"
            ),
            pytest.param(
                "bedrock/global.anthropic.claude-sonnet-4-6",
                True,
                id="4.6-sonnet-bedrock",
            ),
            pytest.param(
                "bedrock/global.anthropic.claude-opus-4-6-v1",
                True,
                id="4.6-opus-bedrock",
            ),
            pytest.param(
                "bedrock/global.anthropic.claude-opus-4-7",
                True,
                id="4.7-opus-bedrock",
            ),
            # Claude <= 4.5: prefill supported, payload untouched
            pytest.param("claude-sonnet-4-5@20250929", False, id="4.5-sonnet-vertex"),
            pytest.param("claude-haiku-4-5@20251001", False, id="4.5-haiku-vertex"),
            pytest.param("claude-opus-4-5@20251101", False, id="4.5-opus-vertex"),
            pytest.param(
                "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                False,
                id="4.5-sonnet-bedrock",
            ),
        ],
    )
    def test_claude_46_prefill_workaround(self, model, expect_rewrite):
        chat = ChatLiteLLM(model=model)
        messages = [HumanMessage(content="hello"), AIMessage(content="Thought: ")]

        dicts, _ = chat._create_message_dicts(messages, stop=None)

        if expect_rewrite:
            expected = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{PREVIOUS_ASSISTANT_CONTEXT_PREFIX}Thought: ",
                    }
                ],
            }
        else:
            expected = {"role": "assistant", "content": "Thought: "}
        assert dicts[-1] == expected


class TestGpt5MaxCompletionTokens:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-5-nano"],
    )
    def test_gpt_5_on_custom_openai_moves_to_extra_body(self, model):
        kwargs = {
            "model": model,
            "custom_llm_provider": "custom_openai",
            "max_tokens": 12,
        }

        _force_gpt_5_max_completion_tokens(kwargs)

        assert kwargs["custom_llm_provider"] == "custom_openai"
        assert "max_tokens" not in kwargs
        assert kwargs["extra_body"] == {"max_completion_tokens": 12}

    @pytest.mark.parametrize(
        "provider",
        ["azure", "custom_openai", "openai", "fireworks_ai", None],
    )
    def test_gpt_5_moves_to_extra_body_regardless_of_provider(self, provider):
        kwargs = {
            "model": "gpt-5",
            "custom_llm_provider": provider,
            "max_tokens": 12,
        }

        _force_gpt_5_max_completion_tokens(kwargs)

        assert kwargs["custom_llm_provider"] == provider
        assert "max_tokens" not in kwargs
        assert kwargs["extra_body"] == {"max_completion_tokens": 12}

    @pytest.mark.parametrize(
        "model", ["gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet", "gpt-5-chat"]
    )
    @pytest.mark.parametrize("provider", ["custom_openai", "azure", "openai"])
    def test_non_gpt_5_is_left_alone(self, model, provider):
        kwargs = {
            "model": model,
            "custom_llm_provider": provider,
            "max_tokens": 12,
        }

        _force_gpt_5_max_completion_tokens(kwargs)

        assert kwargs["max_tokens"] == 12
        assert "extra_body" not in kwargs

    def test_missing_model_is_noop(self):
        kwargs = {"custom_llm_provider": "custom_openai", "max_tokens": 12}

        _force_gpt_5_max_completion_tokens(kwargs)

        assert kwargs["max_tokens"] == 12

    def test_no_max_tokens_is_noop(self):
        kwargs = {"model": "gpt-5", "custom_llm_provider": "custom_openai"}

        _force_gpt_5_max_completion_tokens(kwargs)

        assert "extra_body" not in kwargs

    def test_existing_extra_body_is_preserved(self):
        kwargs = {
            "model": "gpt-5",
            "custom_llm_provider": "custom_openai",
            "max_tokens": 12,
            "extra_body": {"foo": "bar"},
        }

        _force_gpt_5_max_completion_tokens(kwargs)

        assert kwargs["extra_body"] == {"foo": "bar", "max_completion_tokens": 12}

    @pytest.mark.asyncio
    async def test_acompletion_with_retry_applies_before_calling_super(self):
        chat = ChatLiteLLM(model="gpt-5", custom_llm_provider="custom_openai")

        with patch.object(
            _LChatLiteLLM,
            "acompletion_with_retry",
            new=AsyncMock(return_value="ok"),
        ) as super_call:
            result = await chat.acompletion_with_retry(
                model="gpt-5", custom_llm_provider="custom_openai", max_tokens=12
            )

        assert result == "ok"
        super_kwargs = super_call.call_args.kwargs
        assert super_kwargs["custom_llm_provider"] == "custom_openai"
        assert "max_tokens" not in super_kwargs
        assert super_kwargs["extra_body"] == {"max_completion_tokens": 12}


class TestBedrockGuardrailConfig:
    @pytest.fixture(name="guardrail_config")
    def guardrail_config_fixture(self):
        return ConfigBedrockGuardrail(
            guardrailIdentifier="abc123",
            guardrailVersion="1",
            trace="enabled",
        )

    @staticmethod
    def _expected_guardrail_config():
        return {
            "guardrailIdentifier": "abc123",
            "guardrailVersion": "1",
            "trace": "enabled",
        }

    @pytest.mark.parametrize("provider", sorted(BEDROCK_GUARDRAIL_PROVIDERS))
    def test_default_params_includes_guardrail_config(self, guardrail_config, provider):
        chat = ChatLiteLLM(
            model="test-model",
            custom_llm_provider=provider,
            bedrock_guardrail_config=guardrail_config,
        )

        params = chat._default_params

        assert params["guardrailConfig"] == self._expected_guardrail_config()

    def test_default_params_no_guardrail_when_not_bedrock(self, guardrail_config):
        chat = ChatLiteLLM(
            model="test-model",
            custom_llm_provider="anthropic",
            bedrock_guardrail_config=guardrail_config,
        )

        params = chat._default_params

        assert "guardrailConfig" not in params

    def test_default_params_no_guardrail_when_config_is_none(self):
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="bedrock")

        params = chat._default_params

        assert "guardrailConfig" not in params

    @pytest.mark.parametrize("provider", sorted(BEDROCK_GUARDRAIL_PROVIDERS))
    @pytest.mark.asyncio
    async def test_agenerate_passes_guardrail_config(self, guardrail_config, provider):
        chat = ChatLiteLLM(
            model="test-model",
            custom_llm_provider=provider,
            bedrock_guardrail_config=guardrail_config,
        )
        message = HumanMessage(content="Hello")

        mock_response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        with patch(
            "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
            new=AsyncMock(return_value=mock_response),
        ) as mock_acompletion:
            await chat._agenerate(messages=[message])

            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["guardrailConfig"] == self._expected_guardrail_config()
