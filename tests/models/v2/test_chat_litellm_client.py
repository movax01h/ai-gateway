import time
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM
from ai_gateway.vendor.langchain_litellm.litellm import _create_usage_metadata


@pytest.mark.asyncio
async def test_astream_with_stream_options_and_stop_reason():
    """Test that stream_options is added correctly and finish_reason is extracted from stop_reason."""

    message = HumanMessage(content="Hello")

    # Mock the raw LiteLLM response chunks
    mock_chunks = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "delta": {"content": " world"},
                    "finish_reason": None,
                    "index": 0,
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

    async def mock_acompletion(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    with patch(
        "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
        new=AsyncMock(return_value=mock_acompletion()),
    ):
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

    async def mock_acompletion(*args, **kwargs):
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
async def test_fireworks_logprobs_in_non_streaming():
    """Test that Fireworks logprobs are correctly extracted and returned as score in non-streaming mode."""

    message = HumanMessage(content="def hello():")

    # Mock Fireworks non-streaming response with logprobs
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "    print('Hello, World!')",
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "token_logprobs": [-0.5, -1.2, -0.8],
                    "tokens": ["    ", "print", "('Hello, World!')"],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    with patch(
        "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
        new=AsyncMock(return_value=mock_response),
    ):
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
async def test_fireworks_session_affinity_header():
    """Test that Fireworks session affinity header is correctly set."""

    message = HumanMessage(content="def hello():")

    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "    print('Hello')",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch(
        "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
        new=AsyncMock(return_value=mock_response),
    ) as mock_acompletion:
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

        await chat._agenerate(messages=[message], session_id="test-session-123")

        # Verify the call was made with session affinity header
        call_kwargs = mock_acompletion.call_args[1]
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"]["x-session-affinity"] == "test-session-123"


@pytest.mark.asyncio
async def test_fireworks_prompt_caching_disabled():
    """Test that Fireworks prompt caching can be disabled."""

    message = HumanMessage(content="def hello():")

    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "    print('Hello')",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch(
        "ai_gateway.vendor.langchain_litellm.litellm.ChatLiteLLM.acompletion_with_retry",
        new=AsyncMock(return_value=mock_response),
    ) as mock_acompletion:
        chat = ChatLiteLLM(model="test-model", custom_llm_provider="fireworks_ai")

        await chat._agenerate(messages=[message], using_cache="false")

        # Verify the call was made with prompt_cache_max_len=0
        call_kwargs = mock_acompletion.call_args[1]
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

    async def mock_acompletion(*args, **kwargs):
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

    with patch.object(
        chat.client, "acompletion", new=AsyncMock(side_effect=mock_acompletion)
    ):
        result = await chat._agenerate(messages=[message])

        assert call_count == 3  # Failed twice, succeeded on third
        assert result.generations[0].message.content == "success"


@pytest.mark.asyncio
async def test_non_fireworks_503_fails_immediately():
    """Verify 503 errors do NOT trigger retries for non-Fireworks providers."""
    message = HumanMessage(content="test")

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
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

    async def mock_acompletion(*args, **kwargs):
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

    async def mock_acompletion(*args, **kwargs):
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
            assert (
                0.8 <= delays[0] <= 1.5
            ), f"First delay should be ~1s, got {delays[0]}"
            assert (
                1.8 <= delays[1] <= 2.5
            ), f"Second delay should be ~2s, got {delays[1]}"
            assert (
                3.5 <= delays[2] <= 4.5
            ), f"Third delay should be ~4s, got {delays[2]}"
            assert (
                7.0 <= delays[3] <= 9.0
            ), f"Fourth delay should be ~8s, got {delays[3]}"


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
