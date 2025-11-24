import time
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ai_gateway.models.agentic_mock import AgenticFakeModel, ResponseHandler


class TestAgenticFakeModel:  # pylint: disable=too-many-public-methods
    @pytest.fixture
    def model(self):
        return AgenticFakeModel()

    @pytest.fixture
    def latency_messages(self):
        return [
            HumanMessage(
                content="<response latency_ms='100'>Delayed response</response>"
            )
        ]

    def test_generate_returns_chatresult_structure(self, model):
        messages = [HumanMessage(content="<response>Test response</response>")]

        result = model._generate(messages)

        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert isinstance(result.generations[0], ChatGeneration)

        ai_message = result.generations[0].message
        assert isinstance(ai_message, AIMessage)
        assert ai_message.content == "Test response"
        assert isinstance(ai_message.tool_calls, list)

    def test_generate_reuses_response_handler(self, model):
        messages = [
            HumanMessage(
                content="<response>First response</response><response>Second response</response>"
            )
        ]

        # First call creates the handler
        result1 = model._generate(messages)
        handler1 = model._response_handler

        # Second call should reuse the same handler
        result2 = model._generate(messages)
        handler2 = model._response_handler

        assert handler1 is handler2
        assert result1.generations[0].message.content == "First response"
        assert result2.generations[0].message.content == "Second response"

    def _assert_latency_simulation(self, result, start_time, end_time):
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms >= 90  # Allow some tolerance for timing

        assert result.generations[0].message.content == "Delayed response"

    @pytest.mark.asyncio
    async def test_agenerate_with_latency_simulation(self, model, latency_messages):
        start_time = time.time()
        result = await model._agenerate(latency_messages)
        end_time = time.time()
        self._assert_latency_simulation(result, start_time, end_time)

    def test_generate_with_latency_simulation(self, model, latency_messages):
        start_time = time.time()
        result = model._generate(latency_messages)
        end_time = time.time()
        self._assert_latency_simulation(result, start_time, end_time)

    def test_stream_with_latency_simulation(self, model, latency_messages):
        start_time = time.time()
        chunks = list(model._stream(latency_messages))
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms >= 90  # Allow some tolerance for timing

        assert len(chunks) == 1
        assert chunks[0].message.content == "Delayed response"

    @pytest.mark.asyncio
    async def test_astream_with_latency_simulation(self, model, latency_messages):
        start_time = time.time()
        chunks = []
        async for chunk in model._astream(latency_messages):
            chunks.append(chunk)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms >= 90  # Allow some tolerance for timing

        assert len(chunks) == 1
        assert chunks[0].message.content == "Delayed response"

    def test_model_properties(self, model):
        assert model._llm_type == "agentic-fake-provider"
        assert model._identifying_params == {"model": "agentic-fake-model"}

    def test_bind_tools_returns_self(self, model):
        result = model.bind_tools()
        assert result is model

        result = model.bind_tools("tool1", "tool2", param=True)
        assert result is model

    @pytest.mark.asyncio
    async def test_astream_with_streaming_enabled(self, model):
        messages = [
            HumanMessage(
                content="<response stream='true' chunk_delay_ms='10'>Hello world test</response>"
            )
        ]

        chunks = []
        async for chunk in model._astream(messages):
            assert isinstance(chunk, ChatGenerationChunk)
            assert isinstance(chunk.message, AIMessageChunk)
            chunks.append(chunk.message.content)

        # Should have 3 chunks: "Hello", " world", " test"
        assert len(chunks) == 3
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"
        assert chunks[2] == " test"

    @pytest.mark.asyncio
    async def test_astream_with_streaming_and_tool_calls(self, model):
        messages = [
            HumanMessage(
                content='<response stream="true">Analyze this <tool_calls>[{"name": "search"}]</tool_calls></response>'
            )
        ]

        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk)

        # Should have text chunks + final chunk with tool calls
        assert len(chunks) == 3  # "Analyze", " this", and tool call chunk

        # Last chunk should have tool calls
        assert chunks[-1].message.tool_calls
        assert chunks[-1].message.tool_calls[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_astream_without_streaming(self, model):
        messages = [
            HumanMessage(content="<response>Complete response at once</response>")
        ]

        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk)

        # Should have only 1 chunk with complete response
        assert len(chunks) == 1
        assert chunks[0].message.content == "Complete response at once"

    def test_stream_with_streaming_enabled(self, model):
        messages = [
            HumanMessage(content="<response stream='true'>Hello world</response>")
        ]

        chunks = []
        for chunk in model._stream(messages):
            assert isinstance(chunk, ChatGenerationChunk)
            assert isinstance(chunk.message, AIMessageChunk)
            chunks.append(chunk.message.content)

        # Should have 2 chunks: "Hello", " world"
        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"

    def test_generate_invokes_callback_on_llm_end(self, model):
        messages = [HumanMessage(content="<response>Test response</response>")]
        mock_run_manager = Mock()

        model._generate(messages, run_manager=mock_run_manager)

        mock_run_manager.on_llm_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_agenerate_invokes_callback_on_llm_end(self, model):
        messages = [HumanMessage(content="<response>Test response</response>")]
        mock_run_manager = AsyncMock()

        await model._agenerate(messages, run_manager=mock_run_manager)

        mock_run_manager.on_llm_end.assert_called_once()

    def test_stream_invokes_callback_on_llm_new_token_streaming(self, model):
        messages = [
            HumanMessage(content="<response stream='true'>Hello world</response>")
        ]
        mock_run_manager = Mock()

        list(model._stream(messages, run_manager=mock_run_manager))

        assert mock_run_manager.on_llm_new_token.call_count == 2
        calls = mock_run_manager.on_llm_new_token.call_args_list
        assert calls[0][0][0] == "Hello"
        assert calls[1][0][0] == " world"

    def test_stream_invokes_callback_on_llm_new_token_non_streaming(self, model):
        messages = [HumanMessage(content="<response>Complete response</response>")]
        mock_run_manager = Mock()

        list(model._stream(messages, run_manager=mock_run_manager))

        mock_run_manager.on_llm_new_token.assert_called_once_with("Complete response")

    @pytest.mark.asyncio
    async def test_astream_invokes_callback_on_llm_new_token_streaming(self, model):
        messages = [
            HumanMessage(content="<response stream='true'>Hello world test</response>")
        ]
        mock_run_manager = AsyncMock()

        async for _ in model._astream(messages, run_manager=mock_run_manager):
            pass  # Just consume the generator

        assert mock_run_manager.on_llm_new_token.call_count == 3
        calls = mock_run_manager.on_llm_new_token.call_args_list
        assert calls[0][0][0] == "Hello"
        assert calls[1][0][0] == " world"
        assert calls[2][0][0] == " test"

    @pytest.mark.asyncio
    async def test_astream_invokes_callback_on_llm_new_token_non_streaming(self, model):
        messages = [HumanMessage(content="<response>Complete response</response>")]
        mock_run_manager = AsyncMock()

        async for _ in model._astream(messages, run_manager=mock_run_manager):
            pass  # Just consume the generator

        mock_run_manager.on_llm_new_token.assert_called_once_with("Complete response")

    def test_stream_with_tool_calls_invokes_callbacks(self, model):
        """Test that _stream invokes callbacks correctly with tool calls."""
        messages = [
            HumanMessage(
                content='<response stream="true">Analyze <tool_calls>[{"name": "search"}]</tool_calls></response>'
            )
        ]
        mock_run_manager = Mock()

        list(model._stream(messages, run_manager=mock_run_manager))

        # Verify on_llm_new_token was called for the text chunk
        assert mock_run_manager.on_llm_new_token.call_count == 1
        assert mock_run_manager.on_llm_new_token.call_args[0][0] == "Analyze"

    @pytest.mark.asyncio
    async def test_astream_with_tool_calls_invokes_callbacks(self, model):
        """Test that _astream invokes callbacks correctly with tool calls."""
        messages = [
            HumanMessage(
                content='<response stream="true">Analyze this <tool_calls>[{"name": "search"}]</tool_calls></response>'
            )
        ]
        mock_run_manager = AsyncMock()

        async for _ in model._astream(messages, run_manager=mock_run_manager):
            pass  # Just consume the generator

        # Verify on_llm_new_token was called for each text chunk
        assert mock_run_manager.on_llm_new_token.call_count == 2
        calls = mock_run_manager.on_llm_new_token.call_args_list
        assert calls[0][0][0] == "Analyze"
        assert calls[1][0][0] == " this"

    @pytest.mark.asyncio
    async def test_astream_with_chunk_delay(self, model):
        messages = [
            HumanMessage(
                content="<response stream='true' chunk_delay_ms='50'>Hello world</response>"
            )
        ]

        start_time = time.time()
        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk.message.content)
        end_time = time.time()

        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"

        # Verify delay was applied (at least 40ms to allow for timing variance)
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms >= 40

    def test_stream_with_chunk_delay(self, model):
        messages = [
            HumanMessage(
                content="<response stream='true' chunk_delay_ms='50'>Hello world</response>"
            )
        ]

        start_time = time.time()
        chunks = []
        for chunk in model._stream(messages):
            chunks.append(chunk.message.content)
        end_time = time.time()

        assert len(chunks) == 2
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"

        # Verify delay was applied (at least 40ms to allow for timing variance)
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms >= 40


class TestResponseHandler:
    def test_response_handler_simple_response(self):
        messages = [
            HumanMessage(content="Task: <response>Simple response text</response>")
        ]

        handler = ResponseHandler(messages)
        response = handler.get_next_response()

        assert response.content == "Simple response text"
        assert not response.tool_calls
        assert response.latency_ms == 0

    def test_response_handler_with_tool_calls(self):
        messages = [
            HumanMessage(
                content="Task: <response>Response with "
                '<tool_calls>[{"name": "search", "args": {"q": "test"}}]</tool_calls>tool call</response>'
            )
        ]

        handler = ResponseHandler(messages)
        response = handler.get_next_response()

        assert response.content == "Response with tool call"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search"
        assert response.tool_calls[0]["args"] == {"q": "test"}

    def test_response_handler_prevents_infinite_loop(self):
        messages = [
            HumanMessage(
                content="<response>Task with tool call "
                '<tool_calls>[{"name": "search", "args": {}}]</tool_calls></response>'
            )
        ]

        handler = ResponseHandler(messages)

        # First call returns the response with tool call
        response = handler.get_next_response()
        assert "Task with tool call" in response.content
        assert len(response.tool_calls) == 1

        # Second call returns exhausted message with no tool calls (prevents infinite loop)
        response = handler.get_next_response()
        assert "all scripted responses exhausted" in response.content
        assert not response.tool_calls

    def test_response_handler_multiple_latencies(self):
        messages = [
            HumanMessage(
                content="""
                <response latency_ms='100'>Fast response</response>
                <response latency_ms='1000'>Slow response</response>
                <response>No latency response</response>
            """
            )
        ]

        handler = ResponseHandler(messages)

        response1 = handler.get_next_response()
        assert response1.content == "Fast response"
        assert response1.latency_ms == 100

        response2 = handler.get_next_response()
        assert response2.content == "Slow response"
        assert response2.latency_ms == 1000

        response3 = handler.get_next_response()
        assert response3.content == "No latency response"
        assert response3.latency_ms == 0

        assert "all scripted responses exhausted" in handler.get_next_response().content

    @pytest.mark.parametrize(
        "content,expected_error",
        [
            (
                '<response>Response with <tool_calls>{"name": "search"}</tool_calls></response>',
                "Tool calls must be an array",
            ),
            (
                '<response>Response with <tool_calls>[{"args": {"q": "test"}}]</tool_calls></response>',
                "Tool call missing required 'name' field",
            ),
            (
                '<response>Response with <tool_calls>[{"name": "search", invalid}]</tool_calls></response>',
                "Invalid JSON in tool_calls",
            ),
            (
                '<response>Response with <tool_calls>["not_an_object"]</tool_calls></response>',
                "Each tool call must be an object",
            ),
        ],
    )
    def test_response_handler_invalid_tool_calls_raise_error(
        self, content, expected_error
    ):
        messages = [HumanMessage(content=content)]
        with pytest.raises(ValueError, match=expected_error):
            ResponseHandler(messages)

    def test_response_handler_latency_parsing_edge_cases(self):
        messages = [
            HumanMessage(content="<response latency_ms='invalid'>Test</response>")
        ]
        handler = ResponseHandler(messages)
        response = handler.get_next_response()
        assert response.latency_ms == 0

    @pytest.mark.parametrize(
        "messages",
        [
            [],
            ["not a HumanMessage"],
            [HumanMessage(content="")],
            [HumanMessage(content="Just regular text without tags")],
            [HumanMessage(content="Invalid <response tag")],
            [HumanMessage(content=["not", "a", "string"])],
            [AIMessage(content="<response>AIMessage is not user input</response>")],
        ],
    )
    def test_response_handler_no_response_tags_in_user_input(self, messages):
        handler = ResponseHandler(messages)

        response = handler.get_next_response()
        assert response.content == "mock response (no response tag specified)"
        assert not response.tool_calls

    def test_response_handler_streaming_attributes(self):
        messages = [
            HumanMessage(
                content="""
                <response stream='true' chunk_delay_ms='100'>Streaming response</response>
                <response stream="false">Non-streaming response</response>
                <response>Default response</response>
            """
            )
        ]

        handler = ResponseHandler(messages)

        response1 = handler.get_next_response()
        assert response1.content == "Streaming response"
        assert response1.stream is True
        assert response1.chunk_delay_ms == 100

        response2 = handler.get_next_response()
        assert response2.content == "Non-streaming response"
        assert response2.stream is False
        assert response2.chunk_delay_ms == 0

        response3 = handler.get_next_response()
        assert response3.content == "Default response"
        assert response3.stream is False
        assert response3.chunk_delay_ms == 0
