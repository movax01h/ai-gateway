import time
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ai_gateway.models.agentic_mock import AgenticFakeModel, ResponseHandler


class TestAgenticFakeModel:  # pylint: disable=too-many-public-methods
    @pytest.fixture
    def model(self):
        return AgenticFakeModel(auto_tool_approval=False, use_last_human_message=False)

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
        assert model._is_agentic_mock_model is True
        assert model._is_auto_approved_by_agentic_mock_model is False

    def test_bind_tools_returns_self(self, model):
        result = model.bind_tools()
        assert result is model

        result = model.bind_tools("tool1", "tool2", param=True)
        assert result is model

    @pytest.mark.asyncio
    async def test_astream_returns_correct_chunk_types(self, model):
        """Test that _astream returns proper chunk types (integration test)."""
        messages = [
            HumanMessage(
                content="<response stream='true' chunk_delay_ms='10'>Hello world</response>"
            )
        ]

        chunks = []
        async for chunk in model._astream(messages):
            assert isinstance(chunk, ChatGenerationChunk)
            assert isinstance(chunk.message, AIMessageChunk)
            chunks.append(chunk.message.content)

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

    @pytest.mark.asyncio
    async def test_astream_with_streaming_tool_calls_in_chunks(self, model):
        messages = [
            HumanMessage(
                content="""
                <response stream="true">
                Analyzing <tool_calls>[{"name": "search", "args": {"query": "test"}}]</tool_calls>
                </response>'
                """
            )
        ]

        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk)

        # Should have text chunks + tool call chunk
        # With short content, we get: 1 text chunk + 1 tool call chunk
        assert len(chunks) >= 2

        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) == 1  # "Analyzing"

        tool_chunks = [
            c
            for c in chunks
            if hasattr(c.message, "tool_call_chunks") and c.message.tool_call_chunks
        ]
        assert len(tool_chunks) >= 1  # Should have at least one chunk for the tool call

    def test_stream_with_streaming_tool_calls_in_chunks(self, model):
        messages = [
            HumanMessage(
                content="""
                <response stream="true">
                Analyzing <tool_calls>[{"name": "search", "args": {"query": "test"}}]</tool_calls>
                </response>'
                """
            )
        ]

        chunks = []
        for chunk in model._stream(messages):
            chunks.append(chunk)

        # Should have text chunks + tool call chunks
        assert len(chunks) > 1  # At least text chunks + tool call chunks

        text_chunks = [c for c in chunks if c.message.content]
        assert len(text_chunks) == 1  # "Analyzing"

        tool_chunks = [
            c
            for c in chunks
            if hasattr(c.message, "tool_call_chunks") and c.message.tool_call_chunks
        ]
        assert len(tool_chunks) > 0  # Should have multiple chunks for the tool call


class TestResponseHandler:  # pylint: disable=too-many-public-methods
    @pytest.mark.parametrize("use_last_human_message", [True, False])
    def test_response_handler_simple_response(self, use_last_human_message):
        messages = [
            HumanMessage(content="Task: <response>Simple response text</response>")
        ]

        handler = ResponseHandler(
            messages, use_last_human_message=use_last_human_message
        )
        response = handler.get_next_response()

        assert response.content == "Simple response text"
        assert not response.tool_calls
        assert response.latency_ms == 0

    @pytest.mark.parametrize("use_last_human_message", [True, False])
    def test_response_handler_with_tool_calls(self, use_last_human_message):
        messages = [
            HumanMessage(
                content="Task: <response>Response with "
                '<tool_calls>[{"name": "search", "args": {"q": "test"}}]</tool_calls>tool call</response>'
            )
        ]

        handler = ResponseHandler(
            messages, use_last_human_message=use_last_human_message
        )
        response = handler.get_next_response()

        assert response.content == "Response with tool call"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search"
        assert response.tool_calls[0]["args"] == {"q": "test"}

    @pytest.mark.parametrize("use_last_human_message", [True, False])
    def test_response_handler_prevents_infinite_loop(self, use_last_human_message):
        messages = [
            HumanMessage(
                content="<response>Task with tool call "
                '<tool_calls>[{"name": "search", "args": {}}]</tool_calls></response>'
            )
        ]

        handler = ResponseHandler(
            messages, use_last_human_message=use_last_human_message
        )

        # First call returns the response with tool call
        response = handler.get_next_response()
        assert "Task with tool call" in response.content
        assert len(response.tool_calls) == 1

        # Second call returns exhausted message with no tool calls (prevents infinite loop)
        response = handler.get_next_response()
        assert "all scripted responses exhausted" in response.content
        assert not response.tool_calls

    @pytest.mark.parametrize("use_last_human_message", [True, False])
    def test_response_handler_multiple_latencies(self, use_last_human_message):
        messages = [
            HumanMessage(
                content="""
                <response latency_ms='100'>Fast response</response>
                <response latency_ms='1000'>Slow response</response>
                <response>No latency response</response>
            """
            )
        ]

        handler = ResponseHandler(
            messages, use_last_human_message=use_last_human_message
        )

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
            ResponseHandler(messages, use_last_human_message=False)

    def test_response_handler_latency_parsing_edge_cases(self):
        messages = [
            HumanMessage(content="<response latency_ms='invalid'>Test</response>")
        ]
        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()
        assert response.latency_ms == 0

    @pytest.mark.parametrize(
        "messages",
        [
            [HumanMessage(content="<response>Unclosed tag")],
            [HumanMessage(content="Invalid <response tag")],
        ],
    )
    def test_response_handler_invalid_xml_returns_error_response(self, messages):
        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()

        assert "invalid xml" in response.content.lower()

    @pytest.mark.parametrize(
        "messages,use_last_human_message,expected_content",
        [
            (
                [
                    HumanMessage(content="<response>test1</response>"),
                    HumanMessage(content="<response>test2</response>"),
                ],
                False,
                "test1",
            ),
            (
                [
                    HumanMessage(content="<response>test1</response>"),
                    HumanMessage(content="<response>test2</response>"),
                ],
                True,
                "test2",
            ),
        ],
    )
    def test_response_handler_with_use_last_human_message(
        self, messages, use_last_human_message, expected_content
    ):
        handler = ResponseHandler(
            messages, use_last_human_message=use_last_human_message
        )
        response = handler.get_next_response()

        assert expected_content in response.content.lower()

    @pytest.mark.parametrize(
        "messages,expected_response",
        [
            (
                [
                    HumanMessage(
                        content="""
            <responses>
                <response>
                    Create an issue for an awesome feature.
                    <tool_calls>[{"name": "create_work_item", "args": {"project_id": 1000000, "title": "Implement feature", "type_name": "Issue"}}]</tool_calls>
                </response>
                <response>Issue created</response>
            </responses>
                                """
                    ),
                ],
                "Create an issue for an awesome feature.",
            ),
            (
                [
                    HumanMessage(
                        content="""
<responses>
    <response>
        Create an issue for an awesome feature.
        <tool_calls>[{"name": "create_work_item", "args": {"project_id": 1000000, "title": "Implement feature", "type_name": "Issue"}}]</tool_calls>
    </response>
    <response>Issue created</response>
</responses>
                                """
                    ),
                    AIMessage(
                        content="Create an issue for an awesome feature.",
                        tool_calls=[
                            ToolCall(
                                name="create_work_item",
                                args={
                                    "project_id": 1000000,
                                    "title": "Implement feature",
                                    "type_name": "Issue",
                                },
                                id="call_1",
                                type="tool_call",
                            )
                        ],
                    ),
                ],
                "Issue created",
            ),
        ],
    )
    def test_response_handler_ignore_already_responded(
        self, messages, expected_response
    ):
        handler = ResponseHandler(messages, use_last_human_message=True)
        response = handler.get_next_response()

        assert expected_response == response.content

    @pytest.mark.parametrize(
        "messages",
        [
            [],
            ["not a HumanMessage"],
            [HumanMessage(content="")],
            [HumanMessage(content="Just regular text without tags")],
            [HumanMessage(content=["not", "a", "string"])],
            [AIMessage(content="<response>AIMessage is not user input</response>")],
        ],
    )
    def test_response_handler_no_response_tags_in_user_input(self, messages):
        handler = ResponseHandler(messages, use_last_human_message=False)

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

        handler = ResponseHandler(messages, use_last_human_message=False)

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

    def test_response_handler_file_based_response(self, tmp_path):
        response_file = tmp_path / "test_response.txt"
        response_file.write_text("Response from file")

        messages = [HumanMessage(content=f'<response file="{response_file}" />')]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()

        assert response.content == "Response from file"
        assert not response.tool_calls

    def test_response_handler_file_based_response_with_tool_calls(self, tmp_path):
        response_file = tmp_path / "test_response_with_tools.txt"
        response_file.write_text(
            "I will search for information.\n"
            '<tool_calls>[{"name": "search", "args": {"query": "test"}}]</tool_calls>'
        )

        messages = [HumanMessage(content=f'<response file="{response_file}" />')]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()

        assert response.content == "I will search for information."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search"
        assert response.tool_calls[0]["args"] == {"query": "test"}

    def test_response_handler_file_caching(self, tmp_path):
        response_file = tmp_path / "cached_response.txt"
        response_file.write_text("Cached content")

        ResponseHandler._file_cache.clear()

        # First handler loads the file
        messages1 = [HumanMessage(content=f'<response file="{response_file}" />')]
        handler1 = ResponseHandler(messages1, use_last_human_message=False)
        response1 = handler1.get_next_response()

        assert str(response_file) in ResponseHandler._file_cache

        # Modify the file
        response_file.write_text("Modified content")

        # Second handler should use cached content
        messages2 = [HumanMessage(content=f'<response file="{response_file}" />')]
        handler2 = ResponseHandler(messages2, use_last_human_message=False)
        response2 = handler2.get_next_response()

        # Should still have original cached content
        assert response1.content == "Cached content"
        assert response2.content == "Cached content"

        # Clear cache for other tests
        ResponseHandler._file_cache.clear()

    def test_response_handler_file_not_found(self):
        messages = [HumanMessage(content='<response file="nonexistent_file.txt" />')]

        with pytest.raises(FileNotFoundError, match="Response file not found"):
            ResponseHandler(messages, use_last_human_message=False)

    def test_response_handler_template_variables(self, tmp_path):
        content_file = tmp_path / "template_vars.txt"
        content_file.write_text(
            "Project ID is {{ projectId }} in {{ namespace }}\n"
            """
            <tool_calls>
            [{"name": "list_repository_tree", "args": {"project_id": {{ projectId }}, "recursive": true}}]
            </tool_calls>
            """
        )

        messages = [
            HumanMessage(
                content=f"""
                {{% set projectId = 1000001 %}}
                {{% set namespace = "test-namespace" %}}
                <response file="{content_file}" />
                """
            )
        ]

        handler = ResponseHandler(messages, use_last_human_message=False)

        response = handler.get_next_response()
        assert response.content == "Project ID is 1000001 in test-namespace"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["args"]["project_id"] == 1000001

    def test_response_handler_file_with_attributes(self, tmp_path):
        response_file = tmp_path / "response_with_attrs.txt"
        response_file.write_text("Delayed streaming response")

        messages = [
            HumanMessage(
                content=f'<response file="{response_file}" stream="true" chunk_delay_ms="50" latency_ms="100" />'
            )
        ]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()

        assert response.content == "Delayed streaming response"
        assert response.stream is True
        assert response.chunk_delay_ms == 50
        assert response.latency_ms == 100

    def test_response_handler_file_with_xml_special_characters(self, tmp_path):
        # Create a file with content that would break XML parsing
        response_file = tmp_path / "code_with_xml_chars.txt"
        response_file.write_text(
            "Creating a Python function.\n"
            """
            '<tool_calls>
            [{"name": "create_file", "args": {"code": "if x < 10 and y > 5:\\n    print(\\"x & y are valid\\")"}}]
            </tool_calls>'
            """
        )

        messages = [HumanMessage(content=f'<response file="{response_file}" />')]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()

        assert response.content.startswith("Creating a Python function.")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "create_file"
        assert "x < 10" in response.tool_calls[0]["args"]["code"]
        assert "y > 5" in response.tool_calls[0]["args"]["code"]
        assert "x & y" in response.tool_calls[0]["args"]["code"]

    def test_extract_template_variables_with_invalid_jinja(self):
        messages = [
            HumanMessage(
                content="""
                {% set invalid syntax here %}
                <response>Test</response>
                """
            )
        ]
        handler = ResponseHandler(messages, use_last_human_message=False)
        # Should still create handler with empty template_vars
        assert handler.template_vars == {}
        response = handler.get_next_response()
        assert response.content == "Test"

    def test_render_template_with_jinja_error(self, tmp_path):
        response_file = tmp_path / "jinja_error.txt"
        response_file.write_text("Project {% if invalid syntax %}")

        messages = [
            HumanMessage(
                content=f"""
                {{% set projectId = 123 %}}
                <response file="{response_file}" />
                """
            )
        ]
        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()
        # Should return content as-is when rendering fails
        assert "Project" in response.content

    def test_tool_calls_without_closing_tag(self, tmp_path):
        response_file = tmp_path / "unclosed_tool_calls.txt"
        response_file.write_text('Text before <tool_calls>[{"name": "search"}]')

        messages = [HumanMessage(content=f'<response file="{response_file}" />')]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()
        # Should extract text before the unclosed tag
        assert "Text before" in response.content
        # Tool calls should not be parsed due to missing closing tag
        assert len(response.tool_calls) == 0

    def test_stream_tool_calls_with_large_json(self):
        model = AgenticFakeModel(auto_tool_approval=False, use_last_human_message=False)
        # Create a tool call with large args to trigger multiple chunks (chunk_size is 100)
        large_data = "x" * 150

        messages = [
            HumanMessage(
                content=f"""
                <response stream="true">
                    Processing <tool_calls>
                        [{{"name": "process_data", "args": {{"data": "{large_data}"}}}}]
                    </tool_calls>
                </response>
                """
            )
        ]

        chunks = []
        for chunk in model._stream(messages):
            chunks.append(chunk)

        # Should have text chunk + multiple tool call chunks
        assert len(chunks) > 2

        tool_chunks = [
            c
            for c in chunks
            if hasattr(c.message, "tool_call_chunks") and c.message.tool_call_chunks
        ]
        assert len(tool_chunks) > 1

        # First chunk should have name
        assert tool_chunks[0].message.tool_call_chunks[0]["name"] == "process_data"
        # Subsequent chunks should have name=None
        if len(tool_chunks) > 1:
            assert tool_chunks[1].message.tool_call_chunks[0]["name"] is None

    def test_extract_text_from_nested_elements(self):
        messages = [
            HumanMessage(
                content="""
                <response>
                    Text before
                    <custom_tag>Nested content</custom_tag>
                    Text after
                </response>
                """
            )
        ]

        handler = ResponseHandler(messages, use_last_human_message=False)
        response = handler.get_next_response()
        # Should extract all text including nested elements
        assert "Text before" in response.content
        assert "Nested content" in response.content
        assert "Text after" in response.content
