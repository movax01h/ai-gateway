import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_gateway.models.agentic_mock import AgenticFakeModel, ResponseHandler


class TestAgenticFakeModel:
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

    def test_model_properties(self, model):
        assert model._llm_type == "agentic-fake-provider"
        assert model._identifying_params == {"model": "agentic-fake-model"}

    def test_bind_tools_returns_self(self, model):
        result = model.bind_tools()
        assert result is model

        result = model.bind_tools("tool1", "tool2", param=True)
        assert result is model


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
