import fastapi
import pytest
from langchain_core.messages import SystemMessage
from starlette_context import context, request_cycle_context
from structlog.testing import capture_logs

from ai_gateway.chat.agents.react import (
    AgentError,
    AgentFinalAnswer,
    AgentToolAction,
    AgentUnknownAction,
    ReActAgent,
    ReActAgentInputs,
    ReActPlainTextParser,
)
from ai_gateway.chat.agents.typing import (
    AdditionalContext,
    AgentStep,
    CurrentFile,
    Message,
)
from ai_gateway.chat.context.current_page import Context, IssueContext
from ai_gateway.chat.tools.gitlab import IssueReader, MergeRequestReader
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.models.base_chat import Role
from ai_gateway.prompts.config.models import (
    ChatAnthropicParams,
    ChatLiteLLMParams,
    ModelClassProvider,
    TypeModelParams,
)


@pytest.fixture
def prompt_class():
    return ReActAgent


@pytest.fixture
def inputs():
    return ReActAgentInputs(
        messages=[
            Message(role=Role.USER, content="Hi, how are you?"),
            Message(role=Role.ASSISTANT, content="I'm good!"),
        ]
    )


@pytest.fixture
def prompt_template():
    return {
        "system": "{% include 'chat/react/system/1.0.0.jinja' %}",
        "user": "{% include 'chat/react/user/1.0.0.jinja' %}",
        "assistant": "{% include 'chat/react/assistant/1.0.0.jinja' %}",
    }


@pytest.fixture(autouse=True)
def stub_feature_flags():
    current_feature_flag_context.set(["expanded_ai_logging"])
    yield


class TestReActPlainTextParser:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "thought1\nAction: tool1\nAction Input: tool_input1\n",
                AgentToolAction(
                    thought="thought1",
                    tool="tool1",
                    tool_input="tool_input1",
                ),
            ),
            (
                "thought1\nFinal Answer: final answer\n",
                AgentFinalAnswer(
                    text="final answer",
                ),
            ),
            (
                "Hi, I'm GitLab Duo Chat.",
                AgentUnknownAction(
                    text="Hi, I'm GitLab Duo Chat.",
                ),
            ),
        ],
    )
    def test_agent_message(self, text: str, expected: AgentToolAction):
        parser = ReActPlainTextParser()
        actual = parser.parse(text)

        assert actual == expected


class TestReActAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "inputs",
            "model_response",
            "expected_actions",
        ),
        [
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                "Thought: I'm thinking...\nAction: issue_reader\nAction Input: random input",
                [
                    AgentToolAction(
                        thought="I'm thinking...",
                        tool="issue_reader",
                        tool_input="random input",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Summarize this Merge request",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[MergeRequestReader()],
                ),
                "Thought: I'm thinking...\nAction: MergeRequestReader\nAction Input: random input",
                [
                    AgentToolAction(
                        thought="I'm thinking...",
                        tool="merge_request_reader",
                        tool_input="random input",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(role=Role.USER, content="How can I log output?"),
                        Message(
                            role=Role.ASSISTANT,
                            content="Use print function",
                            agent_scratchpad=[
                                AgentStep(
                                    action=AgentToolAction(
                                        thought="thought",
                                        tool="tool",
                                        tool_input="tool_input",
                                    ),
                                    observation="observation",
                                )
                            ],
                        ),
                        Message(
                            role=Role.USER,
                            content="Can you explain the print function?",
                        ),
                    ],
                    agent_scratchpad=[],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(text="A"),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="what's the description of this issue",
                        ),
                        Message(
                            role=Role.ASSISTANT,
                            content="PoC ReAct",
                            agent_scratchpad=[
                                AgentStep(
                                    action=AgentToolAction(
                                        thought="thought",
                                        tool="tool",
                                        tool_input="tool_input",
                                    ),
                                    observation="observation",
                                )
                            ],
                        ),
                        Message(role=Role.USER, content="What's your name?"),
                    ],
                    agent_scratchpad=[
                        AgentStep(
                            action=AgentToolAction(
                                thought="thought",
                                tool="ci_issue_reader",
                                tool_input="random input",
                            ),
                            observation="observation",
                        )
                    ],
                ),
                "Thought: I'm thinking...\nFinal Answer: Bar",
                [
                    AgentFinalAnswer(
                        text="B",
                    ),
                    AgentFinalAnswer(text="a"),
                    AgentFinalAnswer(text="r"),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Explain this issue",
                            context=Context(
                                type="issue", content="this issue is about Duo Chat"
                            ),
                        ),
                    ],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Explain this issue",
                            context=IssueContext(type="issue", title="Duo Chat issue"),
                        ),
                    ],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Explain this code",
                            current_file=CurrentFile(
                                file_path="main.py",
                                data="print",
                                selected_code=True,
                            ),
                        ),
                    ],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Explain this code",
                            additional_context=[
                                AdditionalContext(
                                    id="id",
                                    category="file",
                                    content="print",
                                    metadata={"a": "b"},
                                )
                            ],
                        ),
                    ],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="Hi, how are you? Do not include Final Answer:, Thought: and Action: in response.",
                        ),
                    ],
                    agent_scratchpad=[],
                ),
                "I'm good. How about you?",
                [
                    AgentUnknownAction(
                        text="I'm good. How about you?",
                    ),
                ],
            ),
        ],
    )
    @pytest.mark.xdist_group("capture_logs")
    async def test_stream(
        self,
        inputs: ReActAgentInputs,
        model_response: str,
        expected_actions: list[AgentToolAction | AgentFinalAnswer | AgentUnknownAction],
        prompt: ReActAgent,
    ):
        with capture_logs() as cap_logs, request_cycle_context({}):
            actual_actions = [action async for action in prompt.astream(inputs)]

            if isinstance(expected_actions[0], AgentToolAction):
                assert (
                    context.get("duo_chat.agent_tool_action")
                    == expected_actions[0].tool
                )

        assert actual_actions == expected_actions
        response_streaming_events = list(
            filter(lambda entry: entry["event"] == "Response streaming", cap_logs)
        )
        assert len(response_streaming_events) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "feature_flag_enabled",
            "inputs",
            "model_params",
            "should_add_anthropic_cache",
        ),
        [
            (
                True,
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                ChatAnthropicParams(model_class_provider=ModelClassProvider.ANTHROPIC),
                True,
            ),
            (
                False,
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                ChatAnthropicParams(model_class_provider=ModelClassProvider.ANTHROPIC),
                False,
            ),
            (
                False,
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                ChatLiteLLMParams(model_class_provider=ModelClassProvider.LITE_LLM),
                False,
            ),
            (
                True,
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                ChatLiteLLMParams(model_class_provider=ModelClassProvider.LITE_LLM),
                False,
            ),
            (
                True,
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this issue?",
                        ),
                    ],
                    agent_scratchpad=[],
                    tools=[IssueReader()],
                ),
                ChatLiteLLMParams(model_class_provider=ModelClassProvider.LITE_LLM),
                False,
            ),
        ],
    )
    async def test_message_cache_control(
        self,
        prompt: ReActAgent,
        feature_flag_enabled: bool,
        inputs: ReActAgentInputs,
        model_params: TypeModelParams,
        should_add_anthropic_cache: bool,
    ):
        if feature_flag_enabled:
            current_feature_flag_context.set({"enable_anthropic_prompt_caching"})
        else:
            current_feature_flag_context.set(set[str]())

        prompt_value = prompt.prompt_tpl.invoke(inputs)

        for msg in prompt_value.to_messages():
            if isinstance(msg, SystemMessage):
                if should_add_anthropic_cache:
                    if isinstance(msg.content[0], dict):
                        content_dict = msg.content[0]
                        assert content_dict["type"] == "text"
                        assert content_dict["cache_control"] == {"type": "ephemeral"}
                    else:
                        raise TypeError(
                            f"Expected msg.content[0] to be a dict, but got {type(msg.content[0])}"
                        )
                else:
                    assert isinstance(msg.content, str)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "inputs",
            "model_error",
            "error_message",
            "expected_events",
        ),
        [
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this epic?",
                        ),
                    ],
                ),
                ValueError("overloaded_error"),
                "overloaded_error",
                [
                    AgentError(message="overloaded_error", retryable=True),
                ],
            ),
            (
                ReActAgentInputs(
                    messages=[
                        Message(
                            role=Role.USER,
                            content="What's the title of this epic?",
                        ),
                    ],
                ),
                ValueError("api_error"),
                "api_error",
                [
                    AgentError(message="api_error", retryable=False),
                ],
            ),
        ],
    )
    async def test_stream_error(
        self,
        inputs: ReActAgentInputs,
        model_error: Exception,
        error_message: str,
        expected_events: list[AgentError],
        prompt: ReActAgent,
    ):
        actual_events = []
        with pytest.raises(ValueError) as exc_info:
            async for event in prompt.astream(inputs):
                actual_events.append(event)

        assert actual_events == expected_events
        assert str(exc_info.value) == error_message

    @pytest.mark.asyncio
    async def test_message_agent_scratchpad_validation(self):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            Message(
                role=Role.USER,
                content="test",
                agent_scratchpad=[AgentStep(action=None, observation="test")],
            )

        assert exc_info.value.status_code == 400
        assert (
            exc_info.value.detail
            == "agent_scratchpad can only be present when role is ASSISTANT"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("inputs", "model_response", "expected_actions"),
        [
            (
                ReActAgentInputs(
                    messages=[
                        Message(role=Role.USER, content="What's the weather like?"),
                        Message(role=Role.ASSISTANT, content=None),
                    ],
                    agent_scratchpad=[],
                    tools=[],
                ),
                "Thought: I'm thinking...\nFinal Answer: A",
                [
                    AgentFinalAnswer(
                        text="A",
                    ),
                ],
            ),
        ],
    )
    @pytest.mark.xdist_group("capture_logs")
    async def test_stream_with_empty_assistant_content(
        self,
        inputs: ReActAgentInputs,
        model_response: str,
        expected_actions: list[AgentFinalAnswer],
        prompt: ReActAgent,
    ):
        with capture_logs() as cap_logs, request_cycle_context({}):
            actual_actions = [action async for action in prompt.astream(inputs)]

        prompt_value = prompt.prompt_tpl.invoke(inputs)

        assert actual_actions == expected_actions
        response_streaming_events = list(
            filter(lambda entry: entry["event"] == "Response streaming", cap_logs)
        )
        assert len(response_streaming_events) > 0

        messages = prompt_value.to_messages()
        assert len(messages) == 3  # System, User and one added AI message
