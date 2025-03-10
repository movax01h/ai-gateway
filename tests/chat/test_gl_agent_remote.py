from typing import Optional
from unittest.mock import AsyncMock, Mock, call

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain_core.runnables import Runnable
from pydantic import AnyUrl
from starlette_context import context, request_cycle_context

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.chat import GLAgentRemoteExecutor
from ai_gateway.chat.agents import (
    AgentError,
    AgentFinalAnswer,
    AgentToolAction,
    Message,
    ReActAgentInputs,
)
from ai_gateway.chat.toolset import DuoChatToolsRegistry
from ai_gateway.models.base_chat import Role
from ai_gateway.prompts.typing import (
    AmazonQModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
)


@pytest.fixture
def agent_events():
    return [
        AgentToolAction(thought="thought", tool="issue_reader", tool_input="tool_input")
    ]


@pytest.fixture
def agent(agent_events):
    async def _stream_agent(*_args, **_kwargs):
        for action in agent_events:
            yield action

    agent = Mock(spec=Runnable)
    agent.ainvoke = AsyncMock(side_effect=lambda *_args, **_kwargs: agent_events)
    agent.astream = Mock(side_effect=_stream_agent)

    return agent


@pytest.fixture
def tools_registry():
    return DuoChatToolsRegistry()


@pytest.mark.parametrize(
    ("inputs", "user"),
    [
        (
            ReActAgentInputs(
                messages=[
                    Message(role=Role.USER, content="debug chat_history"),
                    Message(role=Role.ASSISTANT, content="debug chat_history"),
                    Message(role=Role.USER, content="debug question"),
                ],
                agent_scratchpad=[],
            ),
            StarletteUser(
                CloudConnectorUser(
                    authenticated=True,
                    is_debug=True,
                    claims=UserClaims(scopes=["ask_issue"]),
                )
            ),
        ),
        (
            ReActAgentInputs(
                messages=[
                    Message(role=Role.USER, content="chat_history"),
                    Message(role=Role.ASSISTANT, content="chat_history"),
                    Message(role=Role.USER, content="question"),
                ],
                agent_scratchpad=[],
            ),
            StarletteUser(
                CloudConnectorUser(
                    authenticated=True,
                    is_debug=False,
                    claims=UserClaims(scopes=["ask_issue"]),
                )
            ),
        ),
    ],
)
class TestGLAgentRemoteExecutor:
    @pytest.mark.asyncio
    async def test_stream(
        self,
        agent: Mock,
        agent_events,
        tools_registry: DuoChatToolsRegistry,
        internal_event_client: Mock,
        inputs: ReActAgentInputs,
        user: StarletteUser,
    ):
        executor = GLAgentRemoteExecutor(
            agent=agent,
            tools_registry=tools_registry,
            internal_event_client=internal_event_client,
        )

        gl_version = "17.2.0"
        executor.on_behalf(user, gl_version)

        with request_cycle_context({}):
            actual_actions = [action async for action in executor.stream(inputs=inputs)]

            if user.is_debug:
                assert set(context.get("duo_chat.agent_available_tools")) == {
                    "build_reader",
                    "gitlab_documentation",
                    "epic_reader",
                    "issue_reader",
                    "merge_request_reader",
                    "commit_reader",
                }
            else:
                assert context.get("duo_chat.agent_available_tools") == ["issue_reader"]

        agent.astream.assert_called_once_with(inputs)
        assert actual_actions == agent_events


class TestGLAgentRemoteExecutorToolAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "user",
            "gl_version",
            "inputs",
            "model_metadata",
            "agent_events",
            "expected_available_tools",
            "expected_internal_events",
        ),
        [
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["ask_issue"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [AgentToolAction(thought="", tool="issue_reader", tool_input="")],
                ["issue_reader"],
                [call("request_ask_issue", category="ai_gateway.chat.executor")],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["ask_epic"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [AgentToolAction(thought="", tool="epic_reader", tool_input="")],
                ["epic_reader"],
                [call("request_ask_epic", category="ai_gateway.chat.executor")],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["documentation_search"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [
                    AgentToolAction(
                        thought="", tool="gitlab_documentation", tool_input=""
                    )
                ],
                ["gitlab_documentation"],
                [
                    call(
                        "request_documentation_search",
                        category="ai_gateway.chat.executor",
                    )
                ],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["documentation_search"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [
                    AgentToolAction(
                        thought="", tool="GitlabDocumentationTool", tool_input=""
                    )
                ],
                ["gitlab_documentation"],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["documentation_search"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [AgentToolAction(thought="", tool="", tool_input="")],
                ["gitlab_documentation"],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["documentation_search"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [AgentToolAction(thought="", tool="issue_reader", tool_input="")],
                ["gitlab_documentation"],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["documentation_search"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [AgentFinalAnswer(text="I'm good")],
                ["gitlab_documentation"],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=[]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                None,
                [],
                [],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=[]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                AmazonQModelMetadata(
                    role_arn="role-arn", provider="amazon_q", name="amazon_q"
                ),
                [],
                [],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["amazon_q_integration"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                ModelMetadata(
                    provider="litellm",
                    name="mistral",
                    endpoint=AnyUrl("http://localhost:4000"),
                ),
                [],
                [],
                [],
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["amazon_q_integration"]),
                    )
                ),
                "17.2.0",
                ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")]),
                AmazonQModelMetadata(
                    role_arn="role-arn", provider="amazon_q", name="amazon_q"
                ),
                [],
                ["epic_reader", "issue_reader", "gitlab_documentation"],
                [],
            ),
        ],
    )
    async def test_stream_tool_action(
        self,
        agent: Mock,
        tools_registry: DuoChatToolsRegistry,
        internal_event_client: Mock,
        inputs: ReActAgentInputs,
        user: StarletteUser,
        model_metadata: Optional[TypeModelMetadata],
        gl_version: str,
        expected_available_tools,
        expected_internal_events,
    ):
        executor = GLAgentRemoteExecutor(
            agent=agent,
            tools_registry=tools_registry,
            internal_event_client=internal_event_client,
        )

        executor.on_behalf(user, gl_version, model_metadata)

        with request_cycle_context({}):
            async for _ in executor.stream(inputs=inputs):
                pass

            assert (
                context.get("duo_chat.agent_available_tools")
                == expected_available_tools
            )

        internal_event_client.track_event.assert_has_calls(expected_internal_events)


class TestGLAgentRemoteExecutorToolValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("user", "agent_events", "expected_event"),
        [
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["ask_issue"]),
                    )
                ),
                [AgentToolAction(thought="", tool="issue_reader", tool_input="")],
                AgentToolAction(thought="", tool="issue_reader", tool_input=""),
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["ask_epic"]),
                    )
                ),
                [AgentToolAction(thought="", tool="issue_reader", tool_input="")],
                AgentError(message="tool not available", retryable=False),
            ),
            (
                StarletteUser(
                    CloudConnectorUser(
                        authenticated=True,
                        claims=UserClaims(scopes=["ask_issue"]),
                    )
                ),
                [AgentToolAction(thought="", tool="IssueReader", tool_input="")],
                AgentError(message="tool not available", retryable=False),
            ),
        ],
    )
    async def test_stream_tool_validation(
        self,
        agent: Mock,
        tools_registry: DuoChatToolsRegistry,
        internal_event_client: Mock,
        user: StarletteUser,
        expected_event,
    ):
        executor = GLAgentRemoteExecutor(
            agent=agent,
            tools_registry=tools_registry,
            internal_event_client=internal_event_client,
        )

        gl_version = "17.2.0"
        inputs = ReActAgentInputs(messages=[Message(role=Role.USER, content="Hi")])
        executor.on_behalf(user, gl_version)

        with request_cycle_context({}):
            async for event in executor.stream(inputs=inputs):
                assert event == expected_event
