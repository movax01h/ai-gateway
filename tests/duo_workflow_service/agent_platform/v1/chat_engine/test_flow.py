from functools import partial
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from ai_gateway.prompts.config.template import Template
from contract import contract_pb2
from duo_workflow_service.agent_platform.core.inbox import Claimant, TaskId
from duo_workflow_service.agent_platform.v1.chat_engine import ChatFlow
from duo_workflow_service.agent_platform.v1.chat_engine.dispatch import (
    LEGACY_SESSION_MESSAGE,
    EntryRejectedException,
)
from duo_workflow_service.agent_platform.v1.chat_engine.graph_builder import (
    ChatGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.components.base import BaseComponent
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    PartialFlowConfig,
)
from duo_workflow_service.agent_platform.v1.state import FlowEventType, FlowStateKeys
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import MessageTypeEnum, WorkflowStatusEnum
from duo_workflow_service.workflows.abstract_workflow import GraphInvocation
from lib.events import GLReportingEventContext
from tests.duo_workflow_service.agent_platform.v1.chat_engine.template_chat import (
    AGENT,
    RecordingModel,
    chat_graph,
    template_chat_flow,
)

_SESSION = "chat-session-1"
_GRAPH_CONFIG: Any = {"configurable": {"thread_id": _SESSION}}
_FLOW_CHANNELS = {
    "status": None,
    "conversation_history": None,
    "ui_chat_log": None,
    "context": None,
}
_APPROVED = contract_pb2.Approval(approval=contract_pb2.Approval.Approved())


def _config(template: bool = True) -> FlowConfig:
    """A chat-partial version with one agent, on the `template` format or on the old one."""
    partial_config = PartialFlowConfig(
        version="v1",
        environment="chat-partial",
        components=[{"name": "chat_agent", "type": "AgentComponent", "toolset": []}],
        prompts=[
            InMemoryPromptConfig(
                prompt_id="agent_prompt",
                name="agent_prompt",
                unit_primitives=["duo_agent_platform"],
                # Only the `template` format delivers through an inbox.
                **(
                    {"template": Template(system="instructions")}
                    if template
                    else {"prompt_template": {"system": "a", "user": "{{ goal }}"}}
                ),
            )
        ],
    )
    # The identity the registry loader gives a version from this repository: only those may declare `template`.
    partial_config._config_id = "developer"
    return partial_config.to_config()


def _chat_flow(user, config: Optional[FlowConfig] = None, **kwargs: Any) -> ChatFlow:
    if config is None:
        config = _config()
    factory = partial(ChatFlow, config=config)

    flow = factory(
        workflow_id=_SESSION,
        workflow_metadata={
            "git_url": "https://gitlab.com/test/project",
            "git_sha": "abc",
        },
        workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
        user=user,
        **kwargs,
    )
    # What a run fetches before its graph compiles.
    flow._project = None
    flow._namespace = None
    return flow


def _checkpoint(checkpoint_id: str, status: WorkflowStatusEnum) -> CheckpointTuple:
    """A checkpoint as the stop walk reads it: its ID and the status it holds."""
    return CheckpointTuple(
        config={
            "configurable": {"thread_id": _SESSION, "checkpoint_id": checkpoint_id}
        },
        checkpoint=cast(
            Any, {"id": checkpoint_id, "channel_values": {"status": status.value}}
        ),
        metadata={},
    )


def _stand_in_graph(
    *next_nodes: str,
    values: Optional[dict[str, Any]] = None,
    channels: Optional[dict[str, Any]] = None,
    source: Optional[str] = None,
    has_checkpoint: bool = True,
) -> Mock:
    """A compiled graph whose tip reads as given, for the classification wiring alone."""
    graph = Mock()
    graph.aget_state = AsyncMock(
        return_value=SimpleNamespace(next=next_nodes, values=values or {})
    )
    graph.checkpointer.aget_tuple = AsyncMock(
        return_value=(
            SimpleNamespace(
                checkpoint={"channel_values": channels or _FLOW_CHANNELS},
                metadata={"source": source} if source else {},
            )
            if has_checkpoint
            else None
        )
    )
    return graph


@pytest.fixture(name="enter")
def enter_fixture(mock_checkpointer, mock_state_graph):
    """Enter a session through ``AbstractWorkflow.run`` and return what the run handed the graph.

    The graph is a stand-in whose tip reads as the test says, and its components are stand-ins too. A rejected entry
    hands the graph nothing: ``run`` records why in ``last_error`` and returns.
    """
    component = MagicMock(spec=BaseComponent)
    component.__entry_hook__.return_value = "chat_agent#agent"
    component.max_iterations.return_value = 1
    component.graph_steps.side_effect = lambda added_nodes: added_nodes
    tools_registry = Mock()
    tools_registry.toolset.return_value = []
    tools_registry.mcp_tool_names.return_value = []

    async def enter(
        flow: ChatFlow, graph: Mock, status_event: str, goal: str = "hello"
    ) -> Optional[GraphInvocation]:
        mock_checkpointer.initial_status_event = status_event
        # The run and the entry read the same checkpoints.
        mock_checkpointer.aget_tuple = graph.checkpointer.aget_tuple
        compiled = mock_state_graph.compile.return_value
        compiled.aget_state = graph.aget_state
        compiled.checkpointer = mock_checkpointer

        await flow.run(goal)

        if not compiled.astream.called:
            return None
        streamed = compiled.astream.call_args.kwargs
        return GraphInvocation(
            streamed["input"], streamed["config"], streamed.get("context")
        )

    with (
        patch(
            "duo_workflow_service.agent_platform.v1.flows.graph_builder.load_component_class",
            return_value=MagicMock(return_value=component),
        ),
        patch("duo_workflow_service.agent_platform.v1.flows.graph_builder.Router"),
        patch(
            "duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure",
            AsyncMock(return_value=tools_registry),
        ),
    ):
        yield enter


def _rejection(flow: ChatFlow) -> str:
    """Why the entry was rejected."""
    assert isinstance(flow.last_error, EntryRejectedException)
    return str(flow.last_error)


def _delivered(invocation: GraphInvocation) -> list[str]:
    """The text of each message the entry put in the run's inbox."""
    claimed = invocation.context.claim(
        TaskId("the-agent"), Claimant(accepts_unaddressed=True)
    )
    return [message.text for message in claimed]


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
def test_chat_flow_builds_through_the_registry_factory(user):
    flow = _chat_flow(user)

    assert isinstance(flow, Flow)


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
def test_chat_flow_builds_its_graph_with_the_chat_graph_builder(user):
    flow = _chat_flow(user)

    builder = flow._graph_builder(Mock(spec=ToolsRegistry))

    assert isinstance(builder, ChatGraphBuilder)
    assert builder._workflow_id == _SESSION


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
@pytest.mark.parametrize(
    "workflow_config,expected",
    [
        ({"allow_agent_to_request_user": True}, True),
        ({"allow_agent_to_request_user": False}, False),
        ({}, False),
    ],
)
def test_chat_flow_wires_allow_agent_to_request_user_into_the_builder(
    user, workflow_config, expected, monkeypatch
):
    """Guard the ``_graph_builder`` override against a dropped ``allow_agent_to_request_user``.

    Dropping the kwarg defaults ``ChatGraphBuilder`` to ``False``, which flips every chat
    approval to a halt with no one to answer it, so the wiring needs its own assertion
    here rather than relying on the base ``Flow`` seam test.
    """
    flow = _chat_flow(user)
    monkeypatch.setattr(flow, "_workflow_config", workflow_config)

    with patch(
        "duo_workflow_service.agent_platform.v1.chat_engine.flow.ChatGraphBuilder"
    ) as mock_builder_class:
        flow._graph_builder(Mock(spec=ToolsRegistry))

    assert (
        mock_builder_class.call_args.kwargs["allow_agent_to_request_user"] is expected
    )


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestEntryClassification:
    """``ChatFlow`` keeps what belongs to the boundary and hands ``Flow`` everything that waits."""

    @pytest.mark.asyncio
    async def test_the_first_message_starts_the_session_with_its_goal(
        self, user, enter
    ):
        graph = _stand_in_graph(has_checkpoint=False)

        invocation = await enter(
            _chat_flow(user), graph, WorkflowStatusEventEnum.START, goal="hello"
        )
        state = invocation.graph_input

        assert _delivered(invocation) == ["hello"]
        assert state["conversation_history"] == {}
        (entry,) = state["ui_chat_log"]
        assert entry["message_type"] == MessageTypeEnum.USER
        assert entry["content"] == "hello"
        assert state["context"]["goal"] == "hello"
        graph.aget_state.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "values,starts",
        [
            # What LangGraph reads from a session's first input checkpoint: every reducer channel at its default.
            (
                {
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {},
                    "agent_context_limits": {},
                },
                True,
            ),
            (
                {
                    "conversation_history": {
                        "chat_agent": [AIMessage("hello"), AIMessage("answer")]
                    }
                },
                False,
            ),
        ],
        ids=[
            "the_first_turns_starts_the_session",
            "a_later_turns_keeps_the_earlier_ones",
        ],
    )
    async def test_a_message_pinned_to_an_input_checkpoint_forks_from_it(
        self, user, enter, values, starts
    ):
        """``AbstractWorkflow`` restarts a pinned input checkpoint as START, so its input applies once."""
        graph = _stand_in_graph("__start__", source="input", values=values)

        invocation = await enter(
            _chat_flow(user, resume_checkpoint_ts="cp-input"),
            graph,
            WorkflowStatusEventEnum.RESUME,
            goal="edited",
        )
        graph_input = invocation.graph_input

        assert invocation.graph_config["configurable"]["checkpoint_id"] == "cp-input"
        assert _delivered(invocation) == ["edited"]
        # Only a session that starts sets the goal, which stays the first message's,
        # and only a later turn leaves the history the checkpoint holds as it is.
        assert ("goal" in graph_input["context"]) is starts
        assert ("conversation_history" in graph_input) is starts

    @pytest.mark.asyncio
    async def test_a_session_whose_checkpoints_are_gone_starts_over(self, user, enter):
        """Rails still says the session is waiting, but it has no checkpoint left to continue from."""
        graph = _stand_in_graph(has_checkpoint=False)

        invocation = await enter(
            _chat_flow(user), graph, WorkflowStatusEventEnum.RESUME, goal="hello"
        )
        state = invocation.graph_input

        assert _delivered(invocation) == ["hello"]
        assert state["context"]["goal"] == "hello"

    @pytest.mark.asyncio
    async def test_a_message_at_the_boundary_enters_as_plain_input(self, user, enter):
        earlier = AIMessage(content="earlier answer")
        graph = _stand_in_graph(
            values={
                "conversation_history": {"chat_agent": [earlier]},
            },
        )

        invocation = await enter(
            _chat_flow(user), graph, WorkflowStatusEventEnum.RESUME, goal="next"
        )
        graph_input = invocation.graph_input

        assert graph_input["status"] == WorkflowStatusEnum.EXECUTION.value
        assert _delivered(invocation) == ["next"]
        # The run continues from the tip: nothing pins it elsewhere.
        assert "checkpoint_id" not in invocation.graph_config["configurable"]
        # The agent appends what it renders, so the entry leaves its history alone.
        assert "conversation_history" not in graph_input
        (entry,) = graph_input["ui_chat_log"]
        assert entry["message_type"] == MessageTypeEnum.USER
        assert entry["content"] == "next"
        # The agent starts its own cycle budget again when it claims the message.
        assert graph_input["context"] == {"inputs": {}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "goal,approval,expected",
        [
            ("", _APPROVED, FlowEventType.APPROVE),
            ("use a branch", None, FlowEventType.RESPONSE),
        ],
        ids=[
            "a_decision_goes_to_whatever_waits",
            "a_message_goes_to_whatever_waits",
        ],
    )
    async def test_flow_resumes_what_waits_inside_a_turn(
        self, user, enter, goal, approval, expected
    ):
        graph = _stand_in_graph("chat_agent#tool_approval_fetch")

        invocation = await enter(
            _chat_flow(user, approval=approval),
            graph,
            WorkflowStatusEventEnum.RESUME,
            goal=goal,
        )

        assert isinstance(invocation.graph_input, Command)
        assert invocation.graph_input.resume["event_type"] == expected
        # What answers a wait is the resume value, so nothing waits in the inbox for another task.
        assert _delivered(invocation) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_event",
        [WorkflowStatusEventEnum.RESUME, WorkflowStatusEventEnum.RETRY],
        ids=["while_waiting", "while_running"],
    )
    async def test_a_reconnect_in_a_turn_replays_through_flows_retry(
        self, user, enter, workflow_config, status_event
    ):
        """Resuming with no answer would fail the approval; ``Flow``'s retry path replays the tip instead."""
        # Rails reports the session's checkpoints, which is what tells a retry to replay and not to start over.
        workflow_config["first_checkpoint"] = {"checkpoint": "{}"}
        graph = _stand_in_graph("chat_agent#tool_approval_fetch")

        invocation = await enter(_chat_flow(user), graph, status_event, goal="")

        assert invocation.graph_input is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "approval,goal",
        [(None, "next"), (_APPROVED, "")],
        ids=["a_message", "a_decision"],
    )
    async def test_a_running_turn_asks_for_a_retry_first(
        self, user, enter, approval, goal
    ):
        """A message isn't dropped, and a decision has nothing paused to answer."""
        graph = _stand_in_graph("chat_agent#tool_approval_fetch")
        flow = _chat_flow(user, approval=approval)

        assert await enter(flow, graph, WorkflowStatusEventEnum.RETRY, goal) is None
        assert "Retry it before sending a new message" in _rejection(flow)

    @pytest.mark.asyncio
    async def test_a_pinned_checkpoint_takes_a_message_as_plain_input(
        self, user, enter
    ):
        graph = _stand_in_graph("chat_agent#tool_approval_fetch")

        invocation = await enter(
            _chat_flow(user, resume_checkpoint_ts="cp-3"),
            graph,
            WorkflowStatusEventEnum.RESUME,
        )

        # A plain input on a pinned checkpoint is what makes LangGraph fork from it.
        assert invocation.graph_config["configurable"]["checkpoint_id"] == "cp-3"
        assert not isinstance(invocation.graph_input, Command)
        assert invocation.graph_input["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flow_kwargs,graph,goal,reason",
        [
            ({"approval": _APPROVED}, _stand_in_graph(), "", "decision is stale"),
            ({}, _stand_in_graph(), "", "waiting for your next message"),
            (
                {},
                _stand_in_graph(channels={**_FLOW_CHANNELS, "plan": None}),
                "hello",
                LEGACY_SESSION_MESSAGE,
            ),
            (
                {"approval": _APPROVED, "resume_checkpoint_ts": "cp-3"},
                _stand_in_graph("chat_agent#tool_approval_fetch"),
                "",
                "takes a new message",
            ),
        ],
        ids=[
            "a_decision_at_the_boundary",
            "a_reconnect_at_the_boundary",
            "a_legacy_session",
            "a_decision_on_a_pin",
        ],
    )
    async def test_a_rejected_entry_hands_the_graph_nothing(
        self, user, enter, flow_kwargs, graph, goal, reason
    ):
        flow = _chat_flow(user, **flow_kwargs)

        assert await enter(flow, graph, WorkflowStatusEventEnum.RESUME, goal) is None
        assert reason in _rejection(flow)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flow_kwargs,graph,reason",
        [
            (
                {"approval": _APPROVED},
                _stand_in_graph(has_checkpoint=False),
                "no checkpoints",
            ),
            ({}, _stand_in_graph(has_checkpoint=False), "no checkpoints"),
            (
                {"approval": _APPROVED, "resume_checkpoint_ts": "cp-1"},
                _stand_in_graph(source="input"),
                "takes a new message",
            ),
        ],
        ids=[
            "a_decision_on_a_new_session",
            "a_reconnect_on_a_new_session",
            "a_decision_on_a_pinned_input_checkpoint",
        ],
    )
    async def test_a_start_without_a_message_is_rejected(
        self, user, enter, flow_kwargs, graph, reason
    ):
        """Only a message starts a session: anything else that arrives as START goes through the table."""
        flow = _chat_flow(user, **flow_kwargs)

        assert await enter(flow, graph, WorkflowStatusEventEnum.START, "") is None
        assert reason in _rejection(flow)

    @pytest.mark.asyncio
    async def test_a_message_after_a_stop_is_a_turn_from_where_the_walk_landed(
        self, user, enter, mock_checkpointer
    ):
        """The walk lands on the stopped turn's input checkpoint, which holds the boundary its run started from."""

        async def newest_first(**_kwargs):
            for checkpoint_id, status in (
                ("cp-2", WorkflowStatusEnum.EXECUTION),
                ("cp-1", WorkflowStatusEnum.IDLE),
                ("cp-0", WorkflowStatusEnum.NOT_STARTED),
            ):
                yield _checkpoint(checkpoint_id, status)

        mock_checkpointer.checkpoints_reversed = newest_first
        graph = _stand_in_graph(
            "__start__",
            source="input",
            values={"conversation_history": {"chat_agent": [AIMessage("answer")]}},
        )
        stopped = [
            {"message_type": MessageTypeEnum.USER, "content": "the stopped message"}
        ]

        with patch(
            "duo_workflow_service.agent_platform.v1.flows.base.cancelled_turn_context",
            return_value=stopped,
        ):
            invocation = await enter(
                _chat_flow(user),
                graph,
                WorkflowStatusEventEnum.STOP_RECOVERY,
                goal="next",
            )

        assert invocation.graph_config["configurable"]["checkpoint_id"] == "cp-1"
        assert _delivered(invocation) == ["next"]
        assert "conversation_history" not in invocation.graph_input
        # Nothing on the `template` format reads the cancelled turn, so it isn't stored where nothing would clear it.
        assert "cancelled_turn" not in invocation.graph_input["context"]["inputs"]

    @pytest.mark.asyncio
    async def test_a_message_is_refused_where_no_inbox_would_hold_it(self, user, enter):
        """A version on the old format has no inbox, so reaching ``ChatFlow`` with one is a routing mistake."""
        flow = _chat_flow(user, _config(template=False))

        invocation = await enter(
            flow, _stand_in_graph(has_checkpoint=False), WorkflowStatusEventEnum.START
        )

        assert invocation is None
        assert isinstance(flow.last_error, ValueError)
        assert "no inbox" in str(flow.last_error)


async def _turn(
    flow: ChatFlow, compiled: Any, status_event: str, goal: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enter the session with a message, run the turn, and return the graph input and the state it ended in."""
    invocation = await flow._resolve_graph_invocation(
        compiled_graph=compiled,
        graph_config=_GRAPH_CONFIG,
        goal=goal,
        status_event=status_event,
        checkpoint_tuple={"threadTs": "tip"},
    )
    # The run hands the graph its inbox as runtime context, as `AbstractWorkflow` does.
    state = await compiled.ainvoke(
        invocation.graph_input, invocation.graph_config, context=invocation.context
    )
    return invocation.graph_input, state


def _history(state: dict[str, Any]) -> list[str]:
    return [
        message.text for message in state[FlowStateKeys.CONVERSATION_HISTORY][AGENT]
    ]


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestAMessageIsDeliveredOnce:
    """``ChatFlow`` delivers a message and the agent renders it: only the model is faked."""

    @pytest.fixture(name="model")
    def model_fixture(self):
        return RecordingModel(
            responses=[AIMessage(content="answer"), AIMessage(content="next answer")]
        )

    @pytest.fixture(name="flow")
    def flow_fixture(self, model):
        return template_chat_flow(model, _SESSION)

    @pytest.fixture(name="compiled")
    def compiled_fixture(self, flow):
        return chat_graph(flow).compile(checkpointer=MemorySaver())

    @pytest.mark.asyncio
    async def test_the_first_message_reaches_the_model_once(
        self, flow, compiled, model
    ):
        _, state = await _turn(flow, compiled, WorkflowStatusEventEnum.START, "hello")

        assert model.conversation() == ["hello"]
        assert _history(state) == ["hello", "answer"]
        assert state[FlowStateKeys.STATUS] == WorkflowStatusEnum.IDLE
        assert [
            entry["content"]
            for entry in state[FlowStateKeys.UI_CHAT_LOG]
            if entry["message_type"] == MessageTypeEnum.USER
        ] == ["hello"]

    @pytest.mark.asyncio
    async def test_a_follow_up_on_an_idle_session_is_delivered_after_the_first_turn(
        self, flow, compiled, model
    ):
        await _turn(flow, compiled, WorkflowStatusEventEnum.START, "hello")

        _, state = await _turn(flow, compiled, WorkflowStatusEventEnum.RESUME, "next")

        assert model.conversation() == ["hello", "answer", "next"]
        assert _history(state) == ["hello", "answer", "next", "next answer"]
        assert state[FlowStateKeys.STATUS] == WorkflowStatusEnum.IDLE

    @pytest.mark.asyncio
    async def test_a_blank_first_message_is_rejected_instead_of_starting_the_session(
        self, flow, compiled, model
    ):
        with pytest.raises(EntryRejectedException, match="isn't blank"):
            await _turn(flow, compiled, WorkflowStatusEventEnum.START, "   ")

        assert model.calls == []

    @pytest.mark.asyncio
    async def test_a_blank_follow_up_is_rejected_instead_of_starting_a_turn(
        self, flow, compiled, model
    ):
        await _turn(flow, compiled, WorkflowStatusEventEnum.START, "hello")

        with pytest.raises(EntryRejectedException, match="isn't blank"):
            await _turn(flow, compiled, WorkflowStatusEventEnum.RESUME, "   ")

        assert len(model.calls) == 1
