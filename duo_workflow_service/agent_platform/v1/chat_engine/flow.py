from datetime import datetime, timezone
from typing import Any, Callable, Final, Optional, override
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.types import StateSnapshot

from duo_workflow_service.agent_platform.core.inbox.message import (
    InboundMessage,
    InvalidInput,
    User,
    inbound_message,
)
from duo_workflow_service.agent_platform.v1.chat_engine.dispatch import (
    EntryEvent,
    EntryEventKind,
    EntryMode,
    EntryRejectedException,
    EntryTip,
    classify_entry,
)
from duo_workflow_service.agent_platform.v1.chat_engine.graph_builder import (
    ChatGraphBuilder,
)
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import FlowGraphBuilder
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    CheckpointSourceEnum,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.workflows.abstract_workflow import GraphInvocation

__all__ = ["ChatFlow"]

# The path of ``Flow`` that each delegated mode takes. ``ChatFlow`` names it
# rather than passing on the status event Rails derived: a reconnect while an
# approval waits arrives as RESUME, and resuming it with no answer would fail.
_FLOW_PATH: Final = {
    EntryMode.RESUME: WorkflowStatusEventEnum.RESUME,
    EntryMode.RETRY: WorkflowStatusEventEnum.RETRY,
}


class ChatFlow(Flow):
    """The chat engine: a boundary policy over the shared executor for chat-partial flows.

    ``Flow`` and ``FlowGraphBuilder`` run the turn, and every component is the
    shared class ambient flows use. ``ChatFlow`` owns the conversational
    boundary and nothing else: where a turn ends (the graph builder), where it
    begins (entry classification once the graph is compiled), and how a user
    message is delivered when it crosses. What waits inside a turn, and rolling
    a stopped turn back, are ``Flow``'s.
    The registry routes a chat-partial version here when its prompts declare
    the ``template`` format, and fills the flow-level fields the builder needs;
    the chat-surface defaults are ``AgentComponent``'s own under the
    ``chat-partial`` environment.

    See ``docs/flow_registry/chat_engine.md``.
    """

    def __init__(self, *, config: FlowConfig, **kwargs: Any):
        # Keyword-only: the registry binds ``config`` through ``functools.partial``
        # and the server passes every other argument by keyword.
        super().__init__(config=config, **kwargs)
        self._rolled_back = False

    @override
    def _graph_builder(self, tools_registry: ToolsRegistry) -> FlowGraphBuilder:
        return ChatGraphBuilder(
            tools_registry=tools_registry,
            prompt_registry=self._flow_prompt_registry,
            schema_registry=self._flow_schema_registry,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            user=self._user,
            internal_event_client=self._internal_event_client,
            catalog_items=self._catalog_items,
            allow_agent_to_request_user=self._workflow_config.get(
                "allow_agent_to_request_user", False
            ),
            web_search_blocked_domains=self._workflow_config.get(
                "web_search_denied_domains", []
            ),
        )

    @override
    def _resolve_resume_checkpoint_ts(self) -> Optional[str]:
        # A pinned checkpoint takes a message only, as plain graph input, so
        # LangGraph forks from it instead of skipping time travel.
        return self._resume_checkpoint_ts

    @override
    async def _resolve_stop_recovery(
        self, checkpointer: BaseCheckpointSaver
    ) -> tuple[Optional[CheckpointTuple], WorkflowStatusEventEnum]:
        """Roll a stopped turn back with ``Flow``'s walk, and remember that the entry continues where it landed."""
        landing, status_event = await super()._resolve_stop_recovery(checkpointer)
        self._rolled_back = landing is not None
        return landing, status_event

    @override
    async def _resolve_graph_invocation(
        self,
        *,
        compiled_graph: Any,
        graph_config: RunnableConfig,
        goal: str,
        status_event: str,
        checkpoint_tuple: Any,
    ) -> GraphInvocation:
        """Classify the entry, keep what belongs to the boundary, and hand ``Flow`` everything that waits in a turn.

        A START status event that carries a message is a first message: a session with no checkpoints, or a first turn
        the stop walk starts again. The parent also restarts a pinned input checkpoint as START so its input applies
        once. That is a first message only while the checkpoint holds no earlier turn: a later turn's input checkpoint
        takes the message as a pinned TURN. Anything else that arrives as START goes through the table, which rejects
        it.
        """
        event = EntryEvent.from_request(
            has_user_content=bool(goal),
            decision=self._decision(),
            pinned=self._resume_checkpoint_ts is not None,
        )
        first_message = (
            status_event == WorkflowStatusEventEnum.START
            and event.kind is EntryEventKind.MESSAGE
        )
        if first_message and not event.pinned:
            return self._enter_with(goal, self._first_message_state, graph_config)

        snapshot = await compiled_graph.aget_state(graph_config)
        if first_message and _holds_no_turn(snapshot):
            return self._enter_with(goal, self._first_message_state, graph_config)

        checkpoint = await compiled_graph.checkpointer.aget_tuple(graph_config)
        dispatch = classify_entry(
            _entry_tip(snapshot, checkpoint),
            event,
            waiting_on_user=status_event == WorkflowStatusEventEnum.RESUME,
            rolled_back=self._rolled_back,
        )

        match dispatch.mode:
            case EntryMode.REJECT:
                raise EntryRejectedException(dispatch.reason)
            case EntryMode.RESUME | EntryMode.RETRY:
                return await super()._resolve_graph_invocation(
                    compiled_graph=compiled_graph,
                    graph_config=graph_config,
                    goal=goal,
                    status_event=_FLOW_PATH[dispatch.mode],
                    checkpoint_tuple=checkpoint_tuple,
                )
            case EntryMode.START:
                return self._enter_with(goal, self._first_message_state, graph_config)
            case EntryMode.TURN:
                return self._enter_with(goal, self._message_input, graph_config)

    def _enter_with(
        self,
        goal: str,
        graph_input_for: Callable[[str], dict[str, Any]],
        graph_config: RunnableConfig,
    ) -> GraphInvocation:
        """Enter the graph with a message, delivered through the run's inbox like any other input.

        The message is made first, so one it refuses is rejected before anything else reads it.
        """
        message = self._message(goal)
        return self._with_inbox(
            GraphInvocation(graph_input_for(goal), graph_config), [message]
        )

    def _decision(self) -> Optional[str]:
        if self._approval is None:
            return None
        return self._approval.WhichOneof("user_decision")

    def _message(self, goal: str) -> InboundMessage:
        """The message a request brought, which nothing waits for, the first included.

        The engine delivers and never renders. The message goes into the run's inbox, and the agent renders it once
        through its prompt's ``message`` block, into its own history.

        Raises:
            EntryRejectedException: When the message can't be delivered, such as a blank one.
        """
        message = inbound_message(
            id=f"turn-{uuid4()!s}",
            content=[{"type": "text", "text": goal}],
            sender=User(),
        )
        if isinstance(message, InvalidInput):
            raise EntryRejectedException(message.reason)

        return message

    def _first_message_state(self, goal: str) -> dict[str, Any]:
        # ``Flow``'s initial state, with the message's USER transcript entry in place of its "Starting Flow" entry.
        return {
            **self.get_workflow_state(goal),
            FlowStateKeys.UI_CHAT_LOG: [_user_entry(goal)],
        }

    def _message_input(self, goal: str) -> dict[str, Any]:
        """A message entering at the boundary as plain graph input, the boundary a rollback landed on included.

        The envelope inputs sent with the message refresh ``context.inputs``. What a rollback discarded isn't among
        them: nothing on the ``template`` format reads the ``cancelled_turn`` envelope, so it would stay in state with
        nothing to clear it. The agent that claims the message starts its own cycle budget again.
        """
        return {
            FlowStateKeys.STATUS: WorkflowStatusEnum.EXECUTION.value,
            FlowStateKeys.CONTEXT: {
                "inputs": self._process_additional_context(
                    self._additional_context or []
                ),
            },
            FlowStateKeys.UI_CHAT_LOG: [_user_entry(goal)],
        }


def _holds_no_turn(snapshot: StateSnapshot) -> bool:
    # A checkpoint from before the session's first node still reads every reducer channel at its empty default.
    return not any(snapshot.values.values())


def _entry_tip(
    snapshot: StateSnapshot, checkpoint: Optional[CheckpointTuple]
) -> Optional[EntryTip]:
    if checkpoint is None:
        return None
    return EntryTip(
        next_nodes=snapshot.next,
        channels=frozenset(checkpoint.checkpoint.get("channel_values", {})),
        pre_run_input=(checkpoint.metadata or {}).get("source")
        == CheckpointSourceEnum.INPUT,
    )


def _user_entry(goal: str) -> UiChatLog:
    return UiChatLog(
        message_type=MessageTypeEnum.USER,
        content=goal,
        message_id=f"user-{uuid4()!s}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=ToolStatus.SUCCESS,
        correlation_id=None,
        tool_info=None,
        additional_context=None,
        message_sub_type=None,
    )
