from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
)
from duo_workflow_service.agent_platform.v1.state import (
    BaseIOKey,
    FlowEvent,
    FlowEventType,
    FlowState,
    IOKey,
    NoneIOKey,
    RuntimeIOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.errors.typing import InvalidRequestException

__all__ = ["FetchNode"]

# Meta tags wrapping the cancelled-turn transcript prepended to the first
# user message after a stop-recovery rollback (see Flow._resolve_stop_recovery).
_CANCELLED_TURN_OPEN_TAG = "<cancelled-turn>"
_CANCELLED_TURN_CLOSE_TAG = "</cancelled-turn>"
_CANCELLED_TURN_PREAMBLE = (
    "The user stopped the previous turn before it completed. "
    "The following exchange was discarded by the rollback and is provided "
    "for reference only:"
)


class FetchNode:
    """Node that fetches user input via interrupt() and creates HumanMessage.

    This node interrupts workflow execution to wait for user input, then processes
    the response based on the event type received.

    Supported event types:
        - APPROVE: User approves, stores approval value in output location.
                    No conversation history update.
        - REJECT: User rejects, stores rejection value in output location.
        - MODIFY: User requests modifications, adds message to conversation history.
                    Sets status to EXECUTION to continue workflow.
        - RESPONSE: User provides input/response, adds message to conversation history.
                    Sets status to EXECUTION to continue workflow.

    When a stop-recovery rollback discarded a cancelled turn (see
    ``Flow._resolve_stop_recovery``), the transcript of the cancelled exchange is
    available at the location referenced by ``cancelled_turn_key``. The node
    prepends it to the model-facing ``HumanMessage`` between ``<cancelled-turn>``
    meta tags — giving follow-ups like "actually create it in python" their
    antecedent — and then clears the state location so the context is consumed
    exactly once. The UI chat log keeps showing the clean user message.

    Args:
        name: The name of this node
        component_name: The name of the parent component
        output: IOKey for storing approval/rejection values
        conversation_history_key: IOKey for accessing conversation history (includes target agent)
        ui_history: UI logging history for tracking events
        status_key: IOKey for setting workflow status
        cancelled_turn_key: IOKey referencing the cancelled-turn ``ui_chat_log``
            delta. Defaults to a ``NoneIOKey`` (no cancelled-turn handling) so
            existing callers are unaffected.
    """

    def __init__(
        self,
        *,
        name: str,
        component_name: str,
        output: IOKey,
        conversation_history_key: IOKey,
        ui_history: UIHistory,
        status_key: IOKey,
        cancelled_turn_key: BaseIOKey = NoneIOKey(alias="cancelled_turn"),
    ):
        self.name = name
        self._component_name = component_name
        self._output = output
        self._conversation_history_key = conversation_history_key
        self._ui_history = ui_history
        self._status_key = status_key
        self._cancelled_turn_key = cancelled_turn_key

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute the fetch node - interrupt for user input and create HumanMessage."""
        # Interrupt workflow to wait for user input
        event: FlowEvent = interrupt("Workflow interrupted; waiting for user input.")

        existing_history = self._conversation_history_key.value_from_state(state) or []
        # Handle different event types
        if event["event_type"] in (FlowEventType.APPROVE, FlowEventType.REJECT):
            # Handle approval/rejection events
            # Store the user decision in the specified output location
            approval_value = event["event_type"].value  # "approve" or "reject"
            result = merge_nested_dict(
                self._status_key.to_nested_dict(WorkflowStatusEnum.EXECUTION.value),
                self._output.to_nested_dict(approval_value),
            )

            # For REJECT events, also add HumanMessage to conversation history
            if event["event_type"] == FlowEventType.REJECT:
                rejection_message = "User rejected this action. Do not proceed and stop any tool execution in progress."

                self._ui_history.log.success(
                    content="Action rejected.",
                    event=UILogEventsHumanInput.ON_USER_RESPONSE,
                )

                # Append user rejection message to target component's history for replace-based reducer.
                # The reducer will replace the target component's conversation history with this list.
                human_message = HumanMessage(content=rejection_message)
                conversation_history_dict = (
                    self._conversation_history_key.to_nested_dict(
                        existing_history + [human_message]
                    )
                )
                result = merge_nested_dict(result, conversation_history_dict)

            result.update(self._ui_history.pop_state_updates())

            return result

        if event["event_type"] in (FlowEventType.MODIFY, FlowEventType.RESPONSE):
            if "message" not in event or not event["message"]:
                event_name = event["event_type"].value.upper()
                raise InvalidRequestException(
                    f"{event_name} event must include a non-empty message. "
                    "The workflow remains paused; please provide real user input to continue."
                )

            # Extract user message from event
            user_message = event["message"]

            self._ui_history.log.success(
                content=user_message,
                event=UILogEventsHumanInput.ON_USER_RESPONSE,
            )

            # The UI log above already recorded the clean user message;
            # the cancelled-turn block is model-facing only.
            cancelled_turn_block = self._cancelled_turn_block(state)
            message_content = (
                f"{cancelled_turn_block}\n{user_message}"
                if cancelled_turn_block
                else user_message
            )

            # Create HumanMessage for conversation history
            human_message = HumanMessage(content=message_content)

            result = merge_nested_dict(
                self._ui_history.pop_state_updates(),
                self._status_key.to_nested_dict(WorkflowStatusEnum.EXECUTION.value),
            )

            if cancelled_turn_block:
                # Consume-once: clear the state location so subsequent user
                # messages don't re-inject stale cancelled context.
                result = merge_nested_dict(result, self._cancelled_turn_cleanup(state))

            if event["event_type"] == FlowEventType.MODIFY:
                result = merge_nested_dict(
                    result, self._output.to_nested_dict("modify")
                )

            conversation_history_dict = self._conversation_history_key.to_nested_dict(
                existing_history + [human_message]
            )
            result = merge_nested_dict(result, conversation_history_dict)

            return result

        # For any other event type, raise error as this should not happen
        raise NotifiableAgentException(
            "An internal error occurred: an unexpected event type was received.",
            internal_detail=f"Unknown event type: {event['event_type']}. Expected one of: {list(FlowEventType)}",
        )

    def _cancelled_turn_block(self, state: FlowState) -> Optional[str]:
        """Render the cancelled-turn transcript wrapped in meta tags.

        Reads the ``ui_chat_log`` delta from ``cancelled_turn_key`` and formats
        it as a plain ``USER:``/``AGENT:`` transcript. Returns ``None`` when no
        cancelled-turn context is available (the common case) so the caller can
        skip injection entirely.
        """
        entries = self._cancelled_turn_key.value_from_state(state)
        if not entries or not isinstance(entries, list):
            # Covers the absent envelope (None / empty), NoneIOKey (always
            # None), and shape violations: a literal override resolves to its
            # static text, and a misdirected override may point at a non-list
            # state value - neither is a ui_chat_log delta, so injection (and
            # therefore the consume-once cleanup) is skipped entirely.
            return None

        lines = []
        for entry in entries:
            content = entry.get("content")
            if not content:
                continue
            message_type = entry.get("message_type")
            label = message_type.upper() if message_type else "UNKNOWN"
            lines.append(f"{label}: {content}")

        if not lines:
            return None

        transcript = "\n".join(lines)
        return (
            f"{_CANCELLED_TURN_OPEN_TAG}\n"
            f"{_CANCELLED_TURN_PREAMBLE}\n"
            f"{transcript}\n"
            f"{_CANCELLED_TURN_CLOSE_TAG}"
        )

    def _cancelled_turn_cleanup(self, state: FlowState) -> dict[str, Any]:
        """Build the state update that clears the consumed cancelled-turn context.

        Dispatches on the concrete key type: ``RuntimeIOKey`` resolves its
        location from state, plain ``IOKey`` writes directly, and any other key
        (e.g. ``NoneIOKey``) has no backing location to clear.
        """
        if isinstance(self._cancelled_turn_key, RuntimeIOKey):
            return self._cancelled_turn_key.to_nested_dict(None, state)
        if isinstance(self._cancelled_turn_key, IOKey):
            return self._cancelled_turn_key.to_nested_dict(None)
        return {}
