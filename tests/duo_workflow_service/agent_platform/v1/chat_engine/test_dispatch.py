from itertools import product

import pytest

from duo_workflow_service.agent_platform.v1.chat_engine.dispatch import (
    _DISPATCH_TABLE,
    LEGACY_SESSION_MESSAGE,
    EntryDispatch,
    EntryEvent,
    EntryEventKind,
    EntryMode,
    EntryRejectedException,
    EntryTip,
    TipState,
    classify_entry,
    classify_tip,
)
from duo_workflow_service.errors.typing import InvalidRequestException

_FLOW_CHANNELS = frozenset(
    {"status", "conversation_history", "ui_chat_log", "context", "agent_context_limits"}
)
_LANGGRAPH_CHANNELS = frozenset({"__start__", "branch:to:chat_agent#agent"})


def _tip(
    *next_nodes: str,
    channels: frozenset[str] = _FLOW_CHANNELS | _LANGGRAPH_CHANNELS,
    pre_run_input: bool = False,
) -> EntryTip:
    return EntryTip(
        next_nodes=next_nodes, channels=channels, pre_run_input=pre_run_input
    )


BOUNDARY = _tip()
IN_TURN = _tip("chat_agent#tool_approval_fetch")
INPUT_CHECKPOINT = _tip("__start__", pre_run_input=True)
LEGACY = _tip(channels=_FLOW_CHANNELS | {"plan", "last_human_input"})

MESSAGE = EntryEvent(EntryEventKind.MESSAGE)
DECISION = EntryEvent(EntryEventKind.DECISION)
RECONNECT = EntryEvent(EntryEventKind.RECONNECT)
PINNED_MESSAGE = EntryEvent(EntryEventKind.MESSAGE, pinned=True)


@pytest.mark.parametrize(
    "has_user_content,decision,pinned,expected",
    [
        (True, None, False, EntryEvent(EntryEventKind.MESSAGE)),
        (True, "approval", False, EntryEvent(EntryEventKind.DECISION)),
        (False, "rejection", False, EntryEvent(EntryEventKind.DECISION)),
        (True, "rejection", False, EntryEvent(EntryEventKind.DECISION)),
        (False, None, False, EntryEvent(EntryEventKind.RECONNECT)),
        (True, None, True, EntryEvent(EntryEventKind.MESSAGE, pinned=True)),
    ],
    ids=[
        "content_is_a_message",
        "a_decision_outranks_content",
        "a_decision_without_content",
        "a_rejection_with_feedback_is_a_decision",
        "neither_is_a_reconnect",
        "a_pin_is_carried",
    ],
)
def test_entry_event_from_request(has_user_content, decision, pinned, expected):
    assert (
        EntryEvent.from_request(
            has_user_content=has_user_content, decision=decision, pinned=pinned
        )
        == expected
    )


@pytest.mark.parametrize(
    "tip,waiting_on_user,rolled_back,expected",
    [
        (None, False, False, TipState.NONE),
        (BOUNDARY, True, False, TipState.BOUNDARY),
        (BOUNDARY, False, False, TipState.BOUNDARY),
        (IN_TURN, True, False, TipState.WAITING),
        (IN_TURN, False, False, TipState.RUNNING),
        (_tip("developer", "tester"), True, False, TipState.WAITING),
        (_tip("design_gate#fetch"), True, False, TipState.WAITING),
        (_tip("chat_agent#tools"), False, False, TipState.RUNNING),
        (INPUT_CHECKPOINT, True, False, TipState.RUNNING),
        (INPUT_CHECKPOINT, True, True, TipState.BOUNDARY),
        (_tip("design_gate#fetch"), True, True, TipState.WAITING),
        (LEGACY, True, False, TipState.LEGACY),
        (
            _tip(channels=_FLOW_CHANNELS | {"plan"}, pre_run_input=True),
            False,
            True,
            TipState.LEGACY,
        ),
        (
            _tip(channels=frozenset({"__start__"}), pre_run_input=True),
            False,
            True,
            TipState.BOUNDARY,
        ),
        (
            _tip(channels=_FLOW_CHANNELS - {"agent_context_limits"}),
            False,
            False,
            TipState.BOUNDARY,
        ),
        (
            _tip(channels=_FLOW_CHANNELS | {"pregel_tasks"}),
            False,
            False,
            TipState.BOUNDARY,
        ),
    ],
    ids=[
        "no_checkpoint",
        "nothing_next_is_the_boundary_whatever_rails_says",
        "nothing_next_is_the_boundary",
        "a_turn_rails_says_waits_on_the_user",
        "a_turn_rails_says_is_running",
        "subagents_wait_without_the_engine_knowing_them",
        "a_gate_waits_without_the_engine_knowing_it",
        "a_running_node",
        "the_input_checkpoint_at_the_tip_is_a_run_in_progress",
        "a_rollback_onto_an_input_checkpoint_is_the_boundary_it_holds",
        "a_rollback_onto_a_gate_waits_there",
        "a_channel_only_legacy_chat_declares_is_legacy",
        "legacy_outranks_the_input_checkpoint",
        "the_first_turn_input_checkpoint_stores_only_langgraph_channels",
        "a_session_from_before_flow_state_gained_a_channel",
        "a_channel_legacy_chat_does_not_declare_is_not_legacy",
    ],
)
def test_classify_tip(tip, waiting_on_user, rolled_back, expected):
    assert (
        classify_tip(tip, waiting_on_user=waiting_on_user, rolled_back=rolled_back)
        == expected
    )


def test_every_tip_state_and_event_has_a_row():
    assert set(_DISPATCH_TABLE) == set(product(TipState, EntryEventKind))


@pytest.mark.parametrize(
    "tip,event,waiting_on_user,rolled_back,mode,tip_state",
    [
        (None, MESSAGE, False, False, EntryMode.START, TipState.NONE),
        (BOUNDARY, MESSAGE, True, False, EntryMode.TURN, TipState.BOUNDARY),
        (IN_TURN, MESSAGE, True, False, EntryMode.RESUME, TipState.WAITING),
        (IN_TURN, DECISION, True, False, EntryMode.RESUME, TipState.WAITING),
        (IN_TURN, RECONNECT, True, False, EntryMode.RETRY, TipState.WAITING),
        (IN_TURN, RECONNECT, False, False, EntryMode.RETRY, TipState.RUNNING),
        (BOUNDARY, PINNED_MESSAGE, False, False, EntryMode.TURN, TipState.BOUNDARY),
        (IN_TURN, PINNED_MESSAGE, True, False, EntryMode.TURN, TipState.RUNNING),
        (
            INPUT_CHECKPOINT,
            PINNED_MESSAGE,
            False,
            False,
            EntryMode.TURN,
            TipState.RUNNING,
        ),
        (INPUT_CHECKPOINT, MESSAGE, True, True, EntryMode.TURN, TipState.BOUNDARY),
        (BOUNDARY, MESSAGE, True, True, EntryMode.TURN, TipState.BOUNDARY),
        (IN_TURN, MESSAGE, True, True, EntryMode.RESUME, TipState.WAITING),
    ],
    ids=[
        "first_message_starts",
        "message_at_the_boundary_is_a_turn",
        "message_while_waiting_goes_to_whatever_waits",
        "decision_while_waiting_goes_to_whatever_waits",
        "reconnect_while_waiting_replays",
        "reconnect_while_running_replays",
        "pinned_boundary_takes_a_message_as_a_turn",
        "a_pinned_checkpoint_is_never_waiting",
        "a_pinned_input_checkpoint_takes_a_message_as_a_turn",
        "a_message_after_a_rollback_is_a_turn",
        "a_rollback_onto_the_boundary_is_a_turn",
        "a_rollback_onto_a_gate_answers_it",
    ],
)
def test_classify_entry_accepts(
    tip, event, waiting_on_user, rolled_back, mode, tip_state
):
    dispatch = classify_entry(
        tip, event, waiting_on_user=waiting_on_user, rolled_back=rolled_back
    )

    assert dispatch == EntryDispatch(mode, tip_state)


@pytest.mark.parametrize(
    "tip,event,reason",
    [
        (None, DECISION, "no checkpoints yet"),
        (None, RECONNECT, "no checkpoints yet"),
        (BOUNDARY, DECISION, "decision is stale"),
        (BOUNDARY, RECONNECT, "waiting for your next message"),
        (LEGACY, MESSAGE, LEGACY_SESSION_MESSAGE),
        (LEGACY, DECISION, LEGACY_SESSION_MESSAGE),
        (LEGACY, RECONNECT, LEGACY_SESSION_MESSAGE),
        (LEGACY, PINNED_MESSAGE, LEGACY_SESSION_MESSAGE),
        (
            LEGACY,
            EntryEvent(EntryEventKind.DECISION, pinned=True),
            LEGACY_SESSION_MESSAGE,
        ),
        (
            LEGACY,
            EntryEvent(EntryEventKind.RECONNECT, pinned=True),
            LEGACY_SESSION_MESSAGE,
        ),
        (None, PINNED_MESSAGE, "no checkpoints yet"),
        (
            IN_TURN,
            EntryEvent(EntryEventKind.DECISION, pinned=True),
            "takes a new message",
        ),
        (
            IN_TURN,
            EntryEvent(EntryEventKind.RECONNECT, pinned=True),
            "takes a new message",
        ),
    ],
    ids=[
        "decision_without_checkpoints",
        "reconnect_without_checkpoints",
        "decision_at_the_boundary",
        "reconnect_at_the_boundary",
        "message_on_a_legacy_session",
        "decision_on_a_legacy_session",
        "reconnect_on_a_legacy_session",
        "pinned_message_on_a_legacy_session",
        "pinned_decision_on_a_legacy_session",
        "pinned_reconnect_on_a_legacy_session",
        "pin_without_checkpoints",
        "decision_on_a_pin",
        "reconnect_on_a_pin",
    ],
)
def test_classify_entry_rejects(tip, event, reason):
    dispatch = classify_entry(tip, event, waiting_on_user=True)

    assert dispatch.mode is EntryMode.REJECT
    assert reason in dispatch.reason


@pytest.mark.parametrize("event", [MESSAGE, DECISION], ids=["message", "decision"])
@pytest.mark.parametrize(
    "tip",
    [IN_TURN, INPUT_CHECKPOINT],
    ids=["a_turn_in_progress", "a_run_that_never_started"],
)
def test_a_running_turn_asks_for_a_retry_first(tip, event):
    """A message isn't dropped, and a decision has nothing paused to answer.

    A retry is an entry with no message, which replays the turn through ``Flow``'s retry.
    """
    dispatch = classify_entry(tip, event, waiting_on_user=False)

    assert dispatch == EntryDispatch(
        EntryMode.REJECT, TipState.RUNNING, dispatch.reason
    )
    assert "Retry it before sending a new message or decision" in dispatch.reason


@pytest.mark.parametrize(
    "event,reason",
    [
        (DECISION, "decision is stale"),
        (RECONNECT, "waiting for your next message"),
    ],
    ids=["decision_after_a_rollback", "reconnect_after_a_rollback"],
)
def test_a_rollback_onto_an_input_checkpoint_rejects_like_the_boundary(event, reason):
    dispatch = classify_entry(
        INPUT_CHECKPOINT, event, waiting_on_user=True, rolled_back=True
    )

    assert dispatch.mode is EntryMode.REJECT
    assert reason in dispatch.reason


def test_a_reject_is_an_invalid_request_not_a_failure():
    assert issubclass(EntryRejectedException, InvalidRequestException)
