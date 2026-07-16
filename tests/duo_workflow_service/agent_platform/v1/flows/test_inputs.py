"""Tests for duo_workflow_service.agent_platform.v1.flows.inputs.

Exercises only the module's public interface, ``cancelled_turn_context`` — including the
delta-computation behavior (prefix delta, filtering, caps, graceful degradation) that is
implemented by the private ``_compute_cancelled_turn_delta`` helper, since every one of those
scenarios is reachable through the public entry point by wrapping the same ``ui_chat_log`` lists in
``CheckpointTuple``s.
"""

import pytest
import structlog
from langgraph.checkpoint.base import CheckpointTuple
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.flows.inputs import (
    _CANCELLED_TURN_MAX_BYTES,
    _CANCELLED_TURN_MAX_ENTRIES,
    cancelled_turn_context,
)
from duo_workflow_service.entities.state import MessageTypeEnum, ToolStatus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _entry(message_type, content, tool_info=None):
    return {
        "message_type": message_type,
        "message_sub_type": None,
        "content": content,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "status": ToolStatus.SUCCESS,
        "correlation_id": None,
        "tool_info": tool_info,
        "additional_context": None,
        "message_id": f"msg-{content[:8]}",
    }


def _log():
    return structlog.stdlib.get_logger("test")


def _checkpoint_with_ui_chat_log(ui_chat_log):
    """Build a minimal CheckpointTuple carrying the given ui_chat_log."""
    return CheckpointTuple(
        config={"configurable": {"thread_id": "test", "checkpoint_id": "cp-0"}},
        checkpoint={"channel_values": {"ui_chat_log": ui_chat_log}},
        metadata={},
        parent_config=None,
    )


# ---------------------------------------------------------------------------
# TestCancelledTurnContext
# ---------------------------------------------------------------------------


class TestCancelledTurnContext:
    """Unit tests for the cancelled_turn_context public entry point.

    Covers ui_chat_log extraction from CheckpointTuples, the prefix-delta computation, entry-type filtering, tool_info
    stripping, size caps, and graceful-degradation branches — all through the single public function the rest of the
    codebase (and this module's callers) actually use.
    """

    def test_basic_delta_filters_tool_entries(self):
        """Delta contains only USER and AGENT entries beyond the boundary; TOOL entries are filtered out."""
        boundary_log = [_entry(MessageTypeEnum.TOOL, "Starting Flow")]
        tip_log = boundary_log + [
            _entry(MessageTypeEnum.USER, "Create an http server"),
            _entry(MessageTypeEnum.TOOL, "tool call"),
            _entry(MessageTypeEnum.AGENT, "I will create..."),
        ]

        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(tip_log),
            boundary=_checkpoint_with_ui_chat_log(boundary_log),
            log=_log(),
        )

        assert len(result) == 2
        assert result[0]["message_type"] == MessageTypeEnum.USER
        assert result[1]["message_type"] == MessageTypeEnum.AGENT

    def test_tool_info_stripped(self):
        """tool_info payloads are stripped from the delta entries."""
        tip_log = [
            _entry(
                MessageTypeEnum.AGENT,
                "Running tool",
                tool_info={"name": "read_file", "args": {}, "tool_response": "data"},
            )
        ]

        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(tip_log),
            boundary=_checkpoint_with_ui_chat_log([]),
            log=_log(),
        )

        assert len(result) == 1
        assert result[0]["tool_info"] is None

    def test_returns_delta_when_tip_has_extra_entries(self):
        """When the tip has entries beyond the boundary, the delta is returned."""
        user_entry = _entry(MessageTypeEnum.USER, "Create an http server in java")
        agent_entry = _entry(MessageTypeEnum.AGENT, "I am going to create...")
        boundary_log = [_entry(MessageTypeEnum.TOOL, "Starting Flow")]
        tip_log = boundary_log + [user_entry, agent_entry]

        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(tip_log),
            boundary=_checkpoint_with_ui_chat_log(boundary_log),
            log=_log(),
        )

        # TOOL entry is filtered; USER and AGENT entries are kept.
        assert len(result) == 2
        assert result[0]["message_type"] == MessageTypeEnum.USER
        assert result[0]["content"] == "Create an http server in java"
        assert result[1]["message_type"] == MessageTypeEnum.AGENT
        assert result[1]["content"] == "I am going to create..."

    def test_returns_empty_when_latest_is_none(self):
        """When latest is None (no tip checkpoint), the delta is empty."""
        result = cancelled_turn_context(
            latest=None,
            boundary=_checkpoint_with_ui_chat_log([]),
            log=_log(),
        )

        assert result == []

    def test_returns_empty_when_boundary_is_none(self):
        """When boundary is None (flow stopped before first pause), boundary log is treated as empty."""
        tip_log = [
            _entry(MessageTypeEnum.USER, "Create an http server"),
            _entry(MessageTypeEnum.AGENT, "I will create..."),
        ]

        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(tip_log),
            boundary=None,
            log=_log(),
        )

        # With an empty boundary, the full tip (filtered to USER/AGENT) is the delta.
        assert len(result) == 2

    def test_returns_empty_when_tip_has_no_ui_chat_log(self):
        """When the tip checkpoint carries no ui_chat_log key, the delta is empty."""
        tip = CheckpointTuple(
            config={"configurable": {"thread_id": "test", "checkpoint_id": "cp-0"}},
            checkpoint={"channel_values": {}},  # no ui_chat_log key
            metadata={},
            parent_config=None,
        )

        result = cancelled_turn_context(
            latest=tip,
            boundary=_checkpoint_with_ui_chat_log([]),
            log=_log(),
        )

        assert result == []

    def test_returns_empty_when_tip_ui_chat_log_is_none(self):
        """When the tip checkpoint's ui_chat_log is explicitly None, the delta is empty."""
        tip = CheckpointTuple(
            config={"configurable": {"thread_id": "test", "checkpoint_id": "cp-0"}},
            checkpoint={"channel_values": {"ui_chat_log": None}},
            metadata={},
            parent_config=None,
        )

        result = cancelled_turn_context(
            latest=tip,
            boundary=_checkpoint_with_ui_chat_log([]),
            log=_log(),
        )

        assert result == []

    def test_returns_empty_when_tip_equals_boundary(self):
        """When tip and boundary carry the same ui_chat_log, the delta is empty."""
        log_entries = [_entry(MessageTypeEnum.USER, "hello")]

        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(log_entries),
            boundary=_checkpoint_with_ui_chat_log(log_entries),
            log=_log(),
        )

        assert result == []

    def test_prefix_violation_degrades_gracefully(self):
        """When the prefix assumption is violated, the delta is empty and a warning is logged."""
        boundary_log = [_entry(MessageTypeEnum.USER, "original")]
        tip_log = [_entry(MessageTypeEnum.USER, "different")]  # prefix mismatch

        with capture_logs() as cap_logs:
            result = cancelled_turn_context(
                latest=_checkpoint_with_ui_chat_log(tip_log),
                boundary=_checkpoint_with_ui_chat_log(boundary_log),
                log=_log(),
            )

        assert result == []
        warnings = [
            log for log in cap_logs if "prefix assumption" in log.get("event", "")
        ]
        assert warnings, "Expected a prefix-violation warning"

    def test_boundary_longer_than_tip_degrades_gracefully(self):
        """When boundary is longer than tip (impossible in normal operation), the delta is empty."""
        boundary_log = [
            _entry(MessageTypeEnum.USER, "a"),
            _entry(MessageTypeEnum.USER, "b"),
        ]
        tip_log = [_entry(MessageTypeEnum.USER, "a")]

        with capture_logs() as cap_logs:
            result = cancelled_turn_context(
                latest=_checkpoint_with_ui_chat_log(tip_log),
                boundary=_checkpoint_with_ui_chat_log(boundary_log),
                log=_log(),
            )

        assert result == []
        warnings = [
            log for log in cap_logs if "prefix assumption" in log.get("event", "")
        ]
        assert warnings

    def test_entry_count_cap(self):
        """Delta is truncated to _CANCELLED_TURN_MAX_ENTRIES entries."""
        tip_log = [
            _entry(MessageTypeEnum.USER, f"msg {i}")
            for i in range(_CANCELLED_TURN_MAX_ENTRIES + 5)
        ]

        with capture_logs() as cap_logs:
            result = cancelled_turn_context(
                latest=_checkpoint_with_ui_chat_log(tip_log),
                boundary=_checkpoint_with_ui_chat_log([]),
                log=_log(),
            )

        assert len(result) == _CANCELLED_TURN_MAX_ENTRIES
        warnings = [log for log in cap_logs if "entry cap" in log.get("event", "")]
        assert warnings, "Expected an entry-cap warning"

    def test_byte_size_cap_drops_oversized_delta(self):
        """Delta is dropped entirely when it exceeds _CANCELLED_TURN_MAX_BYTES."""
        # Each entry is ~1 KiB; create enough to exceed the cap.
        large_content = "x" * 2048
        tip_log = [
            _entry(MessageTypeEnum.USER, large_content)
            for _ in range((_CANCELLED_TURN_MAX_BYTES // 2048) + 2)
        ]

        with capture_logs() as cap_logs:
            result = cancelled_turn_context(
                latest=_checkpoint_with_ui_chat_log(tip_log),
                boundary=_checkpoint_with_ui_chat_log([]),
                log=_log(),
            )

        assert result == []
        warnings = [log for log in cap_logs if "byte cap" in log.get("event", "")]
        assert warnings, "Expected a byte-cap warning"

    def test_serialisation_failure_drops_delta(self):
        """When json.dumps raises during byte-size checking, the delta is dropped and a warning is logged."""

        class _Unserialisable:
            """An object whose __str__ raises, defeating json.dumps(default=str)."""

            def __str__(self):
                raise TypeError("cannot stringify this object")

        entry = _entry(MessageTypeEnum.USER, "hello")
        # additional_context survives tool_info-stripping, so an unserialisable
        # value there is what actually reaches json.dumps.
        entry["additional_context"] = _Unserialisable()
        tip_log = [entry]

        with capture_logs() as cap_logs:
            result = cancelled_turn_context(
                latest=_checkpoint_with_ui_chat_log(tip_log),
                boundary=_checkpoint_with_ui_chat_log([]),
                log=_log(),
            )

        assert result == []
        warnings = [
            log for log in cap_logs if "serialisation failed" in log.get("event", "")
        ]
        assert warnings, "Expected a serialisation-failure warning"

    @pytest.mark.parametrize(
        "tip_log,boundary_log,expected_count",
        [
            # Tip equals boundary → empty delta
            (
                [_entry(MessageTypeEnum.USER, "hello")],
                [_entry(MessageTypeEnum.USER, "hello")],
                0,
            ),
            # Tip has one extra USER entry beyond boundary
            (
                [
                    _entry(MessageTypeEnum.TOOL, "start"),
                    _entry(MessageTypeEnum.USER, "do it"),
                ],
                [_entry(MessageTypeEnum.TOOL, "start")],
                1,
            ),
        ],
        ids=["tip_equals_boundary", "one_extra_user_entry"],
    )
    def test_parametrized_delta_sizes(self, tip_log, boundary_log, expected_count):
        """Parametrized smoke-test for common delta-size scenarios."""
        result = cancelled_turn_context(
            latest=_checkpoint_with_ui_chat_log(tip_log),
            boundary=_checkpoint_with_ui_chat_log(boundary_log),
            log=_log(),
        )

        assert len(result) == expected_count
