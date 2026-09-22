import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.event_types import (
    LlmInputSentEvent,
    ToolInvokedEvent,
)
from duo_workflow_service.workflows.type_definitions import MAX_MESSAGE_SIZE
from tests.duo_workflow_service.audit_events.conftest import make_audit_event


def _sized_llm_event(prompt_chars, workflow_id="wf-1"):
    return LlmInputSentEvent(
        workflow_id=workflow_id, model_name="m", prompt_content="x" * prompt_chars
    )


@pytest.fixture(name="mock_client")
def mock_client_fixture():
    client = AsyncMock()
    client.send_batch = AsyncMock(return_value=True)
    return client


@pytest.fixture(name="collector")
def collector_fixture(mock_client):
    return AuditEventCollector(
        client=mock_client, buffer_size=3, flush_interval_seconds=0.05
    )


class TestCapture:
    def test_adds_event_to_buffer(self, collector):
        event = make_audit_event()
        collector.capture(event)
        assert len(collector._buffer) == 1
        assert collector._buffer[0] is event

    def test_does_not_flush_below_buffer_size(self, collector, mock_client):
        collector.capture(make_audit_event())
        collector.capture(make_audit_event())
        mock_client.send_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_flush_at_buffer_size(self, collector, mock_client):
        for i in range(3):
            collector.capture(make_audit_event(tool_name=f"tool_{i}"))
        await asyncio.sleep(0.01)
        mock_client.send_batch.assert_called_once()
        assert len(mock_client.send_batch.call_args[0][0]) == 3

    def test_no_event_loop_logs_warning(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1, flush_interval_seconds=1.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.asyncio"
        ) as mock_asyncio:
            mock_asyncio.get_running_loop.side_effect = RuntimeError("no loop")
            collector.capture(make_audit_event())
        assert len(collector._buffer) == 1


class TestCaptureMetrics:
    def test_capture_increments_captured_counter(self, collector):
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.collector.duo_workflow_metrics",
            mock_metrics,
        ):
            event = make_audit_event()
            collector.capture(event)

        mock_metrics.count_audit_events_captured.assert_called_once_with(
            event_type=event.event_type.value
        )

    def test_capture_no_loop_increments_auto_flush_skipped_counter(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1, flush_interval_seconds=1.0
        )
        mock_metrics = MagicMock()
        with (
            patch(
                "duo_workflow_service.audit_events.collector.duo_workflow_metrics",
                mock_metrics,
            ),
            patch(
                "duo_workflow_service.audit_events.collector.asyncio"
            ) as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError("no loop")
            collector.capture(make_audit_event())

        mock_metrics.count_audit_events_auto_flush_skipped.assert_called_once_with()
        mock_metrics.count_audit_events_dropped.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_no_loop_not_incremented_when_below_buffer_size(
        self, mock_client
    ):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=5, flush_interval_seconds=1.0
        )
        mock_metrics = MagicMock()
        with patch(
            "duo_workflow_service.audit_events.collector.duo_workflow_metrics",
            mock_metrics,
        ):
            collector.capture(make_audit_event())

        mock_metrics.count_audit_events_dropped.assert_not_called()


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_sends_buffered_events(self, collector, mock_client):
        collector.capture(make_audit_event())
        collector.capture(make_audit_event())
        await collector.flush()
        mock_client.send_batch.assert_called_once()
        assert len(mock_client.send_batch.call_args[0][0]) == 2

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self, collector):
        collector.capture(make_audit_event())
        await collector.flush()
        assert len(collector._buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_is_noop(self, collector, mock_client):
        await collector.flush()
        mock_client.send_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_passes_is_final_false_by_default(self, collector, mock_client):
        collector.capture(make_audit_event())
        await collector.flush()
        _, kwargs = mock_client.send_batch.call_args
        assert kwargs["is_final"] is False
        assert kwargs["total_events_sent"] is None

    @pytest.mark.asyncio
    async def test_flush_is_final_passes_total_and_flag(self, collector, mock_client):
        collector.capture(make_audit_event())
        collector.capture(make_audit_event())
        await collector.flush(is_final=True)
        _, kwargs = mock_client.send_batch.call_args
        assert kwargs["is_final"] is True
        assert kwargs["total_events_sent"] == 2

    @pytest.mark.asyncio
    async def test_flush_is_final_empty_buffer_still_sends(
        self, collector, mock_client
    ):
        collector.capture(make_audit_event())
        await collector.flush()  # drains buffer
        mock_client.send_batch.reset_mock()
        await collector.flush(is_final=True)
        mock_client.send_batch.assert_called_once()
        _, kwargs = mock_client.send_batch.call_args
        assert kwargs["is_final"] is True
        assert kwargs["total_events_sent"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_flush_safety(self, collector, mock_client):
        for _ in range(5):
            collector.capture(make_audit_event())
        await asyncio.gather(collector.flush(), collector.flush())
        total_events = sum(
            len(call.args[0]) for call in mock_client.send_batch.call_args_list
        )
        assert total_events == 5


class TestStartAndClose:
    @pytest.mark.asyncio
    async def test_periodic_flush(self, collector, mock_client):
        collector.capture(make_audit_event())
        await collector.start()
        await asyncio.sleep(0.1)
        await collector.close()
        assert mock_client.send_batch.call_count >= 1

    @pytest.mark.asyncio
    async def test_close_flushes_remaining(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100, flush_interval_seconds=100.0
        )
        collector.capture(make_audit_event())
        await collector.start()
        await collector.close()
        mock_client.send_batch.assert_called_once()
        assert len(mock_client.send_batch.call_args[0][0]) == 1

    @pytest.mark.asyncio
    async def test_close_without_start(self, collector, mock_client):
        collector.capture(make_audit_event())
        await collector.close()
        mock_client.send_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_cancels_flush_task(self, collector):
        await collector.start()
        assert collector._flush_task is not None
        assert not collector._flush_task.done()
        await collector.close()
        assert collector._flush_task.done()

    @pytest.mark.asyncio
    async def test_close_sends_final_signal(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100, flush_interval_seconds=100.0
        )
        collector.capture(make_audit_event())
        collector.capture(make_audit_event())
        await collector.close()
        _, kwargs = mock_client.send_batch.call_args
        assert kwargs["is_final"] is True
        assert kwargs["total_events_sent"] == 2

    @pytest.mark.asyncio
    async def test_close_sends_final_signal_even_when_buffer_already_drained(
        self, mock_client
    ):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100, flush_interval_seconds=100.0
        )
        collector.capture(make_audit_event())
        await collector.flush()  # drains buffer early
        mock_client.send_batch.reset_mock()
        await collector.close()
        mock_client.send_batch.assert_called_once()
        _, kwargs = mock_client.send_batch.call_args
        assert kwargs["is_final"] is True
        assert kwargs["total_events_sent"] == 1

    @pytest.mark.asyncio
    async def test_close_skips_final_signal_when_no_events_captured(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100, flush_interval_seconds=100.0
        )
        await collector.close()
        mock_client.send_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_final_flush_failure_is_non_recoverable(self, mock_client):
        mock_client.send_batch.return_value = False
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100, flush_interval_seconds=100.0
        )
        collector.capture(make_audit_event())
        await collector.close()
        # Events are dropped — no exception raised, no retry by collector
        mock_client.send_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_final_flush_cancelled_is_swallowed(self, mock_client):
        async def _slow_send_batch(*_args, **_kwargs):
            await asyncio.sleep(1)

        mock_client.send_batch.side_effect = _slow_send_batch
        collector = AuditEventCollector(
            client=mock_client,
            workflow_id="wf-123",
            buffer_size=100,
            flush_interval_seconds=100.0,
        )
        collector.capture(make_audit_event())

        close_task = asyncio.create_task(collector.close())
        await asyncio.sleep(0)  # let close() start awaiting flush()
        close_task.cancel()

        with (
            patch(
                "duo_workflow_service.audit_events.collector.duo_workflow_metrics"
            ) as mock_metrics,
            capture_logs() as cap_logs,
        ):
            await close_task  # must not raise CancelledError

        assert any(
            log["event"] == "Final audit event flush cancelled; events may be lost"
            and log["workflow_id"] == "wf-123"
            for log in cap_logs
        )
        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="cancelled", amount=1
        )

    @pytest.mark.asyncio
    async def test_intermediate_flushes_are_not_final(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=2, flush_interval_seconds=100.0
        )
        # Fill buffer twice to trigger two intermediate flushes
        for _ in range(2):
            collector.capture(make_audit_event())
        await asyncio.sleep(0.01)  # let auto-flush task run
        for _ in range(2):
            collector.capture(make_audit_event())
        await asyncio.sleep(0.01)
        # Check intermediate flushes were not final
        for call in mock_client.send_batch.call_args_list:
            _, kwargs = call
            assert kwargs["is_final"] is False


class TestByteBasedFlush:
    @pytest.mark.asyncio
    async def test_flushes_before_batch_exceeds_byte_cap(self, mock_client):
        # buffer_size high so only the byte threshold can drive the flushes
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000
        ):
            for _ in range(6):
                collector.capture(_sized_llm_event(400))
            await asyncio.sleep(0.01)

        assert mock_client.send_batch.call_count >= 2

    @pytest.mark.asyncio
    async def test_flushed_batches_stay_under_transport_limit(self, mock_client):
        # Every batch the collector hands its client must serialize under the
        # transport limit. Measured here independently of _event_bytes (the
        # estimator that drove the flush), so it catches drift, not itself.
        collector = AuditEventCollector(
            client=mock_client, buffer_size=100_000, flush_interval_seconds=100.0
        )
        for _ in range(20):
            collector.capture(_sized_llm_event(256 * 1024))  # ~256 KiB each
        await asyncio.sleep(0.05)
        await collector.close()

        assert mock_client.send_batch.call_count >= 1
        for call in mock_client.send_batch.call_args_list:
            batch = call.args[0]
            payload = json.dumps(
                {"events": [e.to_cloudevent() for e in batch], "batch_id": "x"}
            )
            assert len(payload.encode("utf-8")) <= MAX_MESSAGE_SIZE

    @pytest.mark.asyncio
    async def test_single_oversized_event_is_dropped(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000
        ):
            collector.capture(_sized_llm_event(5000))  # exceeds cap on its own
            collector.capture(_sized_llm_event(50))  # a normal event still buffers
            await asyncio.sleep(0.01)

        mock_client.send_batch.assert_not_called()
        assert len(collector._buffer) == 1
        assert collector._buffer[0].prompt_content == "x" * 50

    def test_oversized_event_logs_and_increments_dropped_metric(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        mock_metrics = MagicMock()
        with (
            patch("duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000),
            patch(
                "duo_workflow_service.audit_events.collector.duo_workflow_metrics",
                mock_metrics,
            ),
        ):
            collector.capture(_sized_llm_event(5000))

        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="event_too_large", amount=1
        )
        mock_metrics.count_audit_events_captured.assert_not_called()
        assert len(collector._buffer) == 0

    def test_oversized_event_warning_names_offending_field(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with (
            patch("duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000),
            patch("duo_workflow_service.audit_events.collector.logger") as mock_logger,
        ):
            collector.capture(_sized_llm_event(5000))

        mock_logger.warning.assert_called_once()
        _, kwargs = mock_logger.warning.call_args
        assert kwargs["event_type"] == "ai_llm_input_sent"
        assert kwargs["largest_field"] == "prompt_content"
        assert kwargs["largest_field_bytes"] >= 5000

    def test_dropped_event_does_not_consume_a_sequence(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000
        ):
            collector.capture(_sized_llm_event(5000))  # dropped
            collector.capture(_sized_llm_event(50))  # kept

        assert collector._sequence == 1
        assert collector._buffer[0].sequence == 1

    @pytest.mark.asyncio
    async def test_no_flush_while_under_both_thresholds(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 100000
        ):
            for _ in range(3):
                collector.capture(_sized_llm_event(400))

        mock_client.send_batch.assert_not_called()
        assert len(collector._buffer) == 3

    @pytest.mark.asyncio
    async def test_close_sends_final_in_single_post(self, mock_client):
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        collector.capture(_sized_llm_event(100))
        collector.capture(_sized_llm_event(100))
        await collector.close()

        final_calls = [
            c for c in mock_client.send_batch.call_args_list if c.kwargs.get("is_final")
        ]
        assert len(final_calls) == 1
        assert final_calls[0].kwargs["total_events_sent"] == 2

    @pytest.mark.asyncio
    async def test_close_awaits_in_flight_sends_before_final(self, mock_client):
        order = []

        async def _send_batch(events, is_final=False, total_events_sent=None):
            if is_final:
                order.append("final")
            else:
                await asyncio.sleep(0.02)  # a buffer-triggered send still in flight
                order.append("batch")
            return True

        mock_client.send_batch.side_effect = _send_batch
        collector = AuditEventCollector(
            client=mock_client, buffer_size=1000, flush_interval_seconds=100.0
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000
        ):
            for _ in range(6):
                collector.capture(_sized_llm_event(400))  # triggers a byte flush
            await collector.close()

        assert "batch" in order  # in-flight send was awaited, not orphaned
        assert order[-1] == "final"  # final flush went last

    @pytest.mark.asyncio
    async def test_close_cancel_during_drain_records_cancelled(self, mock_client):
        gate = asyncio.Event()  # never set: the in-flight send blocks the drain

        async def _send(events, is_final=False, total_events_sent=None):
            if not is_final:
                await gate.wait()
            return True

        mock_client.send_batch.side_effect = _send
        collector = AuditEventCollector(
            client=mock_client,
            workflow_id="wf-123",
            buffer_size=1000,
            flush_interval_seconds=100.0,
        )
        with patch(
            "duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000
        ):
            for _ in range(6):
                collector.capture(_sized_llm_event(400))  # triggers a byte flush
            await asyncio.sleep(0.01)  # let the byte-flush send start (now gated)
            residual = len(collector._buffer)
            assert residual

            with (
                patch(
                    "duo_workflow_service.audit_events.collector.duo_workflow_metrics"
                ) as mock_metrics,
                capture_logs() as cap_logs,
            ):
                close_task = asyncio.create_task(collector.close())
                await asyncio.sleep(0.01)  # let close() reach the drain
                close_task.cancel()
                await close_task  # cancel during drain is recorded, not propagated

        mock_metrics.count_audit_events_dropped.assert_called_once_with(
            reason="cancelled", amount=residual
        )
        assert any(
            log["event"] == "Final audit event flush cancelled; events may be lost"
            for log in cap_logs
        )

    @pytest.mark.asyncio
    async def test_send_task_exception_is_logged(self, mock_client):
        mock_client.send_batch.side_effect = RuntimeError("boom")
        collector = AuditEventCollector(
            client=mock_client,
            workflow_id="wf-9",
            buffer_size=1000,
            flush_interval_seconds=100.0,
        )
        with (
            patch("duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000),
            capture_logs() as cap_logs,
        ):
            for _ in range(6):
                collector.capture(_sized_llm_event(400))  # triggers a byte-flush send
            await asyncio.sleep(0.01)  # let the send task run and fail

        assert any(
            log["event"] == "Audit event send task raised an exception"
            and log.get("workflow_id") == "wf-9"
            for log in cap_logs
        )

    @pytest.mark.asyncio
    async def test_close_drain_timeout_still_sends_final(self, mock_client):
        gate = asyncio.Event()  # the in-flight send never completes
        sent_final = []

        async def _send(events, is_final=False, total_events_sent=None):
            if is_final:
                sent_final.append(True)
            else:
                await gate.wait()
            return True

        mock_client.send_batch.side_effect = _send
        collector = AuditEventCollector(
            client=mock_client,
            workflow_id="wf-1",
            buffer_size=1000,
            flush_interval_seconds=100.0,
        )
        with (
            patch("duo_workflow_service.audit_events.collector.MAX_BUFFER_BYTES", 2000),
            patch(
                "duo_workflow_service.audit_events.collector.DRAIN_TIMEOUT_SECONDS",
                0.01,
            ),
            capture_logs() as cap_logs,
        ):
            for _ in range(6):
                collector.capture(_sized_llm_event(400))  # spawns a gated send
            await asyncio.sleep(0)
            await collector.close()

        assert any(
            log["event"] == "Audit send drain timed out; sending the final flush anyway"
            for log in cap_logs
        )
        assert sent_final  # final flush still sent after the drain timed out

    @pytest.mark.asyncio
    async def test_schedule_flush_is_noop_on_empty_buffer(self, collector, mock_client):
        collector._schedule_flush()
        await asyncio.sleep(0.01)
        mock_client.send_batch.assert_not_called()


class TestLargestField:
    def test_picks_biggest_string_field(self):
        field, size = AuditEventCollector._largest_field(_sized_llm_event(500))
        assert field == "prompt_content"
        assert size >= 500

    def test_handles_dict_valued_field(self):
        event = ToolInvokedEvent(
            workflow_id="wf-1", tool_name="t", tool_args={"blob": "y" * 500}
        )
        field, size = AuditEventCollector._largest_field(event)
        assert field == "tool_args"
        assert size >= 500

    def test_returns_none_when_data_is_empty(self):
        event = MagicMock()
        event.to_cloudevent.return_value = {"data": {}}
        assert AuditEventCollector._largest_field(event) == (None, 0)
