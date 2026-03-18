import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.audit_events.collector import AuditEventCollector
from tests.duo_workflow_service.audit_events.conftest import make_audit_event


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


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_sends_buffered_events(self, collector, mock_client):
        collector.capture(make_audit_event())
        collector.capture(make_audit_event())
        await collector.flush()
        mock_client.send_batch.assert_called_once()
        assert len(mock_client.send_batch.call_args[0][0]) == 2

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self, collector, mock_client):
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
