import asyncio
import json
from typing import Optional

import structlog

from duo_workflow_service.audit_events.client import AuditEventClient
from duo_workflow_service.audit_events.event_types import AuditEvent
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.workflows.type_definitions import MAX_MESSAGE_SIZE

logger = structlog.stdlib.get_logger("audit_event_collector")

# Derived from the transport limit so the two cannot drift. The 256 KiB
# remainder covers the JSON envelope and the runHTTPRequest proto/HTTP framing.
MAX_BUFFER_BYTES = MAX_MESSAGE_SIZE - (256 * 1024)

# Bound the shutdown drain so a stuck send cannot hold close() past the Cloud Run
# SIGKILL window (SIGTERM, then SIGKILL ~10s later).
DRAIN_TIMEOUT_SECONDS = 3.0


class AuditEventCollector:
    def __init__(
        self,
        client: AuditEventClient,
        workflow_id: str = "",
        buffer_size: int = 100,
        flush_interval_seconds: float = 10.0,
    ):
        self._client = client
        self._workflow_id = workflow_id
        self._buffer: list[AuditEvent] = []
        self._buffer_bytes: int = 0
        self._buffer_size = buffer_size
        self._flush_interval_seconds = flush_interval_seconds
        self._flush_task: Optional[asyncio.Task] = None
        self._pending_sends: set[asyncio.Task] = set()
        self._sequence: int = 0

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    def capture(self, event: AuditEvent) -> None:
        # Measure with the tentative sequence so the size matches the wire;
        # commit it only if the event is kept, so a drop consumes no sequence.
        next_sequence = self._sequence + 1
        event.sequence = next_sequence
        event_bytes = self._event_bytes(event)

        # An event over the cap can never be sent under the 4MB gRPC limit; drop
        # it rather than fail the send.
        if event_bytes > MAX_BUFFER_BYTES:
            largest_field, largest_field_bytes = self._largest_field(event)
            logger.warning(
                "Dropping audit event larger than the size cap",
                workflow_id=self._workflow_id,
                event_type=event.event_type.value,
                event_bytes=event_bytes,
                max_buffer_bytes=MAX_BUFFER_BYTES,
                largest_field=largest_field,
                largest_field_bytes=largest_field_bytes,
            )
            duo_workflow_metrics.count_audit_events_dropped(
                reason="event_too_large", amount=1
            )
            return

        self._sequence = next_sequence

        # Flush the buffered batch before this event would push it over the cap.
        if self._buffer and self._buffer_bytes + event_bytes > MAX_BUFFER_BYTES:
            self._schedule_flush()

        self._buffer.append(event)
        self._buffer_bytes += event_bytes
        duo_workflow_metrics.count_audit_events_captured(
            event_type=event.event_type.value
        )

        if len(self._buffer) >= self._buffer_size:
            self._schedule_flush()

    async def flush(self, is_final: bool = False) -> None:
        events_to_send = self._detach_buffer()
        total = self._sequence if is_final else None
        await self._send(events_to_send, is_final=is_final, total_events_sent=total)

    async def start(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def close(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        pending = len(self._buffer)
        try:
            # Drain in-flight sends before the final flush, inside the try so a
            # cancel here still records a dropped flush; bounded by the timeout.
            if self._pending_sends:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._pending_sends, return_exceptions=True),
                        timeout=DRAIN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Audit send drain timed out; sending the final flush anyway",
                        workflow_id=self._workflow_id,
                    )
            pending = len(self._buffer)
            await self.flush(is_final=self._sequence > 0)
        except asyncio.CancelledError:
            if pending:
                duo_workflow_metrics.count_audit_events_dropped(
                    reason="cancelled", amount=pending
                )
            logger.warning(
                "Final audit event flush cancelled; events may be lost",
                workflow_id=self._workflow_id,
            )

    @staticmethod
    def _event_bytes(event: AuditEvent) -> int:
        return len(json.dumps(event.to_cloudevent()).encode("utf-8"))

    @staticmethod
    def _largest_field(event: AuditEvent) -> tuple[Optional[str], int]:
        data = event.to_cloudevent().get("data") or {}
        sizes = {
            key: len(json.dumps(value).encode("utf-8")) for key, value in data.items()
        }
        if not sizes:
            return None, 0
        field = max(sizes, key=lambda key: sizes[key])
        return field, sizes[field]

    def _schedule_flush(self) -> None:
        if not self._buffer:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop, skipping auto-flush")
            duo_workflow_metrics.count_audit_events_auto_flush_skipped()
            return

        batch = self._detach_buffer()
        # Hold a strong reference until done: asyncio only weakly references
        # tasks, so an un-kept one can be garbage-collected mid-send.
        task = loop.create_task(self._send(batch))
        self._pending_sends.add(task)
        task.add_done_callback(self._on_send_done)

    def _on_send_done(self, task: asyncio.Task) -> None:
        self._pending_sends.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "Audit event send task raised an exception",
                workflow_id=self._workflow_id,
                error=str(task.exception()),
            )

    def _detach_buffer(self) -> list[AuditEvent]:
        batch = self._buffer
        self._buffer = []
        self._buffer_bytes = 0
        return batch

    async def _send(
        self,
        events: list[AuditEvent],
        is_final: bool = False,
        total_events_sent: Optional[int] = None,
    ) -> None:
        if not events and not is_final:
            return
        await self._client.send_batch(
            events, is_final=is_final, total_events_sent=total_events_sent
        )

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval_seconds)
                await self.flush()
        except asyncio.CancelledError:
            pass
