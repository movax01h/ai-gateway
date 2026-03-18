import asyncio
from typing import Optional

import structlog

from duo_workflow_service.audit_events.client import AuditEventClient
from duo_workflow_service.audit_events.event_types import AuditEvent

logger = structlog.stdlib.get_logger("audit_event_collector")


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
        self._buffer_size = buffer_size
        self._flush_interval_seconds = flush_interval_seconds
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._sequence: int = 0

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    def capture(self, event: AuditEvent) -> None:
        self._sequence += 1
        event.sequence = self._sequence
        self._buffer.append(event)
        if len(self._buffer) >= self._buffer_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush())
            except RuntimeError:
                logger.warning("No running event loop, skipping auto-flush")

    async def flush(self, is_final: bool = False) -> None:
        async with self._lock:
            events_to_send = self._buffer.copy()
            self._buffer.clear()

        if not events_to_send and not is_final:
            return

        total = self._sequence if is_final else None
        await self._client.send_batch(
            events_to_send, is_final=is_final, total_events_sent=total
        )

    async def start(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._flush_interval_seconds)
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush(is_final=self._sequence > 0)
