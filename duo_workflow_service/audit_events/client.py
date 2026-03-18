import asyncio
import json
import uuid
from typing import Optional

import structlog
from packaging.version import InvalidVersion, Version

from duo_workflow_service.audit_events.event_types import AuditEvent
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from lib.context import gitlab_version
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

logger = structlog.stdlib.get_logger("audit_event_client")

# TODO: Update to the actual GitLab version that ships the /audit_events API endpoint.
AUDIT_EVENTS_MIN_GITLAB_VERSION = Version("99.99.0")


class AuditEventClient:
    def __init__(
        self,
        http_client: GitlabHttpClient,
        workflow_id: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self._http_client = http_client
        self._workflow_id = workflow_id
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def _is_supported(self) -> bool:
        """Check if the connected GitLab instance supports audit events.

        Audit events require both:
        1. The duo_workflow_audit_events feature flag enabled on the Rails side
        2. A GitLab version that ships the /audit_events endpoint
        """
        if not is_feature_enabled(FeatureFlag.DUO_WORKFLOW_AUDIT_EVENTS):
            return False

        try:
            gl_version = Version(gitlab_version.get())  # type: ignore[arg-type]
        except (InvalidVersion, TypeError):
            return False

        return gl_version >= AUDIT_EVENTS_MIN_GITLAB_VERSION

    async def send_batch(
        self,
        events: list[AuditEvent],
        is_final: bool = False,
        total_events_sent: Optional[int] = None,
    ) -> bool:
        if not self._is_supported:
            return True
        if not events and not is_final:
            return True

        batch_id = str(uuid.uuid4())
        payload_dict: dict = {
            "events": [event.to_cloudevent() for event in events],
            "batch_id": batch_id,
        }
        if is_final:
            payload_dict["final"] = True
            payload_dict["total_events_sent"] = total_events_sent

        payload = json.dumps(payload_dict)
        path = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/audit_events"

        for attempt in range(self._max_retries):
            try:
                response = await self._http_client.apost(
                    path=path,
                    body=payload,
                    parse_json=False,
                )
                if response.is_success():
                    logger.info(
                        "Audit events sent",
                        workflow_id=self._workflow_id,
                        count=len(events),
                    )
                    return True

                if (
                    response.status_code in (429, 500, 503, 529)
                    and attempt < self._max_retries - 1
                ):
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Audit event send failed, retrying",
                        status_code=response.status_code,
                        attempt=attempt + 1,
                        retry_after=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    "Audit event send failed",
                    status_code=response.status_code,
                    response_body=response.body,
                )
                return False
            except Exception as e:
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2**attempt)
                    logger.warning(
                        "Audit event send exception, retrying",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Audit event send failed after retries", error=str(e))
                break

        logger.error(
            "Dropping audit events after retries exhausted",
            workflow_id=self._workflow_id,
            count=len(events),
            event_types=[event.event_type.value for event in events],
        )
        return False
