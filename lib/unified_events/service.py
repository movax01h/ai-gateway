import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, cast

import requests
import structlog
from pydantic import ValidationError

from lib.internal_events.ai_context import AIContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    merge_request_url_context,
    pipeline_source_context,
    tracked_internal_events,
)
from lib.unified_events.context import DeploymentType, EventConstructContext, Realm

__all__ = ["UnifiedEventService"]


class UnifiedEventService:
    """Sends events with the unified event construct to the events sidecar."""

    AI_CONTEXT_SCHEMA = InternalEventsClient.AI_CONTEXT_SCHEMA
    SIDECAR_URL = "http://localhost:8082"
    ENDPOINT_PATH = "/v1/events"
    REQUEST_TIMEOUT = (5.0, 10.0)  # requests' (connect, read) timeouts, in seconds
    MAX_SE_FIELD_LENGTH = 255
    AI_CONTEXT_TOKEN_KWARGS = frozenset(
        {"input_tokens", "output_tokens", "total_tokens"}
    )

    def __init__(self) -> None:
        self._logger = structlog.stdlib.get_logger("unified_event_service")
        self._session = requests.Session()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="unified_events"
        )

    def track(
        self,
        event_name: str,
        additional_properties: Optional[InternalEventAdditionalProperties] = None,
        category: Optional[str] = "default_category",
        **kwargs: Any,
    ) -> None:
        """Track an event by POSTing it to the events sidecar."""
        if additional_properties is None:
            additional_properties = InternalEventAdditionalProperties()

        context: EventContext = current_event_context.get()
        event_values = {
            **context.model_dump(exclude={"user_id"}),
            **kwargs,
        }
        passthrough_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.AI_CONTEXT_TOKEN_KWARGS
        }
        extra = {
            **(context.extra or {}),
            **passthrough_kwargs,
            **additional_properties.extra,
        }
        mr_url = merge_request_url_context.get(None)
        if mr_url:
            extra["merge_request_url"] = mr_url
        p_source = pipeline_source_context.get(None)
        if p_source:
            extra["pipeline_source"] = p_source

        session_id = additional_properties.value
        event_property = _truncate(
            additional_properties.property, self.MAX_SE_FIELD_LENGTH
        )

        ai_context = AIContext(
            session_id=str(session_id) if session_id is not None else None,
            workflow_id=extra.get("workflow_id"),
            flow_type=extra.get("workflow_type"),
            agent_name=extra.get("agent_name"),
            input_tokens=event_values.get("input_tokens"),
            output_tokens=event_values.get("output_tokens"),
            total_tokens=event_values.get("total_tokens"),
            ephemeral_5m_input_tokens=extra.get("ephemeral_5m_input_tokens"),
            ephemeral_1h_input_tokens=extra.get("ephemeral_1h_input_tokens"),
            cache_read=extra.get("cache_read"),
            cache_creation=extra.get("cache_creation"),
        )

        event_construct = self._build_event_construct(event_name, context, extra)
        if event_construct is None:
            return

        body = event_construct.model_dump(exclude_none=True)
        body["category"] = _truncate(
            category or "default_category", self.MAX_SE_FIELD_LENGTH
        )
        if additional_properties.label is not None:
            body["label"] = _truncate(
                additional_properties.label, self.MAX_SE_FIELD_LENGTH
            )
        if event_property is not None:
            body["property"] = event_property
        if additional_properties.value is not None:
            body["value"] = additional_properties.value
        body["contexts"] = [
            {"schema": self.AI_CONTEXT_SCHEMA, "data": asdict(ai_context)},
        ]

        self._logger.info("Tracking unified event", event_name=event_name)
        self._executor.submit(self._post_event, event_name, body)
        tracked_internal_events.get().add(event_name)

    def _post_event(self, event_name: str, body: Dict[str, Any]) -> None:
        url = f"{self.SIDECAR_URL}{self.ENDPOINT_PATH}"
        try:
            response = self._session.post(url, json=body, timeout=self.REQUEST_TIMEOUT)
        except requests.RequestException as e:
            self._logger.error(
                "Failed to send unified event to sidecar",
                event_name=event_name,
                error=str(e),
            )
            return

        if response.status_code == 202:
            self._logger.info(
                "Unified event accepted by sidecar",
                event_name=event_name,
                event_id=body.get("event_id"),
            )
        else:
            self._logger.error(
                "Unified event rejected by sidecar",
                event_name=event_name,
                event_id=body.get("event_id"),
                status_code=response.status_code,
                response_body=_truncate(response.text),
            )

    def _build_event_construct(
        self,
        event_name: str,
        context: EventContext,
        metadata: Dict[str, Any],
    ) -> Optional[EventConstructContext]:
        """Build the event_construct from the current event context."""
        try:
            return EventConstructContext(
                event_id=str(uuid.uuid4()),
                event_type=event_name,
                timestamp=datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                environment=context.environment or "development",
                source=context.source,
                correlation_id=context.correlation_id,
                global_user_id=context.global_user_id,
                user_id=context.user_id,
                is_gitlab_team_member=context.is_gitlab_team_member,
                deployment_type=cast(Optional[DeploymentType], context.deployment_type),
                realm=cast(Optional[Realm], context.realm),
                unique_instance_id=context.unique_instance_id,
                host_name=context.host_name,
                instance_version=context.instance_version,
                organization_id=context.organization_id,
                root_namespace_id=context.ultimate_parent_namespace_id,
                namespace_id=context.namespace_id,
                project_id=context.project_id,
                feature_enablement_type=context.feature_enablement_type,
                plan=context.plan,
                metadata=metadata,
            )
        except ValidationError as e:
            self._logger.error(
                "Failed to build event construct, dropping event",
                event_name=event_name,
                error=str(e),
            )
            return None


def _truncate(
    value: str | None, max_length: int = InternalEventsClient.MAX_VALUE_LENGTH
) -> str | None:
    if value and len(value) > max_length:
        return value[: max_length - 3] + "..."
    return value
