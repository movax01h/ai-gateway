# pylint: disable=super-init-not-called,direct-environment-variable-reference,broad-exception-raised,attribute-defined-outside-init,too-many-lines

import asyncio
import base64
import functools
import json
import os
import time
import zlib
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    override,
)

import structlog
import uuid_utils
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver

from ai_gateway.container import ContainerApplication
from duo_workflow_service.audit_events.context import get_audit_collector
from duo_workflow_service.audit_events.event_types import SessionEndedEvent
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    BILLABLE_STATUSES,
    CHECKPOINT_STATUS_TO_STATUS_EVENT,
    INTERNAL_TO_RAILS_STATUS_EVENT,
    NOOP_WORKFLOW_STATUSES,
    STATUS_TO_EVENT_PROPERTY,
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
    WorkflowStatusEventEnum,
    add_compression_param,
    compress_checkpoint,
    uncompress_checkpoint,
)
from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
from duo_workflow_service.gitlab.gitlab_api import Checkpoint as GitLabCheckpoint
from duo_workflow_service.gitlab.gitlab_api import (
    WorkflowConfig,
)
from duo_workflow_service.gitlab.http_client import (
    GitlabHttpClient,
    GitLabHttpResponse,
    checkpoint_decoder,
)
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.json_encoder.encoder import CustomEncoder
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.status_updater.gitlab_status_updater import (
    ForbiddenStatusEvent,
    GitLabStatusUpdater,
    UnsupportedStatusEvent,
)
from duo_workflow_service.tracking.duo_workflow_metrics import (
    SessionTypeEnum,
    session_type_context,
)
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.tracking.response_schema_tracking_context import (
    response_schema_tracking_results,
)
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
)
from lib.billing_events import BillingEvent, BillingEventService, ExecutionEnvironment
from lib.context import (
    build_orbit_session_summary_extras,
    current_model_metadata_context,
    init_llm_operations,
    init_orbit_counters,
    is_orbit_tool,
)
from lib.context.tool_executions import get_tool_executions, init_tool_executions
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum, EventPropertyEnum

T = TypeVar("T", bound=callable)  # type: ignore

_logger = structlog.stdlib.get_logger("workflow_checkpointer")


def not_implemented_sync_method(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            "The GitLabSaver does not support synchronous methods. "
        )

    return wrapper  # type: ignore


PROPERTY_MAX_LENGTH = 1000

# status is required for blob reconstruction
_ALWAYS_BLOBBED_SCALAR_CHANNELS = frozenset({"status"})


def _attribute_dirty(
    attribute: str, writes: Sequence[Tuple[str, Any]]
) -> tuple[bool, Any]:
    if not writes:
        return False, None

    for node_name, node_write in writes:
        if node_name == attribute:
            return True, node_write

    return False, None


class Delta(NamedTuple):
    values: dict | list
    is_append: bool


def _get_orbit_tool_calls(checkpoint: Checkpoint) -> bool:
    """Check if any Orbit tools were called in the checkpoint state."""
    channel_values = checkpoint.get("channel_values", {})
    ui_chat_log = channel_values.get("ui_chat_log", [])
    return any(
        is_orbit_tool((entry.get("tool_info") or {}).get("name", ""))
        for entry in ui_chat_log
    )


def _list_delta(prev: List[Any], current: List[Any]) -> Optional[Delta]:
    """Compute a delta for a list channel.

    Returns a Delta(values, is_append) or None if unchanged. is_append=True means values is only the newly appended
    tail; is_append=False means values is the full current list (shrink, reorder, or content change).
    """
    if len(current) > len(prev) and current[: len(prev)] == prev:
        return Delta(current[len(prev) :], True)
    if current != prev:
        return Delta(current, False)
    return None


def _dict_of_list_delta(
    prev: Dict[str, Any], current: Dict[str, Any]
) -> Optional[Delta]:
    """Compute a delta for a dict-of-lists channel (e.g. conversation_history).

    Returns a Delta(values, is_append) or None if unchanged. is_append=True means values is a per-key dict of only newly
    appended items (or changed non-list values); is_append=False means values is the full current dict (a list shrunk or
    its prefix changed for at least one key — i.e. compaction).
    """
    delta: Dict[str, Any] = {}
    for key, val in current.items():
        prev_val = prev.get(key)
        if isinstance(val, list) and isinstance(prev_val, list):
            list_delta = _list_delta(prev_val, val)
            if list_delta is None:
                continue
            if list_delta.is_append:
                delta[key] = list_delta.values
            else:
                return Delta(current, False)
        elif val != prev_val:
            delta[key] = val

    if not delta:
        return None
    return Delta(delta, True)


def _thread_started_at_from_id(checkpoint_id: str) -> Optional[str]:
    """ISO8601 UTC start time of a current_thread group, decoded from the group's first checkpoint id and floored to the
    second.

    ``uuid_utils.UUID.timestamp`` returns Unix milliseconds for v1/v6/v7, matching the value Rails derives via
    ``Gitlab::Utils.time_from_uuid`` (langgraph emits v6). Rails bounds the partition-scanned blob query with
    ``created_at >= current_thread_started_at``, so the marker must never be later than the earliest blob's
    ``created_at`` or that blob is dropped from reconstruction. Flooring to the second keeps the marker at-or-before the
    sub-second value Rails stores for the same id, and daily partitioning makes the slack irrelevant to pruning. Returns
    None for non-time-based or malformed ids (uuid_utils raises ValueError for both).
    """
    try:
        timestamp_ms = uuid_utils.UUID(checkpoint_id).timestamp
    except (ValueError, TypeError):
        return None

    return datetime.fromtimestamp(timestamp_ms // 1_000, tz=timezone.utc).isoformat()


def _serialize_channel_blobs(
    checkpoint: Checkpoint,
    new_versions: ChannelVersions,
    prev_channel_values: Dict[str, Any],
    *,
    force_rewrite: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Serialize only the channels that changed in this step as blobs.

    Uses new_versions to identify changed channels. Scalar channels are excluded (see inline comment). Each blob carries
    a step_action field ("conversation" for deltas, "compaction" for full replacements) which is the authoritative
    append-vs-replace signal for Rails. current_thread in the payload is a grouping hint only — it cannot be used as the
    sole signal because it resets to 0 on gateway restart, whereas step_action is derived from the actual channel values
    and remains correct regardless.

    When force_rewrite is True, delta computation is skipped and each blob is serialized as a full replacement with
    step_action='compaction'. is_compaction in the return is then False — the caller asked for the rewrite and is
    responsible for bumping current_thread itself.

    Returns (blobs, is_compaction) where is_compaction is True when delta analysis found a shrink or non-prefix change.
    """
    channel_values = checkpoint.get("channel_values", {})
    blobs = []
    is_compaction = False

    for channel, version in new_versions.items():
        if channel not in channel_values:
            _logger.warning(
                "Channel declared changed in new_versions but absent from channel_values",
                channel=channel,
            )
            continue

        val = channel_values[channel]
        # Scalars are recoverable from compressed_checkpoint, so exclude them —
        # except those needed for blob reconstruction.
        if (
            not isinstance(val, (list, dict))
            and channel not in _ALWAYS_BLOBBED_SCALAR_CHANNELS
        ):
            continue

        prev = prev_channel_values.get(channel)
        step_action = "compaction"

        if force_rewrite:
            pass
        elif isinstance(val, list) and isinstance(prev, list):
            delta = _list_delta(prev, val)
            if delta is None:
                continue
            new_val, is_append = delta
            if is_append:
                val = new_val
                step_action = "conversation"
            else:
                is_compaction = True
        elif isinstance(val, dict) and isinstance(prev, dict):
            dict_delta = _dict_of_list_delta(prev, val)
            if dict_delta is None:
                continue
            new_val, is_append = dict_delta
            if is_append:
                val = new_val
                step_action = "conversation"
            else:
                is_compaction = True

        # Encode with CustomEncoder JSON (not the langgraph msgpack serde) so blob
        # deltas match the header's channel_values representation, which Rails stores
        # as JSON via compress_checkpoint. That keeps reconstruction (which merges
        # blobs onto the JSON header) consistent and lets Rails decode without
        # reimplementing langgraph's msgpack extension types.
        bval = json.dumps(val, cls=CustomEncoder).encode("utf-8")
        blobs.append(
            {
                "channel": channel,
                "version": str(version),
                "data": base64.b64encode(zlib.compress(bval)).decode("utf-8"),
                "write_type": "json",
                "step_action": step_action,
            }
        )

    return blobs, is_compaction


def _serialize_all_channels_full(
    checkpoint: Checkpoint,
) -> List[Dict[str, Any]]:
    """Serialize every reconstructable channel as a full ``compaction`` snapshot.

    Emitted at the start of a new current_thread group so the group is self-contained:
    reconstruction folds these full snapshots plus the group's later ``conversation``
    deltas, without depending on a previous group or the full-checkpoint header (see
    https://gitlab.com/gitlab-org/gitlab/-/issues/605653). Channel selection and JSON
    encoding mirror _serialize_channel_blobs — list/dict channels plus the status
    scalar; other scalars are intentionally dropped. Versions come from the checkpoint
    (not new_versions), since unchanged channels must be re-seeded too.
    """
    channel_values = checkpoint.get("channel_values", {})
    channel_versions = checkpoint.get("channel_versions", {})
    blobs = []

    for channel, val in channel_values.items():
        if (
            not isinstance(val, (list, dict))
            and channel not in _ALWAYS_BLOBBED_SCALAR_CHANNELS
        ):
            continue

        bval = json.dumps(val, cls=CustomEncoder).encode("utf-8")
        blobs.append(
            {
                "channel": channel,
                "version": str(channel_versions.get(channel) or ""),
                "data": base64.b64encode(zlib.compress(bval)).decode("utf-8"),
                "write_type": "json",
                "step_action": "compaction",
            }
        )

    return blobs


class GitLabWorkflow(BaseCheckpointSaver[Any], AbstractAsyncContextManager[Any]):  # pylint: disable=too-many-instance-attributes
    _client: GitlabHttpClient
    _logger: structlog.stdlib.BoundLogger
    _workflow_config: WorkflowConfig

    @inject
    def __init__(
        self,
        client: GitlabHttpClient,
        workflow_id: str,
        workflow_type: GLReportingEventContext,
        workflow_config: WorkflowConfig,
        gitlab_status_update_callback: (
            Callable[[WorkflowStatusEventEnum], NoReturn] | None
        ) = None,
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        billing_event_service: BillingEventService = Provide[
            ContainerApplication.billing_event.service
        ],
    ):
        self._offline_mode = os.getenv("USE_MEMSAVER")
        self._client = client
        self._workflow_id = workflow_id
        self._status_handler = GitLabStatusUpdater(
            client, status_update_callback=gitlab_status_update_callback
        )
        self._logger = structlog.stdlib.get_logger("workflow_checkpointer")
        self._workflow_type = workflow_type
        self._workflow_config = workflow_config
        self._internal_event_client = internal_event_client
        self._billing_event_service = billing_event_service
        self._orbit_called = False
        self.serde = CheckpointSerializer()
        self._prev_channel_values: Dict[str, Any] = {}
        self._prev_checkpoint_id: Optional[str] = None
        self._current_thread: int = 0
        self._current_thread_started_at: Optional[str] = None

    @override
    @not_implemented_sync_method
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return None

    @override
    @not_implemented_sync_method
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        return iter([])

    @override
    @not_implemented_sync_method
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return RunnableConfig()

    @override
    @not_implemented_sync_method
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
        # We are ignoring this parameter for now since we don't care for the order the pending writes are fetched in
    ) -> None:
        return

    async def _update_workflow_status(self, status: WorkflowStatusEventEnum) -> None:
        if self._offline_mode:
            return

        rails_status = INTERNAL_TO_RAILS_STATUS_EVENT.get(status, status)
        await self._status_handler.update_workflow_status(
            self._workflow_id, rails_status
        )

    def _track_internal_event(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:
        if self._offline_mode:
            return

        self._record_metric(
            event_name=event_name,
            additional_properties=additional_properties,
        )

        self._logger.info("Tracking Internal event %s", event_name.value)
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._workflow_type.value,
        )

    def _record_metric(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:
        """Records metrics to prometheus for real-time monitoring."""

        # For flow start events
        if event_name == EventEnum.WORKFLOW_START:
            duo_workflow_metrics.count_agent_platform_session_start(
                flow_type=self._workflow_type.value,
            )

        # For flow retry events
        if event_name == EventEnum.WORKFLOW_RETRY:
            duo_workflow_metrics.count_agent_platform_session_retry(
                flow_type=self._workflow_type.value,
            )

        # For flow reject events
        if event_name == EventEnum.WORKFLOW_REJECT:
            duo_workflow_metrics.count_agent_platform_session_reject(
                flow_type=self._workflow_type.value,
            )

        # For flow resume events
        if event_name == EventEnum.WORKFLOW_RESUME:
            duo_workflow_metrics.count_agent_platform_session_resume(
                flow_type=self._workflow_type.value,
            )

        # For session success events
        if event_name == EventEnum.WORKFLOW_FINISH_SUCCESS:
            duo_workflow_metrics.count_agent_platform_session_success(
                flow_type=self._workflow_type.value,
            )

        if event_name == EventEnum.WORKFLOW_FINISH_FAILURE:
            error_type = additional_properties.extra.get("error_type", "unknown")
            if not error_type or error_type == "str":
                error_type = "unknown"
            duo_workflow_metrics.count_agent_platform_session_failure(
                flow_type=self._workflow_type.value,
                failure_reason=error_type,
            )

        if event_name == EventEnum.WORKFLOW_ABORTED:
            duo_workflow_metrics.count_agent_platform_session_abort(
                flow_type=self._workflow_type.value,
            )

    @override
    async def __aenter__(self) -> BaseCheckpointSaver:
        try:
            if self._offline_mode:
                return MemorySaver()

            init_llm_operations()

            response_schema_tracking_results.set({})

            init_tool_executions()

            init_orbit_counters()

            self._flow_start_time = time.time()

            config: RunnableConfig = {"configurable": {}}
            (
                self.initial_status_event,
                event_property,
            ) = await self._get_initial_status_event(config)
            await self._update_workflow_status(self.initial_status_event)

            if self.initial_status_event == WorkflowStatusEventEnum.START:
                label = EventLabelEnum.WORKFLOW_START_LABEL
                event_name = EventEnum.WORKFLOW_START
                session_type_context.set(SessionTypeEnum.START.value)
            elif self.initial_status_event == WorkflowStatusEventEnum.RETRY:
                label = EventLabelEnum.WORKFLOW_RESUME_LABEL
                event_name = EventEnum.WORKFLOW_RETRY
                session_type_context.set(SessionTypeEnum.RETRY.value)
            elif self.initial_status_event == WorkflowStatusEventEnum.RESUME:
                label = EventLabelEnum.WORKFLOW_RESUME_LABEL
                event_name = EventEnum.WORKFLOW_RESUME
                session_type_context.set(SessionTypeEnum.RESUME.value)
            elif self.initial_status_event == WorkflowStatusEventEnum.STOP_RECOVERY:
                # A stop-recovery is recorded in Rails as a `retry`, so reuse the
                # RETRY tracking labels — still an accurate description of what
                # Rails recorded.
                label = EventLabelEnum.WORKFLOW_RESUME_LABEL
                event_name = EventEnum.WORKFLOW_RETRY
                session_type_context.set(SessionTypeEnum.RETRY.value)
            else:
                # no event to track
                return self

            additional_properties = InternalEventAdditionalProperties(
                label=label.value,
                property=event_property.value,
                value=self._workflow_id,
            )
            self._track_internal_event(
                event_name=event_name,
                additional_properties=additional_properties,
            )

            collector = get_audit_collector()
            if collector:
                self._session_start_time = time.monotonic()

            return self
        except (UnsupportedStatusEvent, ForbiddenStatusEvent) as e:
            reject_properties = InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_REJECT_LABEL.value,
                property=repr(e),
                value=self._workflow_id,
            )
            log_exception(e, extra={"additional_properties": str(reject_properties)})
            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_REJECT,
                additional_properties=reject_properties,
            )
            raise e
        except Exception as e:
            failure_properties = InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                property=repr(e)[:PROPERTY_MAX_LENGTH],
                value=self._workflow_id,
                error_type=type(e).__name__,
            )
            log_exception(e, extra={"additional_properties": str(failure_properties)})
            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_FINISH_FAILURE,
                additional_properties=failure_properties,
            )

            try:
                await self._update_workflow_status(WorkflowStatusEventEnum.DROP)
            except Exception as status_error:
                log_exception(
                    status_error,
                    extra={
                        "workflow_id": self._workflow_id,
                        "context": "Failed to update workflow status during startup error",
                    },
                )

            raise

    async def _get_initial_status_event(
        self,
        config: RunnableConfig,  # pylint: disable=unused-argument
    ) -> tuple[WorkflowStatusEventEnum, EventPropertyEnum]:
        """Determine the workflow status event and event property.

        This method analyzes the current state of the workflow to determine whether
        it's a new workflow (START), a resumption of an interrupted workflow (RESUME),
        or a retry of an existing workflow (RETRY).

        Args:
            config: The runnable configuration for the workflow.

        Returns:
            A tuple containing:
                - WorkflowStatusEventEnum: The status event (START, RESUME, or RETRY)
                - EventPropertyEnum: The associated event property for tracking
        """
        checkpoint_tuple = (
            self._workflow_config.get("latest_checkpoint", None)
            or self._workflow_config["first_checkpoint"]
        )
        status = self._workflow_config["workflow_status"]

        if self._workflow_config["archived"]:
            error_msg = (
                "Archived workflow can not be executed. Please create a new workflow."
            )
            raise NotifiableException(error_msg) from UnsupportedStatusEvent(error_msg)

        if self._workflow_config["stalled"]:
            error_msg = (
                "Stalled workflow can not be executed. Please create a new workflow."
            )
            raise NotifiableException(error_msg) from UnsupportedStatusEvent(error_msg)

        if status in [
            WorkflowStatusEnum.INPUT_REQUIRED,
            WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
            WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        ]:
            if not checkpoint_tuple:
                self._logger.error(
                    "The workflow record of the GitLab database is a continuous state "
                    "but there are no associtated checkopints."
                    "This data integrity issue could be caused by the expiring mechanism of checkpoint records."
                    "We try to execute this workflow, however, there might be an unexpected behavior.",
                    **self._workflow_config,
                )

            return WorkflowStatusEventEnum.RESUME, STATUS_TO_EVENT_PROPERTY.get(
                status, EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN
            )

        if status == WorkflowStatusEnum.CREATED:
            if checkpoint_tuple:
                raise UnsupportedStatusEvent(
                    f"Workflow with status 'created' should not have existing checkpoints. "
                    f"Found checkpoint: {checkpoint_tuple}"
                )
            return WorkflowStatusEventEnum.START, EventPropertyEnum.WORKFLOW_ID

        if status == WorkflowStatusEnum.STOPPED:
            # Pure detection — no checkpoint access here. The resolution of this
            # DWS-internal signal into RESUME/START (or a legacy plain RETRY)
            # happens in AbstractWorkflow._resolve_stop_recovery; the wire event
            # sent to Rails is translated to `retry` in _update_workflow_status.
            return (
                WorkflowStatusEventEnum.STOP_RECOVERY,
                EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
            )

        return (
            WorkflowStatusEventEnum.RETRY,
            EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
        )

    def _capture_audit_session_ended(self, exc_type, exc_value):
        collector = get_audit_collector()
        if not collector:
            return

        audit_status = "success"
        if exc_type:
            audit_status = "failure"
            if isinstance(exc_value, asyncio.exceptions.CancelledError):
                if str(exc_value) == AIO_CANCEL_STOP_WORKFLOW_REQUEST:
                    audit_status = "stopped"
                else:
                    audit_status = "aborted"
        duration = None
        start = getattr(self, "_session_start_time", None)
        if start:
            duration = round(time.monotonic() - start, 3)
        collector.capture(
            SessionEndedEvent(
                workflow_id=self._workflow_id,
                status=audit_status,
                duration_seconds=duration,
                error_message=str(exc_value) if exc_value else None,
            )
        )

    @override
    async def __aexit__(self, exc_type, exc_value, trcback):
        """Handle workflow completion and tracking in both success and failure scenarios.

        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        self._capture_audit_session_ended(exc_type, exc_value)

        # In case of exception in async context manager,
        # update status to DROP, track failure event,
        # and return False
        if exc_type:
            # An InvalidRequestException (e.g. an empty-goal "reconnect"/auto-retry
            # that reaches a live interrupt with no real user input) must NOT
            # transition the workflow to FAILED. Skip both _handle_workflow_exception
            # and _update_workflow_status_safely(DROP) and let the exception propagate,
            # so the workflow stays resumable at its current status (e.g.
            # INPUT_REQUIRED) and the server layer maps it to INVALID_ARGUMENT.
            # See gitlab-org/gitlab#602799.
            if isinstance(exc_value, InvalidRequestException):
                # The resume/retry event sent in __aenter__ may have advanced the
                # Rails workflow state (e.g. input_required → running) before the
                # graph inspected and rejected the invalid input.  Reconcile Rails
                # back to whatever the latest checkpoint declares as the true pause
                # boundary, so the next reconnect sees the correct status and is
                # classified as RESUME (not RETRY).  See gitlab-org/gitlab#602799.
                self._logger.info(
                    "InvalidRequestException at interrupt; leaving workflow status "
                    "untouched (no DROP transition) so it stays resumable.",
                    workflow_id=self._workflow_id,
                )
                if not self._offline_mode:
                    await self._reconcile_session_status()
                return False

            stop_exception = str(exc_value) == AIO_CANCEL_STOP_WORKFLOW_REQUEST

            if not stop_exception:
                log_exception(
                    exc_value,
                    extra={"workflow_id": self._workflow_id, "source": __name__},
                )

            event = EventEnum.WORKFLOW_FINISH_FAILURE
            status = WorkflowStatusEventEnum.DROP

            if isinstance(exc_value, asyncio.exceptions.CancelledError):
                if stop_exception:
                    event = EventEnum.WORKFLOW_STOP
                    status = WorkflowStatusEventEnum.STOP
                else:
                    # When this workflow task is cancelled by `workflow_task.cancel`, `CancelledError` is raised.
                    event = EventEnum.WORKFLOW_ABORTED

            if not stop_exception:
                await self._handle_workflow_exception(exc_value, event)

            await self._update_workflow_status_safely(status)
            return False

        if not self._offline_mode:
            return await self._handle_online_mode_completion()

    async def _handle_online_mode_completion(self) -> Optional[bool]:
        """Handle workflow completion in online mode."""
        status = await self._status_handler.get_workflow_status(
            workflow_id=self._workflow_id
        )
        await self._track_workflow_completion(status)
        return True

    async def _handle_workflow_exception(
        self, exc_value: Any, event: EventEnum = EventEnum.WORKFLOW_FINISH_FAILURE
    ) -> None:
        """Track workflow failure event."""
        properties = InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
            property=repr(exc_value),
            value=self._workflow_id,
            error_type=type(exc_value).__name__,
        )

        self._track_internal_event(
            event_name=event,
            additional_properties=properties,
        )

    async def _track_workflow_completion(self, status: str) -> None:
        """Track successful workflow completion based on status."""

        # Track billing event for workflow completion
        if status in BILLABLE_STATUSES:
            try:
                user: CloudConnectorUser = current_user.get()
                tool_executions = get_tool_executions() or []
                self._billing_event_service.track_billing(
                    user,
                    self._workflow_type,
                    workflow_id=self._workflow_id,
                    event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                    execution_env=ExecutionEnvironment.DAP,
                    category=self.__class__.__name__,
                    unit_of_measure="request",
                    quantity=1,
                    tool_execs=tool_executions,
                    orbit_called=self._orbit_called,
                )
                self._logger.info(
                    "Successfully sent billing event for workflow %s", self._workflow_id
                )
            except Exception as e:
                log_exception(
                    e,
                    extra={
                        "context": "Error sending billing event for workflow",
                    },
                )
        else:
            self._logger.info(
                f"Billing Status '{status}' does not match billing event conditions"
            )

        if status == WorkflowStatusEnum.INPUT_REQUIRED:
            event = EventEnum.WORKFLOW_PAUSE
            label = EventLabelEnum.WORKFLOW_PAUSE_LABEL.value
            prop = EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN_AWAIT_INPUT.value
        elif status == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED:
            event = EventEnum.WORKFLOW_PAUSE
            label = EventLabelEnum.WORKFLOW_PAUSE_LABEL.value
            prop = EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN_AWAIT_APPROVAL.value
        elif status in ("finished", "stopped"):
            self.log_if_finished_due_to_stopped(status)
            event = EventEnum.WORKFLOW_FINISH_SUCCESS
            label = EventLabelEnum.WORKFLOW_FINISH_LABEL.value
            prop = STATUS_TO_EVENT_PROPERTY.get(
                status, EventPropertyEnum.WORKFLOW_ID
            ).value
        else:
            # No event to track for other statuses
            return

        extra_kwargs: dict[str, Any] = {}
        if hasattr(self, "_flow_start_time"):
            extra_kwargs["duration_seconds"] = round(
                time.time() - self._flow_start_time, 3
            )

        tracked_outputs = response_schema_tracking_results.get() or {}
        for component_name, output_data in tracked_outputs.items():
            extra_kwargs[f"{component_name}_output"] = json.dumps(
                output_data, default=str
            )
            self._logger.info(
                "Response schema output at workflow completion",
                component_name=component_name,
                response_output=output_data,
                workflow_outcome=status,
                workflow_id=self._workflow_id,
            )

        self._track_internal_event(
            event,
            InternalEventAdditionalProperties(
                label=label, property=prop, value=self._workflow_id, **extra_kwargs
            ),
        )

        # Only fire orbit session summary on terminal statuses. Pause statuses
        # (INPUT_REQUIRED, PLAN_APPROVAL_REQUIRED) would produce partial summaries
        # because init_orbit_counters() resets the counters on each resume.
        if status in ("finished", "stopped"):
            # label/property are intentionally omitted: they are validator-required
            # placeholders that will be dropped once the schema is published. See
            # orbit_dap_session_summary.yml and
            # https://gitlab.com/gitlab-org/gitlab/-/work_items/596959
            orbit_extras = build_orbit_session_summary_extras(
                self._workflow_id, self._workflow_type.value
            )
            if orbit_extras is not None:
                self._track_internal_event(
                    EventEnum.ORBIT_DAP_SESSION_SUMMARY,
                    InternalEventAdditionalProperties(**orbit_extras),
                )

    async def _update_workflow_status_safely(
        self, status: WorkflowStatusEventEnum = WorkflowStatusEventEnum.DROP
    ):
        """Attempt to update workflow status to DROP, handling any exceptions.

        Returns:
            bool: False to indicate non-successful completion
        """
        try:
            await self._update_workflow_status(status)
        except Exception as e:
            log_exception(e, extra={"workflow_id": self._workflow_id})
        return False

    async def _fetch_most_recent_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Fetch the most recent checkpoint dict directly from the Rails API.

        Always goes to the API — does not use the ``latest_checkpoint`` cached in
        ``_workflow_config``, which is populated at session start and may be stale.
        Returns the raw checkpoint dict (with ``checkpoint["checkpoint"]`` already
        decompressed) or ``None`` when no checkpoints exist yet.
        """
        endpoint = add_compression_param(
            f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints?per_page=1"
        )
        with duo_workflow_metrics.time_gitlab_response(
            endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints?per_page=1",
            method="GET",
        ):
            response = await self._client.aget(
                path=endpoint,
                object_hook=checkpoint_decoder,
            )

        if not response.is_success():
            self._logger.error(
                "Failed to fetch checkpoints",
                workflow_id=self._workflow_id,
                status_code=response.status_code,
                response_body=response.body,
            )
            raise Exception(f"Failed to fetch checkpoints: {response.body}")

        gl_checkpoints = response.body
        if not gl_checkpoints:
            return None

        checkpoint = gl_checkpoints[0]
        if "compressed_checkpoint" in checkpoint:
            checkpoint["checkpoint"] = uncompress_checkpoint(
                checkpoint["compressed_checkpoint"]
            )
        return checkpoint

    async def _get_latest_checkpoint_status(self) -> Optional[WorkflowStatusEnum]:
        """Return the workflow status from the most recent checkpoint.

        Primary source: ``_prev_channel_values["status"]``, which is kept current by
        ``_hydrate_incremental_state`` (session start) and ``aput`` (every write).
        Only populated when the client supports ``incremental_checkpoints``.

        Fallback: ``_fetch_most_recent_checkpoint`` — fetches fresh from the Rails
        API, bypassing the stale session-start cache.
        """
        checkpoint_status_value: Optional[WorkflowStatusEnum] = (
            self._prev_channel_values.get("status")
        )
        if checkpoint_status_value is not None:
            return checkpoint_status_value

        checkpoint = await self._fetch_most_recent_checkpoint()
        if checkpoint is None:
            return None
        return checkpoint["checkpoint"].get("channel_values", {}).get("status")

    async def _reconcile_session_status(self) -> None:
        """Reconcile Rails workflow status against the latest checkpoint.

        After an ``InvalidRequestException`` the session status in Rails may have
        been advanced (e.g. ``input_required`` → ``running``) by the ``resume``/
        ``retry`` event sent in ``__aenter__``, while the LangGraph checkpoint still
        reflects the true pause boundary (e.g. ``INPUT_REQUIRED``).  This method
        treats the checkpoint as the single source of truth and — when Rails disagrees
        — drives it back to the status the checkpoint declares.

        The method is intentionally decoupled from any specific component design
        (e.g. ``HumanInputComponent`` / ``FetchNode``): it does not assume *why* the
        mismatch occurred, only that the checkpoint is authoritative.
        """
        try:
            checkpoint_status_value = await self._get_latest_checkpoint_status()

            if checkpoint_status_value is None:
                self._logger.info(
                    "No checkpoint status available for reconciliation; skipping.",
                    workflow_id=self._workflow_id,
                )
                return

            checkpoint_status_str = WORKFLOW_STATUS_TO_CHECKPOINT_STATUS.get(
                checkpoint_status_value
            )
            if checkpoint_status_str is None:
                self._logger.info(
                    "Checkpoint status has no Rails mapping; skipping reconciliation.",
                    workflow_id=self._workflow_id,
                    checkpoint_status=checkpoint_status_value,
                )
                return

            target_event = CHECKPOINT_STATUS_TO_STATUS_EVENT.get(checkpoint_status_str)
            if target_event is None:
                self._logger.info(
                    "Checkpoint status maps to no Rails status event; skipping reconciliation.",
                    workflow_id=self._workflow_id,
                    checkpoint_status_str=checkpoint_status_str,
                )
                return

            rails_status = await self._status_handler.get_workflow_status(
                workflow_id=self._workflow_id
            )

            if rails_status == checkpoint_status_value:
                self._logger.info(
                    "Rails status already matches checkpoint; no reconciliation needed.",
                    workflow_id=self._workflow_id,
                    status=rails_status,
                )
                return

            self._logger.info(
                "Reconciling Rails status to match checkpoint.",
                workflow_id=self._workflow_id,
                rails_status=rails_status,
                checkpoint_status=checkpoint_status_value,
                target_event=target_event,
            )
            await self._update_workflow_status_safely(target_event)

        except Exception as e:
            log_exception(
                e,
                extra={
                    "workflow_id": self._workflow_id,
                    "context": "Failed to reconcile session status",
                },
            )

    def decode_graphql_checkpoint(
        self, checkpoint: GitLabCheckpoint
    ) -> Optional[CheckpointTuple]:
        """Decode a GraphQL-shaped GitLab checkpoint dict (e.g. from ``WorkflowConfig["latest_checkpoint"]``) into a
        ``CheckpointTuple``.

        Handles both compressed (19.0+) and uncompressed (< 19.0) payloads. Public: intended for
        cross-class callers (e.g. ``Flow._resolve_stop_recovery``) that need to inspect checkpoint
        state without issuing additional HTTP requests, in addition to its internal use on the
        ``aget_tuple`` cached-latest-checkpoint path.

        **Side-effect free**: decoding never touches the incremental-checkpoint write cache
        (``_prev_checkpoint_id`` / ``_prev_channel_values`` / ``_current_thread`` /
        ``_current_thread_started_at``). Hydration of that cache is reserved for the ``aget_tuple``
        fetch paths — the checkpoint LangGraph actually resumes from is the only valid delta
        baseline, and an arbitrary decoded checkpoint (e.g. the pre-rollback tip during
        stop-recovery) must never repoint it. Callers for whom the decoded checkpoint IS the resume
        baseline (the ``aget_tuple`` cached-latest path) must call ``_hydrate_incremental_state``
        explicitly afterwards.
        """
        if "compressedCheckpoint" in checkpoint:
            decoded_checkpoint = uncompress_checkpoint(
                checkpoint["compressedCheckpoint"]
            )
        else:
            decoded_checkpoint = json.loads(
                checkpoint["checkpoint"], object_hook=checkpoint_decoder
            )
        decoded_metadata = json.loads(
            checkpoint["metadata"], object_hook=checkpoint_decoder
        )
        return self._convert_gitlab_checkpoint_to_checkpoint_tuple(
            {
                "thread_ts": checkpoint["threadTs"],
                "parent_ts": checkpoint["parentTs"],
                "checkpoint": decoded_checkpoint,
                "metadata": decoded_metadata,
            }
        )

    def _hydrate_incremental_state(
        self,
        gl_checkpoint: Mapping[str, Any],
        decoded_checkpoint: Mapping[str, Any],
    ) -> None:
        """Restore in-memory incremental-checkpoint state from a fetched checkpoint.

        On gateway restart, ``_current_thread`` / ``_current_thread_started_at`` / ``_prev_channel_values`` /
        ``_prev_checkpoint_id`` reset to their __init__ defaults. Without this, the next aput would either trigger a
        stale-cache rewrite or emit a current_thread that no longer matches the server's view. Accepts both REST
        (snake_case) and GraphQL
        (camelCase) field names. Absent values are tolerated: older Rails versions don't expose current_thread, in
        which case the in-memory default is kept.
        """
        if not self._workflow_config.get("incremental_checkpoints_enabled", False):
            return

        current_thread = gl_checkpoint.get("current_thread")
        if current_thread is None:
            current_thread = gl_checkpoint.get("currentThread")
        if current_thread is not None:
            try:
                self._current_thread = int(current_thread)
            except (TypeError, ValueError):
                self._logger.warning(
                    "Unexpected current_thread value from server; keeping default",
                    current_thread=current_thread,
                )

        started_at = gl_checkpoint.get(
            "current_thread_started_at"
        ) or gl_checkpoint.get("currentThreadStartedAt")
        if started_at is not None:
            # Restore the group's original start time so a post-restart checkpoint doesn't
            # re-pin the marker to a mid-group time and drop the group's earlier blobs.
            self._current_thread_started_at = started_at

        self._prev_checkpoint_id = decoded_checkpoint.get("id")
        self._prev_channel_values = dict(decoded_checkpoint.get("channel_values", {}))

    @override
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        # https://blog.langchain.dev/langgraph-v0-2/
        # thread_ts and parent_ts have been renamed to checkpoint_id and parent_checkpoint_id , respectively
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        # execution path with checkpoint_id present is triggered when LangGraph needs to fetch specific checkpoint
        # (instead of a most recent one), this happens in following situations:
        # graph is being replayed from specific snapshot (see:
        # https://langchain-ai.github.io/langgraph/concepts/time-travel/#replaying)
        # specific checkpoint is being requested in `aget_state`
        # in both case grahp_config looks like: {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz'}}
        if checkpoint_id:
            endpoint = add_compression_param(
                f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
            )
            with duo_workflow_metrics.time_gitlab_response(
                endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints",
                method="GET",
            ):
                response = await self._client.aget(
                    path=endpoint,
                    object_hook=checkpoint_decoder,
                )

                if not response.is_success():
                    self._logger.error(
                        "Failed to fetch checkpoints",
                        workflow_id=self._workflow_id,
                        status_code=response.status_code,
                        response_body=response.body,
                    )

                gl_checkpoints = response.body
            checkpoint = next(
                (c for c in gl_checkpoints if c["thread_ts"] == checkpoint_id), None
            )
            if checkpoint:
                if "compressed_checkpoint" in checkpoint:
                    checkpoint["checkpoint"] = uncompress_checkpoint(
                        checkpoint["compressed_checkpoint"]
                    )
                # else: checkpoint["checkpoint"] already exists from old instance, use as-is
                self._hydrate_incremental_state(checkpoint, checkpoint["checkpoint"])
        else:
            # If the latest checkpoint is fetch, we don't need to refetch it on initialization
            if self._workflow_config.get("latest_checkpoint"):
                checkpoint = self._workflow_config["latest_checkpoint"]
                if checkpoint:
                    checkpoint_tuple = self.decode_graphql_checkpoint(checkpoint)
                    if checkpoint_tuple:
                        # LangGraph resumes from this checkpoint, so it is the
                        # write-side delta baseline: hydrate explicitly here
                        # (decoding itself is side-effect free).
                        self._hydrate_incremental_state(
                            checkpoint, checkpoint_tuple.checkpoint
                        )
                    return checkpoint_tuple

            # If the first checkpoint is None, it means that a flow just started and checkpoints are empty anyway
            if self._workflow_config.get("first_checkpoint") is None:
                return None

            # If a flow is resumed and the latest checkpoint couldn't be fetched (<18.8 version of GitLab), fetch it
            checkpoint = await self._fetch_most_recent_checkpoint()

            if checkpoint:
                self._hydrate_incremental_state(checkpoint, checkpoint["checkpoint"])

        if checkpoint:
            return self._convert_gitlab_checkpoint_to_checkpoint_tuple(checkpoint)
        return None

    @override
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        endpoint = add_compression_param(
            f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
        )
        with duo_workflow_metrics.time_gitlab_response(
            endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints", method="GET"
        ):
            response = await self._client.aget(
                path=endpoint,
                object_hook=checkpoint_decoder,
            )

            if not response.is_success():
                self._logger.error(
                    "Failed to fetch checkpoints for list",
                    workflow_id=self._workflow_id,
                    status_code=response.status_code,
                    response_body=response.body,
                )

            gl_checkpoints = response.body
        for gl_checkpoint in gl_checkpoints:
            try:
                if "compressed_checkpoint" in gl_checkpoint:
                    gl_checkpoint["checkpoint"] = uncompress_checkpoint(
                        gl_checkpoint["compressed_checkpoint"]
                    )
                # else: gl_checkpoint["checkpoint"] already exists from old instance, use as-is

                yield self._convert_gitlab_checkpoint_to_checkpoint_tuple(gl_checkpoint)
            except ValueError as e:
                log_exception(e, extra={"context": "Skipping malformed checkpoint"})
                continue

    async def _iter_checkpoint_pages(
        self,
        *,
        per_page: int = 20,
        # `List`, not `list`: inside the class body the bare name `list` resolves
        # to the `list()` checkpoint-saver method defined above, not the builtin.
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """Fetch raw checkpoint pages from Rails, newest-first, one HTTP request per page.

        Stops requesting further pages as soon as the caller stops iterating (e.g. once a match is found downstream) —
        this is the mechanism that makes the backwards walk cheap even for long checkpoint chains. Uses the same
        ``checkpoints`` REST endpoint as ``alist()`` / ``_fetch_most_recent_checkpoint``, following the ``X-Next-Page``
        pagination convention already used in ``duo_workflow_service/tools/duo_base_tool.py``.
        """
        page: str = "1"
        while page:
            endpoint = add_compression_param(
                f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
                f"?per_page={per_page}&page={page}"
            )
            with duo_workflow_metrics.time_gitlab_response(
                endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints",
                method="GET",
            ):
                response = await self._client.aget(
                    path=endpoint, object_hook=checkpoint_decoder
                )

            if not response.is_success():
                self._logger.error(
                    "Failed to fetch checkpoint page",
                    workflow_id=self._workflow_id,
                    status_code=response.status_code,
                    page=page,
                )
                return
            if not response.body:
                return

            yield response.body
            page = response.headers.get("X-Next-Page", "")

    async def checkpoints_reversed(
        self,
        *,
        matches: Callable[[Mapping[str, Any]], bool] = lambda _: True,
        per_page: int = 20,
    ) -> AsyncIterator[CheckpointTuple]:
        """Yield checkpoints newest-first, optionally filtered, fetched page-by-page.

        Public data-access primitive intentionally free of any "recovery" or "boundary" domain knowledge.
        ``matches`` is applied to each checkpoint's ``channel_values`` purely as a predicate; what "matching"
        means is entirely the caller's concern. Because fetching is paginated (see ``_iter_checkpoint_pages``),
        a caller that stops iterating after the first match never requests pages beyond the one containing the match.

        Args:
            matches: Predicate applied to each checkpoint's ``channel_values``; only checkpoints for which
                this returns ``True`` are yielded. Defaults to accepting all checkpoints.
            per_page: Number of checkpoints to request per HTTP page. Defaults to 20.

        Yields:
            CheckpointTuple: Decompressed, converted checkpoints in newest-first order.
        """
        async for gl_checkpoints in self._iter_checkpoint_pages(per_page=per_page):
            for gl_checkpoint in gl_checkpoints:
                try:
                    if "compressed_checkpoint" in gl_checkpoint:
                        gl_checkpoint["checkpoint"] = uncompress_checkpoint(
                            gl_checkpoint["compressed_checkpoint"]
                        )
                    checkpoint_tuple = (
                        self._convert_gitlab_checkpoint_to_checkpoint_tuple(
                            gl_checkpoint
                        )
                    )
                except ValueError as e:
                    log_exception(e, extra={"context": "Skipping malformed checkpoint"})
                    continue
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                if matches(channel_values):
                    yield checkpoint_tuple

    @override
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        # new_versions is an optional parameter
        # it carry information onto which channels was updated
        # it can be used to normalise checkpoints storage in db
        # and save storage space
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config.get("configurable", {})

        if not self._orbit_called:
            self._orbit_called = _get_orbit_tool_calls(checkpoint)

        incremental_enabled = self._workflow_config.get(
            "incremental_checkpoints_enabled", False
        )
        checkpoint_strategy = "incremental" if incremental_enabled else "full"

        # https://blog.langchain.dev/langgraph-v0-2/
        # thread_ts and parent_ts have been renamed to checkpoint_id and parent_checkpoint_id , respectively
        # checkpoint_strategy is a query param (not read by Rails) so it appears in
        # request logs and is searchable in Kibana for strategy monitoring.
        endpoint = (
            f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
            f"?checkpoint_strategy={checkpoint_strategy}"
        )

        payload: Dict[str, Any] = {
            "thread_ts": checkpoint["id"],
            "parent_ts": configurable.get("checkpoint_id"),
            "metadata": metadata,
            "compressed_checkpoint": compress_checkpoint(checkpoint),
        }

        if incremental_enabled:
            parent_checkpoint_id = configurable.get("checkpoint_id")
            stale_cache = (
                self._prev_checkpoint_id is not None
                and parent_checkpoint_id != self._prev_checkpoint_id
            )
            if stale_cache:
                self._logger.warning(
                    "Stale incremental checkpoint cache detected; resetting to full values",
                    expected_prev_checkpoint_id=parent_checkpoint_id,
                    cached_prev_checkpoint_id=self._prev_checkpoint_id,
                )
                self._prev_channel_values = {}

            channel_blobs, is_compaction = _serialize_channel_blobs(
                checkpoint,
                new_versions,
                self._prev_channel_values,
                force_rewrite=stale_cache,
            )
            # A new current_thread group starts on the first checkpoint and whenever
            # current_thread is bumped; its start time bounds the created_at range of a
            # later blob query. Carried forward (and restored on restart) otherwise so the
            # marker stays pinned to the group's first checkpoint.
            starts_new_thread = (
                self._current_thread_started_at is None or stale_cache or is_compaction
            )
            # A genuine group boundary: the workflow's first checkpoint, a stale-cache
            # reset, or a compaction. Keyed on _prev_checkpoint_id (not the started_at
            # marker, which stays None when a checkpoint id isn't time-based) so later
            # checkpoints in a group are never re-seeded.
            is_group_start = (
                self._prev_checkpoint_id is None or stale_cache or is_compaction
            )
            if is_group_start:
                # A group must be self-contained (issue 605653): re-seed EVERY channel
                # as a full snapshot so this group reconstructs without a prior group or
                # the checkpoint header. Otherwise keep the per-channel deltas above.
                channel_blobs = _serialize_all_channels_full(checkpoint)
            if stale_cache or is_compaction:
                self._current_thread += 1
            if starts_new_thread:
                self._current_thread_started_at = _thread_started_at_from_id(
                    checkpoint["id"]
                )
            self._prev_channel_values = dict(checkpoint.get("channel_values", {}))
            self._prev_checkpoint_id = checkpoint["id"]
            payload["channel_blobs"] = channel_blobs
            payload["current_thread"] = self._current_thread
            if self._current_thread_started_at is not None:
                payload["current_thread_started_at"] = self._current_thread_started_at
            self._logger.info(
                "Incremental checkpoint sizes",
                thread_ts=checkpoint["id"],
                current_thread=self._current_thread,
                current_thread_started_at=self._current_thread_started_at,
                compressed_checkpoint_bytes=len(payload["compressed_checkpoint"]),
                channel_blobs_total_bytes=sum(len(b["data"]) for b in channel_blobs),
                channel_blob_count=len(channel_blobs),
            )

        if (model_metadata := current_model_metadata_context.get()) is not None:
            # Exclude `api_key`: this JSON is persisted by GitLab Rails and later
            # echoed back verbatim as a gRPC header for provider-stickiness replay
            # (see ModelMetadataInterceptor), so it must never carry a live secret.
            payload["model_metadata_json"] = model_metadata.model_dump_json(
                exclude={"llm_definition", "friendly_name", "api_key"}
            )

        with duo_workflow_metrics.time_gitlab_response(
            endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints", method="POST"
        ):
            response = await self._client.apost(
                path=endpoint,
                body=json.dumps(payload, cls=CustomEncoder),
            )
            duo_workflow_metrics.count_checkpoints(
                endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoints",
                status_code=(
                    response.status_code
                    if isinstance(response, GitLabHttpResponse)
                    else "unknown"
                ),
                method="POST",
            )
        self._logger.info(
            "Checkpoint saved",
            thread_ts=checkpoint["id"],
            parent_ts=configurable.get("checkpoint_id"),
            checkpoint_strategy=checkpoint_strategy,
        )

        return {
            "configurable": {
                "thread_id": self._workflow_id,
                "checkpoint_id": checkpoint["id"],
            }
        }

    @override
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
        # We are ignoring this parameter for now since we don't care for the order the pending writes are fetched in
    ) -> None:
        status = self._get_workflow_status_event(writes)
        if status:
            self._logger.debug(
                f"Updating workflow status from checkpoints, with status {status.value}"
            )
            await self._update_workflow_status(status)

        configurable = config.get("configurable", {})
        checkpoint_id = configurable.get("checkpoint_id")
        workflow_id = configurable.get("thread_id")

        # for now only interrupts are stored
        if not writes or writes[0][0] != "__interrupt__":
            return None

        encoded_writes = []
        for idx, (channel, val) in enumerate(writes):
            t, bval = self.serde.dumps_typed(val)
            encoded_writes.append(
                {
                    "task": task_id,
                    "channel": channel,
                    "data": base64.b64encode(bval).decode("utf-8"),
                    "write_type": t,
                    "idx": idx,
                }
            )

        endpoint = (
            f"/api/v4/ai/duo_workflows/workflows/{workflow_id}/checkpoint_writes_batch"
        )
        with duo_workflow_metrics.time_gitlab_response(
            endpoint="/api/v4/ai/duo_workflows/workflows/:id/checkpoint_writes_batch",
            method="POST",
        ):
            result = await self._client.apost(
                path=endpoint,
                body=json.dumps(
                    {
                        "thread_ts": checkpoint_id,
                        "checkpoint_writes": encoded_writes,
                    }
                ),
            )
        self._logger.info(
            "Checkpoint updated with pending writes",
            thread_ts=checkpoint_id,
            parent_ts=configurable.get("checkpoint_id"),
            result=result,
        )
        return None

    def _convert_gitlab_checkpoint_to_checkpoint_tuple(
        self,
        gl_checkpoint: Dict[str, Any],
    ) -> CheckpointTuple:

        pending_writes = None
        if "checkpoint_writes" in gl_checkpoint:
            pending_writes = [
                (
                    w["task"],
                    w["channel"],
                    self.serde.loads_typed(
                        (w["write_type"], base64.b64decode(w["data"]))
                    ),
                )
                for w in gl_checkpoint["checkpoint_writes"]
            ]

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": self._workflow_id,
                    "checkpoint_id": gl_checkpoint["thread_ts"],
                }
            },
            checkpoint=gl_checkpoint["checkpoint"],
            metadata=gl_checkpoint["metadata"],
            pending_writes=pending_writes,
            parent_config={
                "configurable": {
                    "thread_id": self._workflow_id,
                    "checkpoint_id": gl_checkpoint["parent_ts"],
                }
            },
        )

    def _get_workflow_status_event(
        self,
        writes: Sequence[Tuple[str, Any]],
    ) -> Optional[WorkflowStatusEventEnum]:
        """Status events are accepted by GitLab Rails API to change a workflow status from one to another, using a state
        machine.

        For example, `drop` status event changes a workflow status from
        `created`, `running` or `paused` to `failed`.
        For workflow status `not started`, there is no status event
        Resume, Retry and start workflow events are handled with  self._get_initial_status_event method
        """
        status_event = None
        is_changed, value = _attribute_dirty("status", writes)
        if is_changed:
            if value is None or value in NOOP_WORKFLOW_STATUSES:
                return status_event
            checkpoint_status = WORKFLOW_STATUS_TO_CHECKPOINT_STATUS.get(value)
            if checkpoint_status:
                status_event = CHECKPOINT_STATUS_TO_STATUS_EVENT.get(checkpoint_status)

        return status_event

    def log_if_finished_due_to_stopped(self, status: str):
        if status == "stopped":
            self._logger.warning("Marking session successful because it's stopped")
