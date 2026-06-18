# pylint: disable=super-init-not-called,direct-environment-variable-reference,broad-exception-raised,attribute-defined-outside-init,too-many-lines

import asyncio
import base64
import functools
import json
import os
import time
import zlib
from contextlib import AbstractAsyncContextManager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    override,
)

import structlog
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
from langgraph.checkpoint.serde.base import SerializerProtocol

from ai_gateway.container import ContainerApplication
from duo_workflow_service.audit_events.context import get_audit_collector
from duo_workflow_service.audit_events.event_types import SessionEndedEvent
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    BILLABLE_STATUSES,
    CHECKPOINT_STATUS_TO_STATUS_EVENT,
    NOOP_WORKFLOW_STATUSES,
    STATUS_TO_EVENT_PROPERTY,
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
    WorkflowStatusEventEnum,
    add_compression_param,
    compress_checkpoint,
    uncompress_checkpoint,
)
from duo_workflow_service.checkpointer.utils.serializer import CheckpointSerializer
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.errors.typing import NotifiableException
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


def _serialize_channel_blobs(
    checkpoint: Checkpoint,
    new_versions: ChannelVersions,
    serde: SerializerProtocol,
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
        # Scalar channels (status, goal, project, etc.) are always full
        # replacements and are tiny — incremental savings come entirely from
        # append-heavy list/dict-of-list channels like conversation_history.
        # Scalars are always recoverable from compressed_checkpoint, so
        # excluding them keeps channel_blobs small without information loss.
        if not isinstance(val, (list, dict)):
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

        t, bval = serde.dumps_typed(val)
        blobs.append(
            {
                "channel": channel,
                "version": str(version),
                "data": base64.b64encode(zlib.compress(bval)).decode("utf-8"),
                "write_type": t,
                "step_action": step_action,
            }
        )

    return blobs, is_compaction


class GitLabWorkflow(
    BaseCheckpointSaver[Any], AbstractAsyncContextManager[Any]
):  # pylint: disable=too-many-instance-attributes
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

        await self._status_handler.update_workflow_status(self._workflow_id, status)

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
            self.initial_status_event, event_property = (
                await self._get_initial_status_event(config)
            )
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
        self, config: RunnableConfig  # pylint: disable=unused-argument
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

    def _decode_graphql_latest_checkpoint(
        self, checkpoint: GitLabCheckpoint
    ) -> Optional[CheckpointTuple]:
        """Convert a latestCheckpoint GraphQL response to a CheckpointTuple.

        Handles both compressed (19.0+) and uncompressed (< 19.0) payloads.
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
        else:
            # If the latest checkpoint is fetch, we don't need to refetch it on initialization
            if self._workflow_config.get("latest_checkpoint"):
                checkpoint = self._workflow_config["latest_checkpoint"]
                if checkpoint:
                    return self._decode_graphql_latest_checkpoint(checkpoint)

            # If the first checkpoint is None, it means that a flow just started and checkpoints are empty anyway
            if self._workflow_config.get("first_checkpoint") is None:
                return None

            # If a flow is resumed and the latest checkpoint couldn't be fetched (<18.8 version of GitLab), fetch it
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

            checkpoint = gl_checkpoints[0] if gl_checkpoints else None

            if checkpoint:
                if "compressed_checkpoint" in checkpoint:
                    checkpoint["checkpoint"] = uncompress_checkpoint(
                        checkpoint["compressed_checkpoint"]
                    )
                # else: checkpoint["checkpoint"] already exists from old instance, use as-is

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

        # https://blog.langchain.dev/langgraph-v0-2/
        # thread_ts and parent_ts have been renamed to checkpoint_id and parent_checkpoint_id , respectively
        endpoint = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"

        payload: Dict[str, Any] = {
            "thread_ts": checkpoint["id"],
            "parent_ts": configurable.get("checkpoint_id"),
            "metadata": metadata,
            "compressed_checkpoint": compress_checkpoint(checkpoint),
        }

        if is_client_capable("incremental_checkpoints"):
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
                self.serde,
                self._prev_channel_values,
                force_rewrite=stale_cache,
            )
            if stale_cache or is_compaction:
                self._current_thread += 1
            self._prev_channel_values = dict(checkpoint.get("channel_values", {}))
            self._prev_checkpoint_id = checkpoint["id"]
            payload["channel_blobs"] = channel_blobs
            payload["current_thread"] = self._current_thread
            self._logger.info(
                "Incremental checkpoint sizes",
                thread_ts=checkpoint["id"],
                current_thread=self._current_thread,
                compressed_checkpoint_bytes=len(payload["compressed_checkpoint"]),
                channel_blobs_total_bytes=sum(len(b["data"]) for b in channel_blobs),
                channel_blob_count=len(channel_blobs),
            )

        if (model_metadata := current_model_metadata_context.get()) is not None:
            payload["model_metadata_json"] = model_metadata.model_dump_json(
                exclude={"llm_definition", "friendly_name"}
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
