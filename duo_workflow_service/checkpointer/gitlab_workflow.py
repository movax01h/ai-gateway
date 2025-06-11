import base64
import functools
import json
import os
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    STATUS_TO_EVENT_PROPERTY,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, checkpoint_decoder
from duo_workflow_service.internal_events import (
    DuoWorkflowInternalEvent,
    InternalEventAdditionalProperties,
)
from duo_workflow_service.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)
from duo_workflow_service.json_encoder.encoder import CustomEncoder
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.status_updater.gitlab_status_updater import (
    GitLabStatusUpdater,
)
from duo_workflow_service.tracking.errors import log_exception

T = TypeVar("T", bound=callable)  # type: ignore


def not_implemented_sync_method(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            "The GitLabSaver does not support synchronous methods. "
        )

    return wrapper  # type: ignore


NOOP_WORKFLOW_STATUSES = [WorkflowStatusEnum.APPROVAL_ERROR]


class WorkflowStatusEventEnum(StrEnum):
    START = "start"
    FINISH = "finish"
    DROP = "drop"
    RESUME = "resume"
    PAUSE = "pause"
    STOP = "stop"
    RETRY = "retry"
    REQUIRE_INPUT = "require_input"
    REQUIRE_PLAN_APPROVAL = "require_plan_approval"
    REQUIRE_TOOL_CALL_APPROVAL = "require_tool_call_approval"


# Maps current checkpoint status to the rails workflow state machine's event (if applicable)
CheckpointStatusToStatusEvent = {
    "FINISHED": WorkflowStatusEventEnum.FINISH,
    "FAILED": WorkflowStatusEventEnum.DROP,
    "STOPPED": WorkflowStatusEventEnum.STOP,
    "PAUSED": WorkflowStatusEventEnum.PAUSE,
    "INPUT_REQUIRED": WorkflowStatusEventEnum.REQUIRE_INPUT,
    "PLAN_APPROVAL_REQUIRED": WorkflowStatusEventEnum.REQUIRE_PLAN_APPROVAL,
    "TOOL_CALL_APPROVAL_REQUIRED": WorkflowStatusEventEnum.REQUIRE_TOOL_CALL_APPROVAL,
}

# Maps WorkflowStatus(status key in LangGraph's WorkflowState) to checkpoint status.
# Checkpoint status represents status human-readable workflow status (displayed in the UI)
WORKFLOW_STATUS_TO_CHECKPOINT_STATUS = {
    **{
        WorkflowStatusEnum.EXECUTION: "RUNNING",
        WorkflowStatusEnum.ERROR: "FAILED",
        WorkflowStatusEnum.INPUT_REQUIRED: "INPUT_REQUIRED",
        WorkflowStatusEnum.PLANNING: "RUNNING",
        WorkflowStatusEnum.PAUSED: "PAUSED",
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED: "PLAN_APPROVAL_REQUIRED",
        WorkflowStatusEnum.NOT_STARTED: "CREATED",
        WorkflowStatusEnum.COMPLETED: "FINISHED",
        WorkflowStatusEnum.CANCELLED: "STOPPED",
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED: "TOOL_CALL_APPROVAL_REQUIRED",
    },
    **{status: "RUNNING" for status in NOOP_WORKFLOW_STATUSES},
}


def _attribute_dirty(attribute: str, metadata: CheckpointMetadata) -> bool:
    writes = metadata.get("writes")
    if not writes:
        return False

    return next(
        (True for node_writes in writes.values() if attribute in node_writes), False
    )


class GitLabWorkflow(BaseCheckpointSaver[Any], AbstractAsyncContextManager[Any]):
    _client: GitlabHttpClient
    _logger: structlog.stdlib.BoundLogger

    def __init__(
        self,
        client: GitlabHttpClient,
        workflow_id: str,
        workflow_type: CategoryEnum,
    ):
        self._offline_mode = os.getenv("USE_MEMSAVER")
        self._client = client
        self._workflow_id = workflow_id
        self._status_handler = GitLabStatusUpdater(client)
        self._logger = structlog.stdlib.get_logger("workflow_checkpointer")
        self._workflow_type = workflow_type

    @not_implemented_sync_method
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return None

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

    @not_implemented_sync_method
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return RunnableConfig()

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

        self._logger.info("Tracking Internal event %s", event_name.value)
        DuoWorkflowInternalEvent.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._workflow_type.value,
        )

    async def __aenter__(self) -> BaseCheckpointSaver:
        try:
            if self._offline_mode:
                return MemorySaver()

            config: RunnableConfig = {"configurable": {}}
            self.initial_status_event, event_property = await self._status_event(config)
            await self._update_workflow_status(self.initial_status_event)

            if self.initial_status_event == WorkflowStatusEventEnum.START:
                label = EventLabelEnum.WORKFLOW_START_LABEL
                event_name = EventEnum.WORKFLOW_START
            elif self.initial_status_event == WorkflowStatusEventEnum.RETRY:
                label = EventLabelEnum.WORKFLOW_RESUME_LABEL
                event_name = EventEnum.WORKFLOW_RETRY
            elif self.initial_status_event == WorkflowStatusEventEnum.RESUME:
                label = EventLabelEnum.WORKFLOW_RESUME_LABEL
                event_name = EventEnum.WORKFLOW_RESUME
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
            return self
        except Exception as e:
            failure_properties = InternalEventAdditionalProperties(
                label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
                property=repr(e),
                value=self._workflow_id,
            )
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

    async def _status_event(
        self, config: RunnableConfig
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
        if not await self.aget_tuple(config):
            return WorkflowStatusEventEnum.START, EventPropertyEnum.WORKFLOW_ID

        status = await self._status_handler.get_workflow_status(self._workflow_id)
        if status in [
            WorkflowStatusEnum.INPUT_REQUIRED,
            WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
            WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        ]:
            return WorkflowStatusEventEnum.RESUME, STATUS_TO_EVENT_PROPERTY.get(
                status, EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN
            )
        else:
            # existing workflows which were not interrupted are retried
            return (
                WorkflowStatusEventEnum.RETRY,
                EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
            )

    async def __aexit__(self, exc_type, exc_value, trcback):
        """Handle workflow completion and tracking in both success and failure scenarios.

        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        # In case of exception in async context manager,
        # update status to DROP, track failure event,
        # and return False
        if exc_type:
            await self._handle_workflow_exception(exc_value)
            await self._update_workflow_status_safely()
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

    async def _handle_workflow_exception(self, exc_value: Any) -> None:
        """Track workflow failure event."""
        properties = InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_FINISH_LABEL.value,
            property=repr(exc_value),
            value=self._workflow_id,
        )
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_FINISH_FAILURE,
            additional_properties=properties,
        )

    async def _track_workflow_completion(self, status: str) -> None:
        """Track successful workflow completion based on status."""

        if status == WorkflowStatusEnum.INPUT_REQUIRED:
            event = EventEnum.WORKFLOW_PAUSE
            label = EventLabelEnum.WORKFLOW_PAUSE_LABEL.value
            prop = EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN_AWAIT_INPUT.value
        elif status == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED:
            event = EventEnum.WORKFLOW_PAUSE
            label = EventLabelEnum.WORKFLOW_PAUSE_LABEL.value
            prop = EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN_AWAIT_APPROVAL.value
        elif status in ("finished", "stopped"):
            event = EventEnum.WORKFLOW_FINISH_SUCCESS
            label = EventLabelEnum.WORKFLOW_FINISH_LABEL.value
            prop = STATUS_TO_EVENT_PROPERTY.get(
                status, EventPropertyEnum.WORKFLOW_ID
            ).value
        else:
            # No event to track for other statuses
            return

        self._track_internal_event(
            event,
            InternalEventAdditionalProperties(
                label=label, property=prop, value=self._workflow_id
            ),
        )

    async def _update_workflow_status_safely(self):
        """Attempt to update workflow status to DROP, handling any exceptions.

        Returns:
            bool: False to indicate non-successful completion
        """
        try:
            await self._update_workflow_status(WorkflowStatusEventEnum.DROP)
        except Exception as e:
            log_exception(e, extra={"workflow_id": self._workflow_id})
        return False

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
            endpoint = (
                f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
            )
            with duo_workflow_metrics.time_gitlab_response(
                endpoint=endpoint, method="GET"
            ):
                gl_checkpoints = await self._client.aget(
                    path=endpoint,
                    object_hook=checkpoint_decoder,
                )
            checkpoint = next(
                (c for c in gl_checkpoints if c["thread_ts"] == checkpoint_id), None
            )
        else:
            endpoint = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints?per_page=1"
            with duo_workflow_metrics.time_gitlab_response(
                endpoint=endpoint, method="GET"
            ):
                gl_checkpoints = await self._client.aget(
                    path=endpoint,
                    object_hook=checkpoint_decoder,
                )
            if (
                isinstance(gl_checkpoints, dict)
                and "status" in gl_checkpoints
                and gl_checkpoints["status"] != 200
            ):
                raise Exception(f"Failed to fetch checkpoints: {gl_checkpoints}")
            checkpoint = gl_checkpoints[0] if gl_checkpoints else None

        if checkpoint:
            return self._convert_gitlab_checkpoint_to_checkpoint_tuple(checkpoint)
        return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        endpoint = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
        with duo_workflow_metrics.time_gitlab_response(endpoint=endpoint, method="GET"):
            gl_checkpoints = await self._client.aget(
                path=endpoint,
                object_hook=checkpoint_decoder,
            )
        for gl_checkpoint in gl_checkpoints:
            try:
                yield self._convert_gitlab_checkpoint_to_checkpoint_tuple(gl_checkpoint)
            except ValueError as e:
                self._logger.info(f"Skipping malformed checkpoint: {e}")
                continue

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

        status = self._get_workflow_status_event(checkpoint, metadata)
        if status:
            self._logger.debug(
                f"Updating workflow status from checkpoints, with status {status.value}"
            )
            await self._update_workflow_status(status)

        # https://blog.langchain.dev/langgraph-v0-2/
        # thread_ts and parent_ts have been renamed to checkpoint_id and parent_checkpoint_id , respectively
        endpoint = f"/api/v4/ai/duo_workflows/workflows/{self._workflow_id}/checkpoints"
        with duo_workflow_metrics.time_gitlab_response(
            endpoint=endpoint, method="POST"
        ):
            await self._client.apost(
                path=endpoint,
                body=json.dumps(
                    {
                        "thread_ts": checkpoint["id"],
                        "parent_ts": configurable.get("checkpoint_id"),
                        "checkpoint": checkpoint,
                        "metadata": metadata,
                    },
                    cls=CustomEncoder,
                ),
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

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
        # We are ignoring this parameter for now since we don't care for the order the pending writes are fetched in
    ) -> None:
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
            endpoint=endpoint, method="POST"
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
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Optional[WorkflowStatusEventEnum]:
        """Status events are accepted by GitLab Rails API to change a workflow status from one to another, using a state
        machine.

        For example, `drop` status event changes a workflow status from
        `created`, `running` or `paused` to `failed`.
        For workflow status `not started`, there is no status event
        Resume, Retry and start workflow events are handled with  self._status_event method
        """
        status_event = None
        if _attribute_dirty("status", metadata):
            workflow_status = checkpoint["channel_values"].get("status")
            if workflow_status is None or workflow_status in NOOP_WORKFLOW_STATUSES:
                return status_event
            checkpoint_status = WORKFLOW_STATUS_TO_CHECKPOINT_STATUS.get(
                workflow_status
            )
            if checkpoint_status:
                status_event = CheckpointStatusToStatusEvent.get(checkpoint_status)

        return status_event
