import time
from contextvars import ContextVar
from enum import StrEnum
from typing import Optional

import structlog
from prometheus_client import REGISTRY, Counter, Histogram

from ai_gateway.instrumentators.model_requests import (
    METADATA_LABELS,
    LLMFinishReason,
    build_metadata_labels,
)

session_type_context: ContextVar[Optional[str]] = ContextVar(
    "session_type", default="unknown"
)

log = structlog.stdlib.get_logger("monitoring")

WORKFLOW_TIME_SCALE_BUCKETS = [
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    120,
    300,
    600,
    1200,
    1800,
    3600,
]
LLM_TIME_SCALE_BUCKETS = [0.25, 0.5, 1, 2, 4, 7, 10, 20, 30, 60]

LLM_FINISH_REASONS = LLMFinishReason.values()


class SessionTypeEnum(StrEnum):
    START = "start"
    RESUME = "resume"
    RETRY = "retry"


class DuoWorkflowMetrics:  # pylint: disable=too-many-instance-attributes
    def __init__(self, registry=REGISTRY):
        self.workflow_duration = Histogram(
            "duo_workflow_total_seconds",
            "Total duration of Duo Workflow processing",
            ["workflow_type"],
            registry=registry,
            buckets=WORKFLOW_TIME_SCALE_BUCKETS,
        )

        self.tool_call_duration = Histogram(
            "duo_workflow_tool_call_seconds",
            "Duration of tool calls in Duo Workflow",
            ["tool_name", "flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.compute_duration = Histogram(
            "duo_workflow_compute_seconds",
            "Duration of computation in Duo Workflow service",
            ["operation_type"],
            registry=registry,
        )

        self.gitlab_response_duration = Histogram(
            "duo_workflow_gitlab_response_seconds",
            "Duration of GitLab instance responses",
            ["endpoint", "method"],
            registry=registry,
        )

        self.network_latency = Histogram(
            "duo_workflow_network_latency_seconds",
            "Network latency between Duo Workflow and other services",
            ["source", "destination"],
            registry=registry,
        )

        self.checkpoint_counter = Counter(
            "duo_workflow_checkpoint_total",
            "Count of checkpoint calls in Duo Workflow",
            ["endpoint", "status_code", "method"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_start_counter = Counter(
            "agent_platform_session_start_total",
            "Count of flow start events in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_retry_counter = Counter(
            "agent_platform_session_retry_total",
            "Count of flow retry events in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_reject_counter = Counter(
            "agent_platform_session_reject_total",
            "Count of flow reject events in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_resume_counter = Counter(
            "agent_platform_session_resume_total",
            "Count of flow resume events in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_success_counter = Counter(
            "agent_platform_session_success_total",
            "Count of successful flow completions in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_failure_counter = Counter(
            "agent_platform_session_failure_total",
            "Count of failed flows in Duo Workflow",
            ["flow_type", "failure_reason", "session_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_tool_failure_counter = Counter(
            "agent_platform_tool_failure_total",
            "Count of failed tools in Duo Workflow",
            ["flow_type", "tool_name", "failure_reason"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_receive_start_counter = Counter(
            "agent_platform_receive_start_total",
            "Count of receive start events in Duo Workflow",
            ["flow_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.agent_platform_session_abort_counter = Counter(
            "agent_platform_session_abort_total",
            "Count of aborted sessions in Duo Agent Platform",
            ["flow_type", "session_type"] + METADATA_LABELS,
            registry=registry,
        )

        self.time_to_first_token = Histogram(
            "duo_workflow_time_to_first_token_seconds",
            "Time from ExecuteWorkflow call to first outgoing action",
            ["workflow_type"] + METADATA_LABELS,
            registry=registry,
            buckets=LLM_TIME_SCALE_BUCKETS,
        )

    def count_checkpoints(
        self,
        endpoint="unknown",
        status_code="unknown",
        method="unknown",
    ):
        self.checkpoint_counter.labels(
            endpoint=endpoint,
            status_code=status_code,
            method=method,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_start(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_start_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_retry(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_retry_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_reject(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_reject_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_resume(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_resume_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_success(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_success_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_failure(
        self,
        flow_type: str = "unknown",
        failure_reason: str = "unknown",
    ) -> None:
        self.agent_platform_session_failure_counter.labels(
            flow_type=flow_type,
            failure_reason=failure_reason,
            session_type=session_type_context.get(),
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_session_abort(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_session_abort_counter.labels(
            flow_type=flow_type,
            session_type=session_type_context.get(),
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_tool_failure(
        self,
        flow_type: str = "unknown",
        tool_name: str = "unknown",
        failure_reason: str = "unknown",
    ) -> None:
        self.agent_platform_tool_failure_counter.labels(
            flow_type=flow_type,
            tool_name=tool_name,
            failure_reason=failure_reason,
            **build_metadata_labels(),
        ).inc()

    def count_agent_platform_receive_start_counter(
        self,
        flow_type: str = "unknown",
    ) -> None:
        self.agent_platform_receive_start_counter.labels(
            flow_type=flow_type,
            **build_metadata_labels(),
        ).inc()

    def time_tool_call(self, tool_name="unknown", flow_type="unknown"):
        return self._timer(
            lambda duration: self.tool_call_duration.labels(
                tool_name=tool_name,
                flow_type=flow_type,
                **build_metadata_labels(),
            ).observe(duration)
        )

    def time_compute(self, operation_type="unknown"):
        return self._timer(
            lambda duration: self.compute_duration.labels(
                operation_type=operation_type
            ).observe(duration)
        )

    def time_gitlab_response(self, endpoint="unknown", method="unknown"):
        return self._timer(
            lambda duration: self.gitlab_response_duration.labels(
                endpoint=endpoint, method=method
            ).observe(duration)
        )

    def time_network_latency(self, source="unknown", destination="unknown"):
        return self._timer(
            lambda duration: self.network_latency.labels(
                source=source, destination=destination
            ).observe(duration)
        )

    def time_workflow(self, workflow_type="unknown"):
        return self._timer(
            lambda duration: self.workflow_duration.labels(
                workflow_type=workflow_type
            ).observe(duration)
        )

    def record_time_to_first_token(
        self, duration: float, workflow_type: str = "unknown"
    ) -> None:
        self.time_to_first_token.labels(
            workflow_type=workflow_type,
            **build_metadata_labels(),
        ).observe(duration)

    class _timer:
        def __init__(self, callback):
            self.callback = callback
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None:
                duration = time.time() - self.start_time
                self.callback(duration)
            else:
                log.warning("Timer was not started")
                self.callback(0.0)
