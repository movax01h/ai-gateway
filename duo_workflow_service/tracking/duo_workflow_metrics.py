import time

import structlog
from prometheus_client import REGISTRY, Counter, Histogram

log = structlog.stdlib.get_logger("monitoring")


class DuoWorkflowMetrics:
    def __init__(self, registry=REGISTRY):
        self.workflow_duration = Histogram(
            "duo_workflow_total_seconds",
            "Total duration of Duo Workflow processing",
            ["workflow_type"],
            registry=registry,
        )

        self.llm_request_duration = Histogram(
            "duo_workflow_llm_request_seconds",
            "Duration of LLM requests in Duo Workflow",
            ["model", "request_type"],
            registry=registry,
        )

        self.tool_call_duration = Histogram(
            "duo_workflow_tool_call_seconds",
            "Duration of tool calls in Duo Workflow",
            ["tool_name"],
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

        self.llm_response_counter = Counter(
            "duo_workflow_llm_response_total",
            "Response count of LLM calls in Duo Workflow with stop reason",
            ["model", "request_type", "stop_reason"],
            registry=registry,
        )

    def count_llm_response(
        self, model="unknown", request_type="unknown", stop_reason="unknown"
    ):
        # Documented stop reasons for Anthropic models
        anthropic_stop_reasons = [
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "unknown",
        ]
        self.llm_response_counter.labels(
            model=model,
            request_type=request_type,
            stop_reason=(
                stop_reason if stop_reason in anthropic_stop_reasons else "other"
            ),
        ).inc()

    def time_llm_request(self, model="unknown", request_type="unknown"):
        return self._timer(
            lambda duration: self.llm_request_duration.labels(
                model=model, request_type=request_type
            ).observe(duration)
        )

    def time_tool_call(self, tool_name="unknown"):
        return self._timer(
            lambda duration: self.tool_call_duration.labels(
                tool_name=tool_name
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
