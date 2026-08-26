# Duo Workflow Service monitoring

The Duo Workflow Service exposes the following Prometheus metrics.

For logs and traces, see [Logging](../duo_workflow_service/README.md#logging) in the Duo Workflow Service README.
For dashboards and alerts, see the [Duo Workflow Service runbook](https://runbooks.gitlab.com/duo-workflow-svc/).

## Enabling the metrics endpoint

`setup_monitoring` in [`duo_workflow_service/monitoring.py`](../duo_workflow_service/monitoring.py) starts a separate
HTTP server that serves metrics; `run` calls it during startup.
The server only starts when **both** of these environment variables are set:

| Variable | Example | Description |
|---|---|---|
| `PROMETHEUS_METRICS__ADDR` | `0.0.0.0` | Address the metrics HTTP server binds to. |
| `PROMETHEUS_METRICS__PORT` | `8083` locally, `8082` when deployed | Port the metrics HTTP server listens on. |

Both are set in `example.env`, so a local service started with `poetry run duo-workflow-service` exposes metrics at
`http://localhost:8083/metrics`:

```shell
curl -s localhost:8083/metrics | grep duo_workflow
```

Staging and production use port `8082`, set in `.runway/duo-workflow-svc/env-{staging,production}.yml` and
`.runway/duo-workflow-svc-eks/default-values.yaml`.

If either variable is missing, the service logs `Metrics are disabled...` at debug level and opens no endpoint.
The metric objects still exist and accumulate values; a scraper simply never reads them.

The Duo Workflow Service registers metrics against the default `prometheus_client` registry, which it shares with the
AI Gateway instrumentation it loads through the DI container. A scrape of this endpoint therefore also returns
AI Gateway metrics such as `model_inferences_total` and `inference_request_duration_seconds`.

## Common metadata labels

Most metrics carry a shared set of request-metadata labels, referred to as `<metadata>` in the tables below.
`METADATA_LABELS` in [`lib/context/request_metadata.py`](../lib/context/request_metadata.py) defines them, and
`build_metadata_labels` populates them from the request context:

| Label | Value |
|---|---|
| `lsp_version` | Language server version from the request, or `unknown`. |
| `gitlab_version` | GitLab instance version, or `unknown` if unparsable. |
| `client_type` | Client that initiated the request, or `unknown`. |
| `gitlab_realm` | `saas` or `self-managed`, or `unknown`. |
| `is_gitlab_team_member` | `yes`, `no`, or `unknown`. |

Because these labels are read from context variables at increment time, they resolve to `unknown` for code paths that
run outside a gRPC request.

## Flow and session metrics

Most of these are emitted from `_record_metric` in
[`duo_workflow_service/checkpointer/gitlab_workflow.py`](../duo_workflow_service/checkpointer/gitlab_workflow.py),
alongside the equivalent [internal events](internal_events.md), so each Prometheus counter has an internal event twin
used for longer-term analysis.

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `agent_platform_receive_start_total` | Counter | `flow_type`, `<metadata>` | The service receives a `StartRequest` over gRPC (`duo_workflow_service/server.py`). |
| `agent_platform_session_start_total` | Counter | `flow_type`, `<metadata>` | A flow starts (`WORKFLOW_START`). |
| `agent_platform_session_resume_total` | Counter | `flow_type`, `<metadata>` | A flow resumes (`WORKFLOW_RESUME`). |
| `agent_platform_session_retry_total` | Counter | `flow_type`, `<metadata>` | A flow is retried (`WORKFLOW_RETRY`). |
| `agent_platform_session_reject_total` | Counter | `flow_type`, `<metadata>` | A flow is rejected (`WORKFLOW_REJECT`). |
| `agent_platform_session_success_total` | Counter | `flow_type`, `<metadata>` | A flow finishes successfully (`WORKFLOW_FINISH_SUCCESS`). |
| `agent_platform_session_failure_total` | Counter | `flow_type`, `failure_reason`, `session_type`, `<metadata>` | A flow fails (`WORKFLOW_FINISH_FAILURE`). `failure_reason` is the `error_type` of the internal event, `unknown` when absent. |
| `agent_platform_session_abort_total` | Counter | `flow_type`, `session_type`, `<metadata>` | A flow is aborted (`WORKFLOW_ABORTED`). |
| `duo_workflow_total_seconds` | Histogram | `workflow_type` | A flow finishes running, measured around `AbstractWorkflow.run`. Buckets: `WORKFLOW_TIME_SCALE_BUCKETS` (0.1s to 1h). |
| `duo_workflow_time_to_first_response_seconds` | Histogram | `flow_type`, `<metadata>` | The first outgoing action is produced. Measured from the timestamp set by the metadata context interceptor at the start of the RPC. Recorded at most once per flow. Buckets: `FIRST_RESPONSE_SCALE_BUCKETS` (0.5s to 60s). |

`session_type` comes from the `session_type_context` context variable, set by the checkpointer to `start`, `resume`, or
`retry` (see `SessionTypeEnum`). It defaults to `unknown`.

`flow_type` is the flow definition name (for example `software_development`, `chat`), and falls back to `unknown` when
the flow could not be identified.

`duo_workflow_total_seconds` uses `workflow_type` instead of `flow_type`. This is intentional: the metric predates the
`flow_type` naming convention and measures duration at the `AbstractWorkflow` level rather than the flow-registry level.
The two labels carry the same kind of value (the workflow/flow name), but renaming `workflow_type` would be a breaking
change for existing dashboards and alert rules.

## Tool and component metrics

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `duo_workflow_tool_call_seconds` | Histogram | `tool_name`, `flow_type`, `<metadata>` | Around every tool invocation. Emitted from `tools_executor`, `run_tool_node`, and the agent platform tool, one-off, and deterministic-step nodes. Default Prometheus buckets. |
| `agent_platform_tool_failure_total` | Counter | `flow_type`, `tool_name`, `failure_reason`, `<metadata>` | A tool call fails (`WORKFLOW_TOOL_FAILURE`), from `tools_executor` and `agent_platform/utils/tool_event_tracker.py`. `failure_reason` is the `error_type` of the internal event. |
| `executor_actions_duration_seconds` | Histogram | `action_class` | An action sent to the Duo Workflow Executor gets a response, measured in `record_metrics` in [`duo_workflow_service/executor/action.py`](../duo_workflow_service/executor/action.py). `action_class` is the protobuf oneof field name, for example `runHTTPRequest`. Only successful responses are recorded — errored responses raise before the observation. |
| `duo_workflow_compute_seconds` | Histogram | `operation_type` | Around agent processing in `Agent.run`. `operation_type` is `<agent name>_processing`. |
| `agent_platform_response_schema_output_total` | Counter | `flow_type`, `component_name`, `<metadata>` | A structured response schema output is produced by an agent component's final response node. |
| `agent_platform_flow_route_decision_total` | Counter | `flow_type`, `component_name`, `route_value`, `is_default_route`, `<metadata>` | A router picks an edge. `is_default_route` is the string `true` or `false`. |

## Conversation compaction metrics

Emitted from
[`duo_workflow_service/conversation/history_optimizer/optimizers/compaction.py`](../duo_workflow_service/conversation/history_optimizer/optimizers/compaction.py).

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `compaction_execution_total` | Counter | `flow_type`, `agent_name`, `status`, `<metadata>` | A compaction summarization completes. `status` is `success` or `error`. |
| `compaction_llm_duration_seconds` | Histogram | `flow_type`, `agent_name`, `<metadata>` | Around the summarization LLM call. Buckets: `LLM_TIME_SCALE_BUCKETS` (0.25s to 60s). |

## GitLab API and checkpoint metrics

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `duo_workflow_gitlab_response_seconds` | Histogram | `endpoint`, `method` | Around HTTP calls to the GitLab instance made by the checkpointer. `endpoint` is a templated path such as `/api/v4/ai/duo_workflows/workflows/:id/checkpoints`, so the label stays low cardinality. |
| `duo_workflow_checkpoint_total` | Counter | `endpoint`, `status_code`, `method`, `<metadata>` | A checkpoint is written to GitLab. `status_code` is the HTTP status, or `unknown` when the response is not a `GitLabHttpResponse`. |
| `duo_workflow_network_latency_seconds` | Histogram | `source`, `destination` | Declared for latency between the service and its dependencies. No production call site currently uses it; only tests exercise `time_network_latency`. |

## Audit event metrics

Emitted from `duo_workflow_service/audit_events/collector.py` and `duo_workflow_service/audit_events/client.py`. These
have no metadata labels.

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `duo_workflow_audit_events_captured_total` | Counter | `event_type` | The collector captures an audit event. |
| `duo_workflow_audit_events_sent_total` | Counter | `result` | A batch POST is attempted. `result` is `success`, `http_error`, or `exception`. Incremented by the number of events in the batch. |
| `duo_workflow_audit_events_dropped_total` | Counter | `reason` | Events are dropped before delivery. `reason` is `http_error`, `retries_exhausted`, or `version_unsupported`. Incremented by the number of events dropped. |
| `duo_workflow_audit_events_batch_size` | Histogram | none | A batch is POSTed. Buckets: 1, 5, 10, 25, 50, 100, 200, 500. |
| `duo_workflow_audit_events_payload_bytes` | Histogram | none | A batch is POSTed. Measures the UTF-8 serialized payload size. Buckets: 512 B to 1 MiB. |
| `duo_workflow_audit_events_auto_flush_skipped_total` | Counter | none | A buffer-full auto-flush is skipped because no event loop is running. |

## gRPC metrics

Emitted from `MonitoringInterceptor` in
[`duo_workflow_service/interceptors/monitoring_interceptor.py`](../duo_workflow_service/interceptors/monitoring_interceptor.py).

| Metric | Type | Labels | Emitted when |
|---|---|---|---|
| `grpc_server_handled_total` | Counter | `grpc_type`, `grpc_service`, `grpc_method`, `grpc_code`, `flow_type`, `<metadata>` | An RPC completes, successfully or not. `grpc_type` is one of `UNARY`, `SERVER_STREAM`, `CLIENT_STREAM`, `BIDI_STREAM`, `UNKNOWN`. `grpc_code` is the gRPC status code name, defaulting to `OK`. |

Two cases are deliberately excluded:

- gRPC health check methods are skipped entirely, so health probes do not inflate the counter.
- Connections that never sent a usable `StartRequest` are skipped. The interceptor sets
  `workflow_no_start_reason` on the monitoring context and logs `connection closed before workflow started` instead, so
  these no-op connections are traceable in logs without appearing as executed flows.

## Adding a metric

1. Declare the metric in `DuoWorkflowMetrics.__init__` in
   [`duo_workflow_service/tracking/duo_workflow_metrics.py`](../duo_workflow_service/tracking/duo_workflow_metrics.py),
   passing `registry=registry` so tests can supply an isolated registry.
1. Add `METADATA_LABELS` to the label names and spread `**build_metadata_labels()` at increment time unless the metric
   is emitted outside a request context.
1. Add a helper method on `DuoWorkflowMetrics` — a `count_*` method for counters, or a `time_*` method returning
   `self._timer(...)` for histograms — rather than calling `.labels(...)` from call sites.
1. Import the shared `duo_workflow_metrics` singleton from `duo_workflow_service.monitoring` at the call site. Do not
   instantiate `DuoWorkflowMetrics` directly; a second instance raises a duplicate-registration error against the
   default registry.
1. Keep label values low cardinality. Never use a workflow ID, user ID, or raw URL as a label value; template paths
   instead, as `duo_workflow_gitlab_response_seconds` does.
1. Add a test in `tests/duo_workflow_service/tracking/test_duo_workflow_metrics.py` and document the metric on this
   page.
1. Follow the steps in [Monitor the metric](#monitor-the-metric) to confirm the metric is visible and, optionally,
   add an SLI.

## Monitor the metric

After adding a metric, confirm it is visible and consider adding it to the service's SLI catalog.

### Confirm the metric exists

Scrape the local metrics endpoint and filter for the new metric name:

```shell
curl -s localhost:8083/metrics | grep <metric_name>
```

For counters, query the rate rather than the raw value:

```promql
rate(<metric_name>_total[5m])
```

For histograms, inspect the 99th-percentile latency:

```promql
histogram_quantile(0.99, sum by (le) (rate(<metric_name>_bucket[5m])))
```

### (Optional) Add an SLI

If the metric is a good signal for service reliability (for example, an error rate or latency histogram), consider
adding an SLI entry in the
[Duo Workflow Service runbook](https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/duo-workflow-svc.jsonnet).
Follow the patterns already present in that file — each SLI is a `metricsCatalog.serviceDefinition` entry with a
`sliSpec` block referencing the metric name and threshold.
