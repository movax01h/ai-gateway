# pylint: disable=direct-environment-variable-reference

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from prometheus_client import CollectorRegistry, generate_latest

from duo_workflow_service.monitoring import setup_monitoring
from duo_workflow_service.tracking.duo_workflow_metrics import (
    TOOL_CALL_TIME_SCALE_BUCKETS,
    DuoWorkflowMetrics,
)
from lib.context.request_metadata import METADATA_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]

# The created-series switch is process-wide, so check it in a child process
# instead of flipping it for the rest of the test session. The child has no
# metrics address or port, so setup_monitoring does not start a server.
_SETUP_SCRIPT = """
import json

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from duo_workflow_service.monitoring import setup_monitoring


def exposition():
    registry = CollectorRegistry()
    Counter("example_events", "Example counter", registry=registry).inc()
    Histogram("example_seconds", "Example histogram", registry=registry).observe(0.1)
    return generate_latest(registry).decode()


before = exposition()
setup_monitoring()
print(json.dumps({"before": before, "after": exposition()}))
"""


_METRICS_ENV_KEYS = (
    "PROMETHEUS_METRICS__ADDR",
    "PROMETHEUS_METRICS__PORT",
    "PROMETHEUS_DISABLE_CREATED_SERIES",
)


def _env_without_metrics_server():
    return {
        key: value for key, value in os.environ.items() if key not in _METRICS_ENV_KEYS
    }


def test_setup_monitoring_disables_created_metrics():
    # The subprocess test below checks the exposition, but coverage cannot see
    # into the child, so call setup_monitoring in-process with the switch mocked.
    with (
        patch.dict(os.environ, _env_without_metrics_server(), clear=True),
        patch("duo_workflow_service.monitoring.disable_created_metrics") as disable,
        patch("duo_workflow_service.monitoring.start_http_server") as start_server,
    ):
        setup_monitoring()

    disable.assert_called_once_with()
    start_server.assert_not_called()


def test_setup_monitoring_stops_exporting_created_series():
    result = subprocess.run(
        [sys.executable, "-c", _SETUP_SCRIPT],
        cwd=REPO_ROOT,
        env=_env_without_metrics_server(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"setup script exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    exposition = json.loads(result.stdout.strip().splitlines()[-1])

    assert "example_events_created" in exposition["before"]
    assert "example_seconds_created" in exposition["before"]
    assert "_created" not in exposition["after"]
    assert "example_events_total" in exposition["after"]
    assert "example_seconds_count" in exposition["after"]


def test_tool_call_histogram_keeps_version_labels():
    # Dashboards and alerts split errors by client and instance version. Cut
    # cardinality some other way instead of removing these labels.
    registry = CollectorRegistry()
    metrics = DuoWorkflowMetrics(registry=registry)

    metrics.tool_call_duration.labels(
        tool_name="find_files",
        flow_type="code_review",
        **dict.fromkeys(METADATA_LABELS, "value"),
    ).observe(0.1)

    bucket_lines = [
        line
        for line in generate_latest(registry).decode().splitlines()
        if line.startswith("duo_workflow_tool_call_seconds_bucket")
    ]

    assert bucket_lines
    assert all(
        'lsp_version="value"' in line and 'gitlab_version="value"' in line
        for line in bucket_lines
    )


def test_tool_call_histogram_uses_trimmed_buckets():
    # 99% of tool calls finish within 10 seconds, and the edges above it show
    # the slow tail that the default buckets fold into +Inf.
    assert TOOL_CALL_TIME_SCALE_BUCKETS == [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]

    registry = CollectorRegistry()
    metrics = DuoWorkflowMetrics(registry=registry)

    metrics.tool_call_duration.labels(
        tool_name="find_files",
        flow_type="code_review",
        **dict.fromkeys(METADATA_LABELS, "value"),
    ).observe(0.1)

    edges = [
        sample.labels["le"]
        for metric in registry.collect()
        for sample in metric.samples
        if sample.name == "duo_workflow_tool_call_seconds_bucket"
    ]

    assert edges == [str(float(b)) for b in TOOL_CALL_TIME_SCALE_BUCKETS] + ["+Inf"]
