# pylint: disable=direct-environment-variable-reference

import os
from typing import Optional

import sentry_sdk
import structlog
from sentry_sdk.integrations.asyncio import enable_asyncio_integration
from sentry_sdk.integrations.grpc import GRPCIntegration
from sentry_sdk.integrations.langchain import LangchainIntegration
from sentry_sdk.integrations.langgraph import LanggraphIntegration

from duo_workflow_service.interceptors import GRPC_HEALTH_METHODS

log = structlog.stdlib.get_logger("error_tracking")

# Every RPC that is not a long-running flow is cheap to trace, so keep all of them.
DEFAULT_TRACES_SAMPLE_RATE = 1.0

# `ExecuteWorkflow` is a bidirectional stream that stays open for the whole flow and
# collects a span per LLM call and per LangGraph node. Sentry buffers the transaction
# in memory until the stream closes and drops everything past 1000 spans, so tracing
# every flow is both expensive and lossy. Sample a slice of them instead; errors are
# unaffected because trace sampling only applies to transactions.
DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE = 0.05

WORKFLOW_METHOD_NAME = "ExecuteWorkflow"


def setup_error_tracking():
    if sentry_tracking_available():
        sentry_sdk.init(
            dsn=os.environ.get("SENTRY_DSN"),
            environment=os.environ.get("DUO_WORKFLOW_SERVICE_ENVIRONMENT"),
            traces_sampler=traces_sampler,
            before_send=remove_private_info_fields,
            profiles_sample_rate=0.0,
            integrations=[
                GRPCIntegration(),
                LangchainIntegration(include_prompts=False),
                LanggraphIntegration(include_prompts=False),
            ],
            max_value_length=30 * 1024,
        )


def setup_async_error_tracking() -> None:
    """Install the asyncio instrumentation from inside the running event loop.

    `AsyncioIntegration` patches the loop's task factory, so passing it to
    `sentry_sdk.init` is a no-op when the loop has not started yet, which is the case
    for `setup_error_tracking`. `task_spans` stays off because a span per asyncio task
    would exhaust Sentry's per-transaction span budget.
    """
    enable_asyncio_integration(task_spans=False)


def traces_sampler(sampling_context: dict) -> float:
    """Pick the transaction sample rate for the RPC being traced.

    https://docs.sentry.io/platforms/python/configuration/sampling/#setting-a-sampling-function
    """
    transaction_context = sampling_context.get("transaction_context") or {}
    name = transaction_context.get("name") or ""

    # Kubernetes probes these constantly and the transactions carry no useful data.
    if name in GRPC_HEALTH_METHODS:
        return 0.0

    # Keep a trace whole: if an upstream service already made a decision, follow it.
    parent_sampled = sampling_context.get("parent_sampled")
    if parent_sampled is not None:
        return float(parent_sampled)

    if name.endswith(f"/{WORKFLOW_METHOD_NAME}"):
        return _sample_rate_from_env(
            "SENTRY_WORKFLOW_TRACES_SAMPLE_RATE", DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE
        )

    return _sample_rate_from_env(
        "SENTRY_TRACES_SAMPLE_RATE", DEFAULT_TRACES_SAMPLE_RATE
    )


def _sample_rate_from_env(name: str, default: float) -> float:
    raw: Optional[str] = os.environ.get(name)
    if not raw:
        return default

    try:
        rate = float(raw)
    except ValueError:
        log.warning("Ignoring unparsable sample rate", variable=name, value=raw)
        return default

    if not 0.0 <= rate <= 1.0:
        log.warning("Ignoring out-of-range sample rate", variable=name, value=raw)
        return default

    return rate


def sentry_tracking_available():
    if os.environ.get("SENTRY_ERROR_TRACKING_ENABLED") == "true":
        if os.environ.get("SENTRY_DSN"):
            log.debug("Using Sentry for error tracking...")
            return True
        log.debug("Could not find Sentry DSN for error tracking setup...")
    else:
        log.debug("Sentry error tracking disabled...")
    return False


def remove_private_info_fields(event, hint):  # pylint: disable=unused-argument
    # Remove sensitive information from event data
    updated_event = event

    if "server_name" in updated_event:
        updated_event["server_name"] = None
    return updated_event
