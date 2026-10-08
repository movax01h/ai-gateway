# pylint: disable=direct-environment-variable-reference

import os
from typing import Optional

import sentry_sdk
import structlog
from langchain_core.tools import ToolException
from sentry_sdk.integrations.asyncio import enable_asyncio_integration
from sentry_sdk.integrations.grpc import GRPCIntegration
from sentry_sdk.integrations.langchain import LangchainIntegration
from sentry_sdk.integrations.langgraph import LanggraphIntegration
from sentry_sdk.types import Event, Hint

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

CLOUD_CONNECTOR_LOGGER = "cloud_connector"
# Raised while decoding a token the client sent, e.g. "Not enough segments" for a value that is not a JWT.
CLOUD_CONNECTOR_CLIENT_ERRORS = frozenset({"JWTError"})


def setup_error_tracking():
    if sentry_tracking_available():
        sentry_sdk.init(
            dsn=os.environ.get("SENTRY_DSN"),
            environment=os.environ.get("DUO_WORKFLOW_SERVICE_ENVIRONMENT"),
            traces_sampler=traces_sampler,
            before_send=before_send,
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


def before_send(event: Event, hint: Hint) -> Optional[Event]:
    """Drop expected errors, then strip private fields from the rest.

    https://docs.sentry.io/platforms/python/configuration/filtering/#using-before-send
    """
    if is_expected_error(event, hint):
        return None

    return remove_private_info_fields(event, hint)


def is_expected_error(event: Event, hint: Hint) -> bool:
    """Whether the event describes a condition the service already handles.

    `ToolException` means a tool failed on the model's input, for example a path that does not exist. The tools
    executor returns the error to the model so it can recover, and tracks it with the `WORKFLOW_TOOL_FAILURE` internal
    event. The LangChain and asyncio integrations still capture it as unhandled, and `log_exception` reports it again.

    `JWTError` from Cloud Connector means the client sent a malformed token. The authentication interceptor answers
    with `UNAUTHENTICATED`, so there is nothing for us to fix.

    Both errors are still written to the application logs.
    """
    exc_info = hint.get("exc_info")
    if exc_info and isinstance(exc_info[1], ToolException):
        return True

    # Logged errors (`log_exception`) arrive without `exc_info`: the structlog pipeline renders the traceback into
    # the message before the record reaches Sentry. Both `log_exception` helpers record the class name instead.
    extra = event.get("extra") or {}
    exception_class = extra.get("exception_class")
    if exception_class in _class_names(ToolException):
        return True

    return (
        event.get("logger") == CLOUD_CONNECTOR_LOGGER
        and exception_class in CLOUD_CONNECTOR_CLIENT_ERRORS
    )


def _class_names(cls: type) -> set[str]:
    """Names of `cls` and all its subclasses, including ones defined after this module was imported."""
    names = {cls.__name__}
    for subclass in cls.__subclasses__():
        names |= _class_names(subclass)
    return names


def remove_private_info_fields(event, hint):  # pylint: disable=unused-argument
    # Remove sensitive information from event data
    updated_event = event

    if "server_name" in updated_event:
        updated_event["server_name"] = None
    return updated_event
