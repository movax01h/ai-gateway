import logging
import os
from unittest.mock import patch

import pytest
import sentry_sdk
import structlog
from gitlab_cloud_connector.logging import (
    log_exception as cloud_connector_log_exception,
)
from jose import JWTError, jwt
from langchain_core.tools import ToolException
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.langchain import LangchainIntegration
from sentry_sdk.integrations.langgraph import LanggraphIntegration
from sentry_sdk.integrations.litellm import LiteLLMIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import Transport

from duo_workflow_service.errors.typing import TierAccessDeniedException
from duo_workflow_service.executor.action import ToolExceptionWithResponse
from duo_workflow_service.structured_logging import setup_logging
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.tracking.sentry_error_tracking import (
    DEFAULT_TRACES_SAMPLE_RATE,
    DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE,
    before_send,
    setup_async_error_tracking,
    setup_error_tracking,
    traces_sampler,
)


@pytest.fixture(name="init_kwargs")
def init_kwargs_fixture(monkeypatch):
    monkeypatch.setenv("SENTRY_ERROR_TRACKING_ENABLED", "true")
    monkeypatch.setenv("SENTRY_DSN", "https://public@example.ingest.sentry.io/1")

    with patch("sentry_sdk.init") as mock_init:
        setup_error_tracking()

    return mock_init.call_args.kwargs


def test_registers_langchain_integrations_without_prompts(init_kwargs):
    by_type = {type(i): i for i in init_kwargs["integrations"]}

    assert by_type[LangchainIntegration].include_prompts is False
    assert by_type[LanggraphIntegration].include_prompts is False


def test_does_not_register_litellm_integration(init_kwargs):
    # It would duplicate the LangChain span on every ChatLiteLLM call.
    assert LiteLLMIntegration not in {type(i) for i in init_kwargs["integrations"]}


def test_does_not_register_asyncio_integration_before_the_loop_runs(init_kwargs):
    # It patches the running event loop, so it has to be installed from inside it.
    assert AsyncioIntegration not in {type(i) for i in init_kwargs["integrations"]}


def test_samples_transactions_through_the_sampler(init_kwargs):
    assert init_kwargs["traces_sampler"] is traces_sampler
    assert "traces_sample_rate" not in init_kwargs


def test_filters_events_before_sending(init_kwargs):
    assert init_kwargs["before_send"] is before_send


def _exc_info(exc):
    return (type(exc), exc, None)


@pytest.mark.parametrize(
    ("event", "hint"),
    [
        pytest.param(
            {"level": "error"},
            {"exc_info": _exc_info(ToolException("404 File Not Found"))},
            id="tool_exception",
        ),
        pytest.param(
            {"level": "error"},
            {"exc_info": _exc_info(ToolExceptionWithResponse("Action error", "out"))},
            id="tool_exception_subclass",
        ),
        pytest.param(
            {
                "logger": "cloud_connector",
                "extra": {"exception_class": "JWTError"},
            },
            {},
            id="cloud_connector_malformed_token",
        ),
        pytest.param(
            {"logger": "exceptions", "extra": {"exception_class": "ToolException"}},
            {},
            id="logged_tool_exception",
        ),
        pytest.param(
            {
                "logger": "exceptions",
                "extra": {"exception_class": TierAccessDeniedException.__name__},
            },
            {},
            id="logged_tool_exception_subclass",
        ),
    ],
)
def test_before_send_drops_expected_errors(event, hint):
    assert before_send(event, hint) is None


@pytest.mark.parametrize(
    ("event", "hint"),
    [
        pytest.param(
            {"level": "error"},
            {"exc_info": _exc_info(ValueError("boom"))},
            id="other_exception",
        ),
        pytest.param(
            {"logger": "cloud_connector", "extra": {"exception_class": "KeyError"}},
            {},
            id="cloud_connector_other_error",
        ),
        pytest.param(
            {"logger": "exceptions", "extra": {"exception_class": "JWTError"}},
            {},
            id="jwt_error_from_another_logger",
        ),
        pytest.param(
            {"logger": "exceptions", "extra": {"exception_class": "RuntimeError"}},
            {},
            id="logged_other_exception",
        ),
        pytest.param({"message": "Failed to save checkpoint"}, {}, id="log_message"),
    ],
)
def test_before_send_keeps_other_errors(event, hint):
    event["server_name"] = "pod-123"

    result = before_send(event, hint)

    assert result is event
    assert result["server_name"] is None


def test_setup_async_error_tracking_installs_the_asyncio_integration():
    with patch(
        "duo_workflow_service.tracking.sentry_error_tracking.enable_asyncio_integration"
    ) as enable:
        setup_async_error_tracking()

    enable.assert_called_once_with(task_spans=False)


def _sampling_context(name, parent_sampled=None):
    return {
        "transaction_context": {"name": name, "op": "grpc.server"},
        "parent_sampled": parent_sampled,
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("/grpc.health.v1.Health/Check", 0.0),
        ("/grpc.health.v1.Health/Watch", 0.0),
        ("/DuoWorkflow/ExecuteWorkflow", DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE),
        ("/DuoWorkflow/GenerateToken", DEFAULT_TRACES_SAMPLE_RATE),
    ],
)
def test_default_sample_rates(name, expected):
    assert traces_sampler(_sampling_context(name)) == expected


@pytest.mark.parametrize("parent_sampled", [True, False])
def test_inherits_an_upstream_sampling_decision(parent_sampled):
    # Half-sampled traces are worse than either decision taken consistently.
    assert traces_sampler(
        _sampling_context("/DuoWorkflow/ExecuteWorkflow", parent_sampled)
    ) == float(parent_sampled)


def test_health_checks_are_dropped_even_when_the_caller_sampled_them():
    assert (
        traces_sampler(_sampling_context("/grpc.health.v1.Health/Check", True)) == 0.0
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.5", 0.5),
        ("1", 1.0),
        ("0", 0.0),
    ],
)
def test_workflow_sample_rate_is_configurable(monkeypatch, value, expected):
    monkeypatch.setenv("SENTRY_WORKFLOW_TRACES_SAMPLE_RATE", value)

    assert traces_sampler(_sampling_context("/DuoWorkflow/ExecuteWorkflow")) == expected


def test_default_sample_rate_is_configurable(monkeypatch):
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")

    assert traces_sampler(_sampling_context("/DuoWorkflow/GenerateToken")) == 0.25


@pytest.mark.parametrize("value", ["", "not-a-number", "-0.1", "2"])
def test_unusable_sample_rates_fall_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv("SENTRY_WORKFLOW_TRACES_SAMPLE_RATE", value)

    assert (
        traces_sampler(_sampling_context("/DuoWorkflow/ExecuteWorkflow"))
        == DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE
    )


def test_sampler_tolerates_a_missing_transaction_context():
    assert traces_sampler({}) == DEFAULT_TRACES_SAMPLE_RATE


class _CaptureTransport(Transport):
    def __init__(self):
        super().__init__()
        self.events = []

    def capture_envelope(self, envelope):
        for item in envelope.items:
            if item.type == "event" and (payload := item.payload.json):
                self.events.append(payload)

    def flush(self, *args, **kwargs):
        pass

    def kill(self):
        pass


@pytest.fixture(name="sentry_events")
def sentry_events_fixture():
    """Send real log records through DWS logging and Sentry's LoggingIntegration into `before_send`.

    The Cloud Connector filter relies on how its logger name and `exception_class` field reach the Sentry event, so
    the whole pipeline is exercised rather than hand-built events.
    """
    # pylint: disable=direct-environment-variable-reference
    root_logger = logging.getLogger()
    original_handlers, original_level = root_logger.handlers[:], root_logger.level
    original_structlog_config = structlog.get_config()
    # Restoring the previous client rather than re-initialising: integrations are installed process-wide.
    old_client = sentry_sdk.get_global_scope().client
    transport = _CaptureTransport()

    with patch.dict(
        os.environ,
        {
            "DUO_WORKFLOW_SERVICE_ENVIRONMENT": "production",
            "DUO_WORKFLOW_LOGGING__JSON_FORMAT": "true",
        },
    ):
        # A cached logger keeps this production processor chain after the fixture
        # restores the config, which hides its events from `capture_logs` in later tests.
        setup_logging(cache_logger_on_first_use=False)
    root_logger.handlers = [logging.NullHandler()]

    sentry_sdk.init(
        dsn="https://public@example.ingest.sentry.io/1",
        default_integrations=False,
        integrations=[LoggingIntegration()],
        before_send=before_send,
        transport=transport,
    )

    try:
        yield transport.events
    finally:
        sentry_sdk.get_global_scope().set_client(old_client)
        structlog.configure(**original_structlog_config)
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)


def _malformed_token_error() -> JWTError:
    try:
        jwt.decode("not-a-jwt", {"keys": []}, algorithms=["RS256"])
    except JWTError as err:
        return err
    raise AssertionError("expected a JWTError")


def test_pipeline_drops_cloud_connector_malformed_token(sentry_events):
    # The exact call `gitlab_cloud_connector` makes when a client sends a token that is not a JWT.
    cloud_connector_log_exception(_malformed_token_error())

    assert not sentry_events


def test_pipeline_keeps_other_cloud_connector_errors(sentry_events):
    # Guards the test above: the pipeline does deliver Cloud Connector errors.
    cloud_connector_log_exception(ValueError("JWKS unavailable"))

    assert [event["logger"] for event in sentry_events] == ["cloud_connector"]
    assert sentry_events[0]["extra"]["exception_class"] == "ValueError"


@pytest.mark.parametrize(
    ("error", "expected_events"),
    [
        (ToolException("Request failed (Get repository file): HTTP 404"), 0),
        (ToolExceptionWithResponse("Action error: File not found", "out"), 0),
        (RuntimeError("boom"), 1),
    ],
)
def test_pipeline_filters_logged_exceptions(sentry_events, error, expected_events):
    # The tools executor reports every ToolException through `log_exception`.
    log_exception(error, extra={"context": "Tools executor raised error"})

    assert len(sentry_events) == expected_events
