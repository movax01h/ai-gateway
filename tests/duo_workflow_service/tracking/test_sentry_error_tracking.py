from unittest.mock import patch

import pytest
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.langchain import LangchainIntegration
from sentry_sdk.integrations.langgraph import LanggraphIntegration
from sentry_sdk.integrations.litellm import LiteLLMIntegration

from duo_workflow_service.tracking.sentry_error_tracking import (
    DEFAULT_TRACES_SAMPLE_RATE,
    DEFAULT_WORKFLOW_TRACES_SAMPLE_RATE,
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
