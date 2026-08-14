# pylint: disable=file-naming-for-tests
"""Integration test: the agentic_chat flow's provider_params reach the OpenAI wire.

The registry `agentic_chat` flow defines its prompt inline, so its model
params travel flow config -> InMemoryPromptRegistry -> real model factory
rather than through the file-based prompt definitions. This test drives that
full path — real gRPC server, real flow, real `ModelMetadataInterceptor`
metadata, real `ChatOpenAI` — and asserts the tuning block lands in the
request payload at the OpenAI SDK boundary.

The OpenAI provider is wired unconditionally in `ai_gateway/prompts/container.py`
(no `mock_model_responses`-gated branch like Anthropic/LiteLLM/AmazonQ have), so
the shared, module-scoped `mock_duo_workflow_service_container` fixture from the
parent conftest is used as-is: `FakeModel` never enters the picture for this
provider regardless of `mock_model_responses`. The SDK call itself is intercepted,
so no network I/O happens.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from openai.resources.responses import AsyncResponses

import tests.duo_workflow_service.integration.conftest as integration_conftest
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from duo_workflow_service.server import DuoWorkflowService
from tests.duo_workflow_service.integration.conftest import (
    FakeExecutor,
    run_exchange,
    start_registry_flow_event,
)


@pytest.fixture(autouse=True)
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def gpt_model_metadata(monkeypatch):
    monkeypatch.setattr(
        integration_conftest,
        "MODEL_METADATA_HEADER",
        (
            ModelMetadataInterceptor.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA,
            # The exact header shape Rails sends for a user-selected model.
            json.dumps(
                {
                    "provider": "gitlab",
                    "feature_setting": "duo_agent_platform_agentic_chat",
                    "identifier": "gpt_5_6_terra",
                }
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
async def test_agentic_chat_flow_sends_tuning_on_the_wire(
    servicer: DuoWorkflowService,
    executor: FakeExecutor,
):
    with patch.object(
        AsyncResponses,
        "create",
        new_callable=AsyncMock,
        side_effect=RuntimeError("payload captured; aborting before network"),
    ) as create:
        await run_exchange(
            servicer,
            executor,
            start_registry_flow_event(
                goal="What does this project do?",
                flow_config_id="agentic_chat",
                schema_version="v1",
                version="1.0.0",
            ),
        )

    assert create.await_args is not None, "flow never reached the OpenAI SDK boundary"
    payload = create.await_args.kwargs

    assert str(payload.get("model", "")).startswith("gpt-5.6"), payload.get("model")
    assert payload.get("reasoning") == {"summary": "auto", "effort": 8}, payload.get(
        "reasoning"
    )
    assert payload.get("text", {}).get("verbosity") == "low", payload.get("text")
