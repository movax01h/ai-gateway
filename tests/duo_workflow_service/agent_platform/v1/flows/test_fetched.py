"""``FetchedFlow``: a saved AI Catalog flow built from the definition GitLab serves."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.agent_platform.v1.catalog import AiCatalogAgent
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogAgentsUnavailableError,
    CatalogFlowUnavailableError,
    CatalogItemsError,
    CatalogItemsFetchError,
)
from duo_workflow_service.agent_platform.v1.catalog.fetch import (
    FetchedCatalogAgents,
    FetchedCatalogFlow,
)
from duo_workflow_service.agent_platform.v1.flows import FetchedFlow
from duo_workflow_service.agent_platform.v1.flows.fetched import (
    FetchedFlowConfigError,
    FetchedFlowDigestMismatchError,
    FetchedFlowFetchError,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.errors.typing import WorkflowTokenRejectedError
from duo_workflow_service.flow_config_digest import definition_digest
from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.workflows.registry import (
    flow_config_from_fetched_definition,
)
from lib.events import GLReportingEventContext

pytestmark = pytest.mark.usefixtures("mock_duo_workflow_service_container")

AI_CATALOG_REF = {
    "source": "ai-catalog",
    "item_type": "agent",
    "item_id": "53",
    "version": "1.0.0",
}


def _definition(include=None, config_id=None) -> dict:
    definition = {
        "version": "v1",
        "environment": "ambient",
        "components": [
            {
                "name": "agent",
                "type": "AgentComponent",
                "inputs": ["context:goal"],
                "prompt_id": "fetched_prompt",
                "toolset": [],
            }
        ],
        "routers": [{"from": "agent", "to": "end"}],
        "flow": {"entry_point": "agent"},
        "prompts": [
            {
                "prompt_id": "fetched_prompt",
                "name": "Fetched prompt",
                "unit_primitives": ["duo_agent_platform"],
                "prompt_template": {"user": "{{goal}}", "system": "You help."},
                "params": {"timeout": 180},
            }
        ],
    }
    if include is not None:
        definition["include"] = include
    if config_id is not None:
        definition["config_id"] = config_id
    return definition


def _agent() -> AiCatalogAgent:
    return AiCatalogAgent.model_validate(
        {
            "item_id": "53",
            "version": "1.0.0",
            "name": "Code Reviewer",
            "description": "Reviews code.",
            "prompt": "You review code.",
        }
    )


def _flow(
    user, config_builder=flow_config_from_fetched_definition, digest=None
) -> FetchedFlow:
    return FetchedFlow(
        config_builder=config_builder,
        authorized_flow_config_digest=digest,
        workflow_id="44",
        workflow_metadata={},
        workflow_type=GLReportingEventContext.from_workflow_definition("my_flow/v1"),
        user=user,
    )


@pytest.fixture(name="fetch")
def fetch_fixture():
    with patch(
        "duo_workflow_service.agent_platform.v1.flows.fetched.fetch_catalog_flow",
        new=AsyncMock(),
    ) as fetch:
        yield fetch


def _served(definition, agents=None, skipped_reason=None) -> FetchedCatalogFlow:
    return FetchedCatalogFlow(
        definition=definition,
        agents=FetchedCatalogAgents(agents=agents or [], skipped_reason=skipped_reason),
    )


def test_the_flow_is_created_without_a_config(user):
    flow = _flow(user)

    assert not hasattr(flow, "_config")
    assert flow._registry_flow_id() is None


@pytest.mark.asyncio
async def test_the_fetched_definition_becomes_the_config(fetch, user):
    fetch.return_value = _served(_definition())
    flow = _flow(user)

    await flow._prepare_compile()

    fetch.assert_awaited_once_with(flow._http_client, "44")
    assert isinstance(flow._config, FlowConfig)
    assert flow._config.components[0]["prompt_id"] == "fetched_prompt"
    # pylint: disable-next=protected-access
    assert "fetched_prompt" in flow._flow_prompt_registry._raw_prompt_data


@pytest.mark.asyncio
async def test_the_agents_bind_from_the_same_response(fetch, user):
    fetch.return_value = _served(_definition(include=[AI_CATALOG_REF]), [_agent()])
    flow = _flow(user)

    await flow._prepare_compile()

    fetch.assert_awaited_once()
    assert [a.item_id for a in flow._catalog_items.ai_catalog_agents] == ["53"]


@pytest.mark.asyncio
async def test_a_skip_reason_is_recorded(fetch, user):
    fetch.return_value = _served(
        _definition(include=[AI_CATALOG_REF]), skipped_reason="custom_agents_disabled"
    )
    flow = _flow(user)

    await flow._prepare_compile()

    assert flow._catalog_items.ai_catalog_agents == []
    assert flow._catalog_items.ai_catalog_skipped_reason == "custom_agents_disabled"


@pytest.mark.asyncio
async def test_agents_are_not_bound_when_the_flow_does_not_reference_the_catalog(
    fetch, user
):
    fetch.return_value = _served(_definition(), [_agent()])
    flow = _flow(user)

    await flow._prepare_compile()

    assert flow._catalog_items.ai_catalog_agents == []


@pytest.mark.asyncio
async def test_a_definition_that_claims_a_registry_id_gets_no_registry_grant(
    fetch, user
):
    fetch.return_value = _served(_definition(config_id="developer"))
    flow = _flow(user)

    await flow._prepare_compile()

    assert flow._registry_flow_id() is None


@pytest.mark.asyncio
async def test_a_missing_definition_fails_the_run(fetch, user):
    fetch.side_effect = CatalogFlowUnavailableError("HTTP 404")
    flow = _flow(user)

    with pytest.raises(CatalogFlowUnavailableError):
        await flow._prepare_compile()


@pytest.mark.asyncio
async def test_a_failed_fetch_fails_the_run_with_a_retry_message(fetch, user):
    error = CatalogItemsFetchError("HTTP 500: boom")
    fetch.side_effect = error
    flow = _flow(user)

    with pytest.raises(FetchedFlowFetchError) as exc_info:
        await flow._prepare_compile()

    assert exc_info.value.ui_message == FetchedFlowFetchError.UI_MESSAGE
    assert "HTTP 500" in exc_info.value.internal_detail
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_a_refused_agent_fails_the_run_with_the_agent_message(fetch, user):
    fetch.side_effect = CatalogItemsError("agent 53 was refused")
    flow = _flow(user)

    with pytest.raises(CatalogAgentsUnavailableError) as exc_info:
        await flow._prepare_compile()

    assert exc_info.value.ui_message == CatalogAgentsUnavailableError.REFUSED_MESSAGE
    assert "agent 53" in exc_info.value.internal_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError("bad"), TypeError("bad"), SecurityException("bad")]
)
async def test_a_definition_that_cannot_be_built_fails_the_run(fetch, user, error):
    fetch.return_value = _served(_definition())
    flow = _flow(user, config_builder=Mock(side_effect=error))

    with pytest.raises(FetchedFlowConfigError) as exc_info:
        await flow._prepare_compile()

    assert exc_info.value.ui_message == FetchedFlowConfigError.UI_MESSAGE
    assert "bad" in exc_info.value.internal_detail
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_a_bad_inline_response_schema_fails_the_run(fetch, user):
    """The builder succeeds, but registering the schema raises ValueError."""
    definition = _definition()
    definition["response_schemas"] = [
        {"schema_id": "agent", "definition": {"type": "array"}}
    ]
    fetch.return_value = _served(definition)
    flow = _flow(user)

    with pytest.raises(FetchedFlowConfigError) as exc_info:
        await flow._prepare_compile()

    assert exc_info.value.ui_message == FetchedFlowConfigError.UI_MESSAGE
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_a_definition_that_matches_the_token_digest_is_used(fetch, user):
    definition = _definition()
    fetch.return_value = _served(definition)
    flow = _flow(user, digest=definition_digest(definition))

    await flow._prepare_compile()

    assert isinstance(flow._config, FlowConfig)


@pytest.mark.asyncio
async def test_a_definition_with_ints_matches_a_digest_of_floats(fetch, user):
    """Rails sent the flow to GenerateToken as a Struct, where 180 is 180.0."""
    definition = _definition()
    as_floats = {
        **definition,
        "prompts": [{**definition["prompts"][0], "params": {"timeout": 180.0}}],
    }
    fetch.return_value = _served(definition)
    flow = _flow(user, digest=definition_digest(as_floats))

    await flow._prepare_compile()

    assert isinstance(flow._config, FlowConfig)


@pytest.mark.asyncio
async def test_a_definition_that_does_not_match_the_token_digest_fails_the_run(
    fetch, user
):
    """A client that answers the fetch itself cannot swap the flow."""
    authorized = _definition()
    swapped = _definition(config_id="swapped")
    fetch.return_value = _served(swapped, [_agent()])
    builder = Mock(wraps=flow_config_from_fetched_definition)
    flow = _flow(user, config_builder=builder, digest=definition_digest(authorized))

    with pytest.raises(FetchedFlowDigestMismatchError) as exc_info:
        await flow._prepare_compile()

    error = exc_info.value
    assert error.ui_message == FetchedFlowDigestMismatchError.UI_MESSAGE
    assert definition_digest(swapped) in error.internal_detail
    assert definition_digest(swapped) not in error.ui_message
    # The server reads last_error from __cause__; a WorkflowTokenRejectedError there
    # maps the run to PERMISSION_DENIED instead of leaving it UNKNOWN.
    assert isinstance(error.__cause__, WorkflowTokenRejectedError)
    assert definition_digest(swapped) not in str(error.__cause__)
    builder.assert_not_called()
    assert not hasattr(flow, "_config")
    assert flow._catalog_items.ai_catalog_agents == []


@pytest.mark.asyncio
async def test_a_run_without_a_digest_is_not_checked(fetch, user):
    """Only a debug user's run has no digest: the server refuses every other token without one."""
    fetch.return_value = _served(_definition())
    flow = _flow(user, digest=None)

    await flow._prepare_compile()

    assert isinstance(flow._config, FlowConfig)
