import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection import (
    LLMDefinition,
    RestrictedModelAccessError,
)
from ai_gateway.model_selection.models import BaseModelParams, ModelClassProvider
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.typing import TypeModelFactory
from lib.context.model import restricted_access_ctx

pytestmark = pytest.mark.usefixtures("fake_restricted_model")


@pytest.fixture(name="authorized_flow")
def authorized_flow_fixture():
    return None


@pytest.fixture(autouse=True)
def restricted_access(authorized_flow: str | None):
    token = restricted_access_ctx.set(authorized_flow)
    yield
    restricted_access_ctx.reset(token)


def _prompt_config(model: str | None = "test_model") -> PromptConfig:
    return PromptConfig(
        name="test_prompt",
        model=ModelConfig(params=BaseModelParams(model=model)),
        unit_primitive=GitLabUnitPrimitive.DUO_AGENT_PLATFORM,
        prompt_template={"system": "Hi", "user": "{{content}}"},
    )


def _build_prompt(
    model_factory: TypeModelFactory,
    config: PromptConfig,
    model_metadata: ModelMetadata | None = None,
) -> Prompt:
    return Prompt(ModelClassProvider.LITE_LLM, model_factory, config, model_metadata)


def _restricted_metadata(fake_restricted_model: LLMDefinition) -> ModelMetadata:
    return ModelMetadata(
        provider="gitlab",
        name=fake_restricted_model.gitlab_identifier,
        llm_definition=fake_restricted_model,
    )


@pytest.mark.parametrize(
    "model",
    [
        "claude-fake-restricted-1",
        "anthropic/claude-fake-restricted-1",
        "vertex_ai/claude-fake-restricted-1@20261001",
    ],
)
def test_raw_restricted_model_string_is_denied(
    model_factory: TypeModelFactory, model: str
):
    """A custom flow naming the provider model directly cannot reach it."""
    with pytest.raises(RestrictedModelAccessError):
        _build_prompt(model_factory, _prompt_config(model))


def test_custom_provider_identifier_is_denied(
    model_factory: TypeModelFactory, llm_definition: LLMDefinition
):
    metadata = ModelMetadata(
        provider="custom_openai",
        name="mistral",
        llm_definition=llm_definition,
        identifier="anthropic/claude-fake-restricted-1",
    )

    with pytest.raises(RestrictedModelAccessError):
        _build_prompt(model_factory, _prompt_config(None), metadata)


def test_restricted_gitlab_identifier_is_denied(
    model_factory: TypeModelFactory, fake_restricted_model: LLMDefinition
):
    with pytest.raises(RestrictedModelAccessError):
        _build_prompt(
            model_factory,
            _prompt_config(None),
            _restricted_metadata(fake_restricted_model),
        )


@pytest.mark.parametrize("authorized_flow", ["developer"])
def test_restricted_model_is_denied_for_other_flow(
    model_factory: TypeModelFactory, fake_restricted_model: LLMDefinition
):
    with pytest.raises(RestrictedModelAccessError):
        _build_prompt(
            model_factory,
            _prompt_config(None),
            _restricted_metadata(fake_restricted_model),
        )


@pytest.mark.parametrize("authorized_flow", ["bl_security"])
def test_restricted_model_is_allowed_for_authorized_flow(
    model_factory: TypeModelFactory, fake_restricted_model: LLMDefinition
):
    prompt = _build_prompt(
        model_factory,
        _prompt_config(None),
        _restricted_metadata(fake_restricted_model),
    )

    assert prompt.name == "test_prompt"


def test_unrestricted_model_is_allowed(model_factory: TypeModelFactory):
    assert _build_prompt(model_factory, _prompt_config()).name == "test_prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized_flow", ["bl_security"])
async def test_authorization_propagates_to_tasks_threads_and_graph_branches(
    model_factory: TypeModelFactory, fake_restricted_model: LLMDefinition
):
    """Prompts built in spawned tasks, compile threads and fan-out branches keep the authorization."""

    def build() -> str:
        return _build_prompt(
            model_factory,
            _prompt_config(None),
            _restricted_metadata(fake_restricted_model),
        ).name

    assert await asyncio.create_task(asyncio.to_thread(build)) == "test_prompt"

    class State(TypedDict):
        items: list[int]
        built: Annotated[list[str], operator.add]

    async def unit(_state: dict) -> dict:
        return {"built": [build()]}

    graph = StateGraph(State)
    graph.add_node("unit", unit)
    graph.add_conditional_edges(
        START, lambda state: [Send("unit", {"item": i}) for i in state["items"]]
    )
    graph.add_edge("unit", END)

    result = await graph.compile().ainvoke({"items": [1, 2, 3], "built": []})

    assert result["built"] == ["test_prompt"] * 3


def test_restricted_model_reached_only_by_metadata_name_is_denied(
    model_factory: TypeModelFactory, llm_definition: LLMDefinition
):
    """Metadata naming a restricted model is denied even when its definition is an ordinary one."""
    metadata = ModelMetadata(
        provider="gitlab", name="fake_restricted_model", llm_definition=llm_definition
    )

    with pytest.raises(RestrictedModelAccessError):
        _build_prompt(model_factory, _prompt_config(None), metadata)
