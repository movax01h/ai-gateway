from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import CatalogItems
from duo_workflow_service.agent_platform.v1.catalog.sources.ai_catalog_agent import (
    AiCatalogAgentSource,
    ai_catalog_agent_prompt,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import BindRequest
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
)

REFERENCE = {
    "source": "ai-catalog",
    "item_type": "agent",
    "item_id": "42",
    "version": "1.0.0",
}


def _ref(**overrides) -> CatalogItemRef:
    return CatalogItemRef.model_validate({**REFERENCE, **overrides})


def _agent(**overrides) -> dict:
    return {
        "item_id": "42",
        "version": "1.0.0",
        "name": "tester",
        "description": "Runs tests.",
        "toolset": ["run_tests"],
        "prompt": "You are a specialist.",
        **overrides,
    }


def _items(*agents: dict) -> CatalogItems:
    return CatalogItems.model_validate({"ai_catalog_agents": list(agents)})


def _bind(
    items: CatalogItems,
    claimant: dict | None = None,
    tools_registry: Mock | None = None,
) -> list[dict]:
    claimant = claimant or {"name": "coordinator", "subagents": [dict(REFERENCE)]}

    return AiCatalogAgentSource().bind(
        BindRequest(
            ref=_ref(),
            claimant_config=claimant,
            components_config=[claimant],
            items=items,
            tools_registry=tools_registry or Mock(),
        )
    )


def test_it_serves_ai_catalog_agents():
    assert AiCatalogAgentSource.SOURCE is CatalogItemSource.AI_CATALOG
    assert AiCatalogAgentSource.ITEM_TYPE is CatalogItemType.AGENT


def test_it_is_not_flag_gated():
    assert AiCatalogAgentSource.EXPERIMENT_FLAG is None
    assert AiCatalogAgentSource().is_enabled


def test_the_shipped_prompt_is_named_after_the_source_and_kind():
    assert ai_catalog_agent_prompt().prompt_id == "ai-catalog_agent_prompt"


class TestValidateRef:
    def test_an_id_with_a_version_is_accepted(self):
        AiCatalogAgentSource().validate_ref(_ref())

    @pytest.mark.parametrize("item_id", ["", "   "], ids=["empty", "whitespace"])
    def test_an_id_naming_no_agent_is_rejected(self, item_id):
        with pytest.raises(CatalogItemConfigError, match="must name an agent"):
            AiCatalogAgentSource().validate_ref(_ref(item_id=item_id))

    @pytest.mark.parametrize(
        "version", [None, "", "   "], ids=["none", "empty", "whitespace"]
    )
    def test_a_version_naming_nothing_is_rejected(self, version):
        with pytest.raises(CatalogItemConfigError, match="must declare a version"):
            AiCatalogAgentSource().validate_ref(_ref(version=version))


class TestBindWithoutAMatch:
    @pytest.mark.parametrize(
        "items",
        [
            CatalogItems(),
            _items(_agent(item_id="43")),
            _items(_agent(version="2.0.0")),
        ],
        ids=["no_items", "other_id", "other_version"],
    )
    def test_the_run_fails_and_names_the_reference(self, items):
        with pytest.raises(
            CatalogItemConfigError,
            match="Subagent '42' at version '1.0.0' is not among this session's",
        ):
            _bind(items)


class TestBindWhenGitLabSkippedTheAgents:
    def test_the_claim_is_removed_and_nothing_is_attached(self):
        items = CatalogItems(ai_catalog_skipped_reason="custom_agents_disabled")

        bound = _bind(items)

        assert len(bound) == 1
        assert "subagents" not in bound[0]


class TestBindWithAMatch:
    def test_the_claimant_is_promoted_to_a_supervisor(self):
        bound = _bind(_items(_agent()))

        assert bound[0]["name"] == "coordinator"
        assert bound[0]["subagents"] == [{"name": "ai-catalog/agents/tester (42)"}]
        assert bound[0]["ui_log_events"] == [
            "on_delegation",
            "on_delegation_returns",
            "on_delegation_error",
        ]

    def test_one_component_is_synthesized_for_the_agent(self):
        bound = _bind(_items(_agent()))

        assert len(bound) == 2
        synthesized = bound[1]
        assert synthesized["type"] == "AgentComponent"
        assert synthesized["prompt_id"] == "ai-catalog_agent_prompt"
        assert synthesized["name"] == "ai-catalog/agents/tester (42)"
        assert synthesized["description"] == "Runs tests."
        assert synthesized["toolset"] == ["run_tests"]

    def test_the_item_prompt_travels_as_a_literal_input(self):
        bound = _bind(_items(_agent()))

        assert {
            "from": "You are a specialist.",
            "as": "item_prompt",
            "literal": True,
        } in bound[1]["inputs"]

    def test_the_match_is_by_id_among_many(self):
        bound = _bind(_items(_agent(item_id="41"), _agent(), _agent(item_id="43")))

        assert [component["name"] for component in bound] == [
            "coordinator",
            "ai-catalog/agents/tester (42)",
        ]

    def test_static_subagents_compose_with_the_matched_agent(self):
        claimant = {
            "name": "coordinator",
            "subagents": [{"name": "static_helper"}, dict(REFERENCE)],
        }

        bound = _bind(_items(_agent()), claimant=claimant)

        assert bound[0]["subagents"] == [
            {"name": "static_helper"},
            {"name": "ai-catalog/agents/tester (42)"},
        ]

    def test_other_components_pass_through_untouched(self):
        claimant = {"name": "coordinator", "subagents": [dict(REFERENCE)]}
        other = {"name": "step", "type": "DeterministicStepComponent"}

        bound = AiCatalogAgentSource().bind(
            BindRequest(
                ref=_ref(),
                claimant_config=claimant,
                components_config=[claimant, other],
                items=_items(_agent()),
                tools_registry=Mock(),
            )
        )

        assert bound[1] is other
        assert len(bound) == 3

    def test_strict_validation_is_inherited_from_the_claimant(self):
        claimant = {
            "name": "coordinator",
            "strict_validation": True,
            "subagents": [dict(REFERENCE)],
        }

        bound = _bind(_items(_agent()), claimant=claimant)

        assert bound[1]["strict_validation"] is True

    def test_toolsets_are_resolved_through_the_registry(self):
        tools_registry = Mock()

        _bind(_items(_agent()), tools_registry=tools_registry)

        tools_registry.toolset.assert_called_once_with(["run_tests"])

    def test_an_unusable_tool_is_rejected(self):
        tools_registry = Mock()
        tools_registry.toolset.side_effect = ValueError("unknown tool 'run_tests'")

        with pytest.raises(CatalogItemsError, match="declares an unusable tool"):
            _bind(_items(_agent()), tools_registry=tools_registry)

    def test_a_name_that_collides_with_a_component_raises(self):
        claimant = {
            "name": "ai-catalog/agents/tester (42)",
            "subagents": [dict(REFERENCE)],
        }

        with pytest.raises(CatalogItemsError, match="collides with a component name"):
            _bind(_items(_agent()), claimant=claimant)
