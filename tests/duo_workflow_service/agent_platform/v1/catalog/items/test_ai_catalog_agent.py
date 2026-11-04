import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog.items import AiCatalogAgent
from duo_workflow_service.agent_platform.v1.catalog.items.ai_catalog_agent import (
    MAX_AI_CATALOG_NAME_LENGTH,
    MAX_AI_CATALOG_PROMPT_LENGTH,
)
from duo_workflow_service.agent_platform.v1.catalog.items.workspace_agent import (
    MAX_PROMPT_LENGTH,
)

REQUIRED_FIELDS = ["item_id", "version", "name", "description", "prompt"]


def _agent(**overrides) -> dict:
    return {
        "item_id": "42",
        "version": "1.0.0",
        "name": "Code Reviewer",
        "description": "Runs tests.",
        "prompt": "You are a specialist.",
        **overrides,
    }


class TestAiCatalogAgent:
    @pytest.mark.parametrize("missing", REQUIRED_FIELDS)
    def test_a_missing_required_field_is_rejected(self, missing):
        payload = {k: v for k, v in _agent().items() if k != missing}

        with pytest.raises(ValidationError, match=missing):
            AiCatalogAgent(**payload)

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_a_blank_required_field_is_rejected(self, field, value):
        with pytest.raises(ValidationError):
            AiCatalogAgent(**_agent(**{field: value}))

    def test_the_toolset_may_be_empty(self):
        assert AiCatalogAgent(**_agent()).toolset == []

    @pytest.mark.parametrize("value", ["", "   "], ids=["empty", "whitespace"])
    def test_a_blank_tool_name_is_rejected(self, value):
        with pytest.raises(ValidationError):
            AiCatalogAgent(**_agent(toolset=["read_file", value]))

    def test_the_component_name_is_the_catalog_name_and_the_item_id(self):
        assert AiCatalogAgent(**_agent()).name == "ai-catalog/agents/Code Reviewer (42)"

    def test_a_name_as_long_as_the_ai_catalog_allows_is_accepted(self):
        agent = AiCatalogAgent(**_agent(name="a" * MAX_AI_CATALOG_NAME_LENGTH))

        assert len(agent.name) == MAX_AI_CATALOG_NAME_LENGTH
        assert agent.name.startswith("ai-catalog/agents/aaaa")
        assert agent.name.endswith(" (42)")

    def test_a_name_longer_than_the_ai_catalog_allows_is_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            AiCatalogAgent(**_agent(name="a" * (MAX_AI_CATALOG_NAME_LENGTH + 1)))

    @pytest.mark.parametrize("character", ["|", ":", "#", "."])
    def test_characters_a_component_name_cannot_hold_are_replaced(self, character):
        agent = AiCatalogAgent(**_agent(name=f"Code{character}Reviewer"))

        assert agent.name == "ai-catalog/agents/Code-Reviewer (42)"

    def test_the_id_and_version_are_stripped_but_not_namespaced(self):
        agent = AiCatalogAgent(**_agent(item_id=" 42 ", version=" 1.0.0 "))

        assert (agent.item_id, agent.version) == ("42", "1.0.0")

    def test_a_prompt_as_long_as_the_ai_catalog_allows_is_accepted(self):
        agent = AiCatalogAgent(**_agent(prompt="a" * MAX_AI_CATALOG_PROMPT_LENGTH))

        assert len(agent.prompt) == MAX_AI_CATALOG_PROMPT_LENGTH

    def test_a_prompt_longer_than_the_ai_catalog_allows_is_rejected(self):
        with pytest.raises(ValidationError, match="prompt"):
            AiCatalogAgent(**_agent(prompt="a" * (MAX_AI_CATALOG_PROMPT_LENGTH + 1)))

    def test_a_prompt_longer_than_the_workspace_ceiling_is_still_accepted(self):
        # GitLab has no field-level limit on a catalog agent's prompt, so a prompt that
        # exceeds the workspace agent's client-text ceiling can be valid in the catalog.
        prompt = "a" * (MAX_PROMPT_LENGTH + 1)

        assert len(AiCatalogAgent(**_agent(prompt=prompt)).prompt) == len(prompt)
