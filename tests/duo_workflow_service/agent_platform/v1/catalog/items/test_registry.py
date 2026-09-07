import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.catalog import (
    CatalogItems,
    CatalogItemsError,
)


def _agent(**overrides) -> dict:
    return {
        "name": "tester",
        "description": "Runs tests.",
        "prompt": "You are a specialist.",
        **overrides,
    }


class TestCatalogItems:
    def test_defaults_to_no_items(self):
        """A request that carries no catalog section builds the flow exactly as authored."""
        assert CatalogItems().workspace_agents == []

    @pytest.mark.parametrize(
        "names",
        [("tester", "tester"), ("tester", "workspace/agents/tester")],
        ids=["identical", "differ_only_by_the_prefix"],
    )
    def test_duplicate_agent_names_raise(self, names):
        """Each name becomes one graph component and one delegate_task choice.

        Uniqueness is judged after namespacing, which is where a collision lands.
        """
        with pytest.raises(ValidationError, match="Duplicate workspace agent name"):
            CatalogItems(workspace_agents=[_agent(name=name) for name in names])

    def test_the_ceiling_itself_is_accepted(self):
        payload = [
            _agent(name=f"a{i}") for i in range(CatalogItems.MAX_WORKSPACE_AGENTS)
        ]

        assert len(CatalogItems(workspace_agents=payload).workspace_agents) == len(
            payload
        )

    def test_one_item_over_the_ceiling_raises(self):
        payload = [
            _agent(name=f"a{i}") for i in range(CatalogItems.MAX_WORKSPACE_AGENTS + 1)
        ]

        with pytest.raises(ValidationError, match="Too many workspace agents"):
            CatalogItems(workspace_agents=payload)


class TestFromPayload:
    """Converting a client payload reports every failure as one typed error.

    The constructor reports through pydantic, which discards the type of anything a
    validator raises, so a caller converting external input cannot tell an item
    problem from any other ``ValidationError``.
    """

    def test_a_valid_payload_builds_items(self):
        items = CatalogItems.from_payload({"workspace_agents": [_agent()]})

        assert [agent.name for agent in items.workspace_agents] == [
            "workspace/agents/tester"
        ]

    def test_an_absent_agents_key_builds_no_items(self):
        assert CatalogItems.from_payload({}).workspace_agents == []

    @pytest.mark.parametrize(
        "payload",
        [
            {"workspace_agents": [_agent(name="  ")]},
            {"workspace_agents": [{"name": "tester"}]},
            {"workspace_agents": [_agent(toolset=["read_file", " "])]},
            {"workspace_agents": [_agent(), _agent()]},
            {
                "workspace_agents": [
                    _agent(name=f"a{i}")
                    for i in range(CatalogItems.MAX_WORKSPACE_AGENTS + 1)
                ]
            },
            {"workspace_agents": "not-a-list"},
        ],
        ids=[
            "blank_name",
            "missing_required_fields",
            "blank_tool_name",
            "duplicate_names",
            "over_the_ceiling",
            "malformed_shape",
        ],
    )
    def test_every_failure_is_an_items_error(self, payload):
        """Field constraints, cross-item rules and shape errors all arrive as one type."""
        with pytest.raises(CatalogItemsError):
            CatalogItems.from_payload(payload)

    def test_the_failing_field_survives_the_conversion(self):
        """Flattening must not cost the caller the location, which is what it acts on."""
        with pytest.raises(CatalogItemsError, match=r"workspace_agents\.0\.name"):
            CatalogItems.from_payload({"workspace_agents": [_agent(name="  ")]})

    def test_the_validation_error_is_chained(self):
        """The structured pydantic error stays reachable for logging."""
        with pytest.raises(CatalogItemsError) as exc_info:
            CatalogItems.from_payload({"workspace_agents": [_agent(name="  ")]})

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_the_constructor_still_reports_through_pydantic(self):
        """``from_payload`` is the boundary door; the constructor stays pydantic-native."""
        with pytest.raises(ValidationError):
            CatalogItems(workspace_agents=[_agent(), _agent()])
