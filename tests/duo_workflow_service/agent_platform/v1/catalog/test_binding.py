from unittest.mock import Mock

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.agent_platform.v1.catalog import (
    CatalogItemConfigError,
    CatalogItemError,
    CatalogItemRef,
    CatalogItems,
    CatalogItemsError,
    WorkspaceAgent,
    bind_catalog_items,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

# The reference a flow declares, and repeats on the component that coordinates the
# items. Spelled out rather than imported: it is the flow config's contract.
REFERENCE = {"source": "workspace", "item_type": "agent_template", "item_id": "*"}

# The prompt every item is built against, shipped by the source rather than declared by
# a flow. Spelled out rather than imported: it is the contract a flow relies on.
PROMPT_ID = "workspace_agent_template_prompt"


def _catalog_name(name: str) -> str:
    """The graph name an item ends up with, once namespaced.

    Spelled out rather than imported: the graph node and the ``delegate_task`` enum carry it.
    """
    return f"workspace/agents/{name}"


def _include() -> list[CatalogItemRef]:
    return [CatalogItemRef.model_validate(REFERENCE)]


def _items(*agents: dict) -> CatalogItems:
    """Build a CatalogItems from dict payloads, filling in valid defaults."""
    return CatalogItems(
        workspace_agents=[
            WorkspaceAgent(
                name=entry.get("name", "tester"),
                description=entry.get("description", "Runs tests."),
                toolset=entry.get("toolset", ["run_command"]),
                prompt=entry.get("prompt", "You are a specialist."),
            )
            for entry in agents
        ]
    )


@pytest.fixture(name="expand")
def expand_fixture(mock_tools_registry):
    """Bind catalog items against the shared tools-registry mock.

    The registry is a required argument so no caller can skip tool validation.
    """
    mock_tools_registry.toolset.return_value = Mock(name="toolset")

    def expand(components_config, include, items):
        return bind_catalog_items(
            components_config, include, items, mock_tools_registry
        )

    return expand


def _components(
    static_entries=None, ui_log_events=None, claimed: bool = True
) -> list[dict]:
    """Build a two-component flow whose agent optionally claims the include entry."""
    developer_agent: dict = {
        "name": "developer_agent",
        "type": "AgentComponent",
        "prompt_id": "developer_agent_prompt",
        "inputs": [{"from": "context:goal", "as": "goal"}],
        "toolset": ["read_file", "edit_file"],
    }
    if ui_log_events is not None:
        developer_agent["ui_log_events"] = ui_log_events

    subagents = list(static_entries or [])
    if claimed:
        subagents.append(dict(REFERENCE))
    if subagents:
        developer_agent["subagents"] = subagents

    return [
        {"name": "git_unshallow", "type": "DeterministicStepComponent"},
        developer_agent,
    ]


def _literal(component: dict, alias: str):
    """Return the literal input aliased *alias*, or None when absent."""
    for entry in component.get("inputs", []):
        if (
            isinstance(entry, dict)
            and entry.get("as") == alias
            and entry.get("literal")
        ):
            return entry
    return None


def _by_name(expanded: list[dict]) -> dict[str, dict]:
    return {component["name"]: component for component in expanded}


class TestFlowConfigDeclaration:
    """A flow declares what it accepts in ``include`` and claims it from a component."""

    @staticmethod
    def _flow_config(components=None, **overrides):
        config: dict = {
            "version": "v1",
            "environment": "ambient",
            "flow": {"entry_point": "developer_agent"},
            "components": components
            or [
                {
                    "name": "developer_agent",
                    "type": "AgentComponent",
                    "subagents": [dict(REFERENCE)],
                },
                {"name": "step", "type": "DeterministicStepComponent"},
            ],
            "routers": [{"from": "developer_agent", "to": "end"}],
            "include": [dict(REFERENCE)],
        }
        config.update(overrides)
        return FlowConfig(**config)

    def test_the_include_section_parses_into_references(self):
        config = self._flow_config()

        assert [str(ref) for ref in config.include] == ["workspace/agent_template/*"]

    def test_the_section_is_optional(self):
        config = self._flow_config(
            components=[{"name": "developer_agent", "type": "AgentComponent"}],
            include=None,
        )

        assert config.include is None

    def test_an_entry_no_source_serves_parses_but_does_not_bind(self, expand):
        """A reference is inert data, so the model accepts it and binding refuses it.

        Shipped configs are still covered: ``test_configs`` dry-run compiles each one, which runs binding.
        """
        config = self._flow_config(include=[{**REFERENCE, "item_type": "flow"}])

        with pytest.raises(CatalogItemConfigError, match="not supported yet"):
            expand(config.components, config.include, CatalogItems())

    @pytest.mark.parametrize(
        "entries,is_coordinator",
        [
            ([dict(REFERENCE)], True),
            ([{"name": "static_helper"}, dict(REFERENCE)], True),
            # Declaring it twice still claims the component once, so the coordinator
            # is not mistaken for two.
            ([dict(REFERENCE), dict(REFERENCE)], True),
            # A statically named subagent is the flow author's, not a reference.
            ([{"name": "static_helper"}], False),
            ([], False),
        ],
        ids=[
            "reference",
            "reference_with_static_entry",
            "reference_twice",
            "static_entry_only",
            "no_subagents",
        ],
    )
    def test_which_subagent_entries_claim_the_include_entry(
        self, entries, is_coordinator, expand
    ):
        """A coordinator is rewritten; an unclaimed flow is returned untouched.

        The ``has_workspace_agents`` input is the observable half of that rewrite, and it
        is injected whether or not the request carried items.
        """
        components = [
            {"name": "developer_agent", "type": "AgentComponent", "subagents": entries}
        ]

        expanded = expand(components, _include(), CatalogItems())

        assert (_literal(expanded[0], "has_workspace_agents") is not None) is (
            is_coordinator
        )

    @pytest.mark.parametrize(
        "entry",
        [
            {"source": "workspace"},
            {"source": "workspace", "item_type": "agent_template"},
            {**REFERENCE, "item_type": "nonsense"},
        ],
        ids=["source_only", "no_item_id", "unknown_item_type"],
    )
    def test_a_malformed_reference_is_reported_rather_than_ignored(self, entry, expand):
        """Anything shaped like a reference is parsed, so a typo cannot reach the graph builder."""
        components = [
            {"name": "developer_agent", "type": "AgentComponent", "subagents": [entry]}
        ]

        with pytest.raises(CatalogItemConfigError):
            expand(components, _include(), CatalogItems())

    def test_a_non_dict_subagents_entry_is_not_a_reference(self, expand):
        """YAML can carry ``subagents: ["helper"]``; binding must not choke on it.

        The graph builder is what rejects it, naming the component.
        """
        components = [
            {
                "name": "developer_agent",
                "type": "AgentComponent",
                "subagents": ["helper"],
            }
        ]

        assert expand(components, _include(), CatalogItems()) is components

    def test_multiple_coordinators_are_rejected_for_now(self, expand):
        """Two coordinators would need two instances per item; see binding for why."""
        components = [
            {
                "name": "developer_agent",
                "type": "AgentComponent",
                "subagents": [dict(REFERENCE)],
            },
            {
                "name": "reviewer_agent",
                "type": "AgentComponent",
                "subagents": [dict(REFERENCE)],
            },
        ]
        # The config itself is accepted: the schema is multi-coordinator ready.
        config = self._flow_config(components=components)

        with pytest.raises(CatalogItemConfigError, match="only one component per flow"):
            expand(config.components, config.include, CatalogItems())


def _broken_flow() -> tuple:
    """A component claims a reference the flow never declared: the flow is wrong."""
    return _components(), None, _items({"name": "tester"})


def _unusable_items() -> tuple:
    """An item whose namespaced name a component already holds: the request is wrong."""
    components = _components() + [
        {"name": _catalog_name("tester"), "type": "DeterministicStepComponent"}
    ]
    return components, _include(), _items({"name": "tester"})


class TestErrorsSeparateConfigFromItems:
    """The two failures have different owners, so a caller can tell them apart.

    A config error means the flow is wrong; an items error means this request is.
    """

    def test_a_config_error_is_not_an_items_error(self, expand):
        with pytest.raises(CatalogItemConfigError) as exc_info:
            expand(*_broken_flow())

        assert not isinstance(exc_info.value, CatalogItemsError)

    def test_an_items_error_is_not_a_config_error(self, expand):
        with pytest.raises(CatalogItemsError) as exc_info:
            expand(*_unusable_items())

        assert not isinstance(exc_info.value, CatalogItemConfigError)

    @pytest.mark.parametrize(
        "scenario",
        [_broken_flow, _unusable_items],
        ids=["config_error", "items_error"],
    )
    def test_both_are_catchable_as_one(self, scenario, expand):
        """``CatalogItemError`` is the handle for callers that treat both alike."""
        with pytest.raises(CatalogItemError):
            expand(*scenario())


class TestBindCatalogItems:
    def test_unclaimed_flow_without_items_returns_unchanged(self, expand):
        components = _components(claimed=False)

        with capture_logs() as logs:
            assert expand(components, None, CatalogItems()) is components

        # Nothing was dropped, so nothing to warn about.
        assert not [log for log in logs if log["log_level"] == "warning"]

    @pytest.mark.parametrize(
        "declared", [None, "include"], ids=["no_section", "section_unclaimed"]
    )
    def test_items_for_a_flow_that_does_not_accept_them_are_ignored(
        self, declared, expand
    ):
        """A flow no component claims builds as authored rather than failing.

        The items are dropped with a warning, and synthesize no components.
        """
        include = _include() if declared else None

        with capture_logs() as logs:
            expanded = expand(
                _components(claimed=False), include, _items({"name": "tester"})
            )

        assert {c["name"] for c in expanded} == {"git_unshallow", "developer_agent"}
        # The warning is the only signal the items were discarded.
        assert [log for log in logs if "Ignoring catalog items" in log["event"]]

    @pytest.mark.parametrize(
        "items",
        [CatalogItems(), _items({"name": "tester"})],
        ids=["zero_items", "items"],
    )
    def test_claiming_an_undeclared_reference_raises(self, items, expand):
        """A reference with nothing behind it is a broken flow, not a served request.

        Left alone it would reach the graph builder as a subagent named after nothing.
        """
        with pytest.raises(CatalogItemConfigError, match="does not declare them"):
            expand(_components(), None, items)

    def test_a_claim_differing_only_by_version_is_not_satisfied(self, expand):
        """A claim repeats a declared entry exactly, so a stray version is a mismatch."""
        components = [
            {
                "name": "developer_agent",
                "type": "AgentComponent",
                "subagents": [{**REFERENCE, "version": "1.2.0"}],
            }
        ]

        with pytest.raises(
            CatalogItemConfigError, match="repeat a declared entry exactly"
        ):
            expand(components, _include(), CatalogItems())

    def test_a_claim_naming_other_items_is_not_satisfied(self, expand):
        """The claim must name the same items, not merely reference something.

        Checked before the claim reaches a source, so a mismatch is reported as one.
        """
        components = [
            {
                "name": "developer_agent",
                "type": "AgentComponent",
                "subagents": [{**REFERENCE, "item_type": "flow"}],
            }
        ]

        with pytest.raises(CatalogItemConfigError, match="does not declare them"):
            expand(components, _include(), CatalogItems())

    def test_zero_items_keeps_plain_agent(self, expand):
        by_name = _by_name(expand(_components(), _include(), CatalogItems()))

        # No promotion to supervisor, and no synthesized components.
        assert "subagents" not in by_name["developer_agent"]
        assert set(by_name) == {"git_unshallow", "developer_agent"}

    def test_items_promote_coordinator_to_supervisor(self, expand):
        by_name = _by_name(
            expand(_components(), _include(), _items({"name": "tester"}))
        )
        supervisor = by_name["developer_agent"]

        assert supervisor["subagents"] == [{"name": _catalog_name("tester")}]
        # The coordinator keeps its own authored prompt, toolset and inputs.
        assert supervisor["prompt_id"] == "developer_agent_prompt"
        assert supervisor["toolset"] == ["read_file", "edit_file"]
        assert supervisor["inputs"][0] == {"from": "context:goal", "as": "goal"}

    @pytest.mark.parametrize(
        "items,expected_value",
        [(CatalogItems(), ""), (_items({"name": "tester"}), "true")],
        ids=["zero_items", "items"],
    )
    def test_the_delegation_flag_is_optional(self, items, expected_value, expand):
        """``has_workspace_agents`` reaches every coordinator, so the prompt need not use it.

        ``optional`` is what excuses a prompt that never reads the variable. ``item_prompt`` needs no
        such excuse.
        """
        by_name = _by_name(expand(_components(), _include(), items))

        assert _literal(by_name["developer_agent"], "has_workspace_agents") == {
            "from": expected_value,
            "as": "has_workspace_agents",
            "literal": True,
            "optional": True,
        }

    def test_synthesized_component_comes_from_the_platform_template(self, expand):
        """The component config is the platform's, so a flow cannot reshape a subagent."""
        expanded = expand(
            _components(),
            _include(),
            _items(
                {"name": "tester", "description": "Runs tests.", "toolset": ["grep"]}
            ),
        )
        subagent = _by_name(expanded)[_catalog_name("tester")]

        assert subagent["type"] == "AgentComponent"
        assert subagent["prompt_id"] == PROMPT_ID
        assert subagent["ui_log_events"] == [
            "on_agent_reasoning",
            "on_tool_execution_success",
            "on_tool_execution_failed",
            "on_agent_final_answer",
        ]
        assert subagent["description"] == "Runs tests."
        assert subagent["toolset"] == ["grep"]

    def test_the_item_prompt_is_appended_to_the_template_inputs(self, expand):
        """A subagent gets the delegated goal and its own prompt, and nothing else."""
        expanded = expand(_components(), _include(), _items({"name": "tester"}))
        subagent = _by_name(expanded)[_catalog_name("tester")]

        assert subagent["inputs"] == [
            {"from": "context:goal", "as": "goal"},
            {"from": "You are a specialist.", "as": "item_prompt", "literal": True},
        ]

    def test_each_item_keeps_its_own_prompt(self, expand):
        """Every item is built from its own template, so one cannot pick up another's prompt."""
        expanded = expand(
            _components(),
            _include(),
            _items({"name": "a", "prompt": "A"}, {"name": "b", "prompt": "B"}),
        )
        by_name = _by_name(expanded)

        assert _literal(by_name[_catalog_name("a")], "item_prompt")["from"] == "A"
        assert _literal(by_name[_catalog_name("b")], "item_prompt")["from"] == "B"

    @pytest.mark.parametrize(
        "prompt",
        [
            "You are a test writer.",
            # Template syntax in an item prompt must stay inert: it travels as a
            # literal input value, never as Jinja source.
            "{% extends 'common/developer/system/1.0.0.jinja' %}{{ 7 * 7 }}",
        ],
        ids=["prompt", "template_syntax"],
    )
    def test_item_prompt_travels_as_a_literal_input(self, prompt, expand):
        expanded = expand(
            _components(), _include(), _items({"name": "tester", "prompt": prompt})
        )
        subagent = _by_name(expanded)[_catalog_name("tester")]

        assert _literal(subagent, "item_prompt") == {
            "from": prompt,
            "as": "item_prompt",
            "literal": True,
        }

    @pytest.mark.parametrize(
        "coordinator_flag,expected",
        [(True, True), (None, None)],
        ids=["inherited", "absent"],
    )
    def test_strict_validation_is_inherited_from_the_coordinator(
        self, coordinator_flag, expected, expand
    ):
        """Synthesized components never reach the loop that stamps authored ones.

        Without the flag, validation would not catch a subagent prompt whose variables do not match its inputs.
        """
        components = _components()
        if coordinator_flag is not None:
            _by_name(components)["developer_agent"]["strict_validation"] = (
                coordinator_flag
            )

        expanded = expand(components, _include(), _items({"name": "tester"}))
        subagent = _by_name(expanded)[_catalog_name("tester")]

        assert subagent.get("strict_validation") is expected

    _DELEGATION_EVENTS = [
        "on_delegation",
        "on_delegation_returns",
        "on_delegation_error",
    ]

    @pytest.mark.parametrize(
        "authored,expected",
        [
            (None, _DELEGATION_EVENTS),
            (["on_agent_reasoning"], ["on_agent_reasoning"] + _DELEGATION_EVENTS),
            # Already-declared events are not repeated.
            (
                ["on_agent_reasoning", "on_delegation"],
                ["on_agent_reasoning"] + _DELEGATION_EVENTS,
            ),
        ],
        ids=["author_declared_none", "merged", "not_duplicated"],
    )
    def test_promotion_merges_the_delegation_events(self, authored, expected, expand):
        """Promotion adds the delegation events, keeping the author's own list intact.

        The author cannot declare them: with zero items the coordinator is a plain
        AgentComponent, whose ui_log_events rejects UILogEventsSupervisor members.
        """
        by_name = _by_name(
            expand(
                _components(ui_log_events=authored),
                _include(),
                _items({"name": "tester"}),
            )
        )

        assert by_name["developer_agent"]["ui_log_events"] == expected

    def test_zero_items_leave_ui_log_events_untouched(self, expand):
        """A plain AgentComponent must never receive the delegation events."""
        by_name = _by_name(
            expand(
                _components(ui_log_events=["on_agent_reasoning"]),
                _include(),
                CatalogItems(),
            )
        )

        assert by_name["developer_agent"]["ui_log_events"] == ["on_agent_reasoning"]

    def test_a_full_set_of_items_binds(self, expand):
        """Binding is exercised at the model's ceiling, one component per item."""
        names = [f"a{i}" for i in range(CatalogItems.MAX_WORKSPACE_AGENTS)]

        expanded = expand(
            _components(), _include(), _items(*[{"name": name} for name in names])
        )

        authored = {"git_unshallow", "developer_agent"}
        assert {c["name"] for c in expanded} - authored == {
            _catalog_name(name) for name in names
        }

    def test_static_subagents_compose_with_catalog_items(self, expand):
        by_name = _by_name(
            expand(
                _components(static_entries=[{"name": "static_helper"}]),
                _include(),
                _items({"name": "tester"}),
            )
        )

        assert by_name["developer_agent"]["subagents"] == [
            {"name": "static_helper"},
            {"name": _catalog_name("tester")},
        ]

    def test_static_subagents_survive_zero_items(self, expand):
        by_name = _by_name(
            expand(
                _components(static_entries=[{"name": "static_helper"}]),
                _include(),
                CatalogItems(),
            )
        )

        assert by_name["developer_agent"]["subagents"] == [{"name": "static_helper"}]


class TestCatalogItemValidation:
    @pytest.mark.parametrize(
        "name",
        ["developer_agent", "git_unshallow", "end"],
        ids=["coordinator", "other_component", "reserved_graph_node"],
    )
    def test_namespacing_keeps_items_off_authored_names(self, name, expand):
        """An item asking for a name this flow already uses gets its own, not a collision.

        The item builds and the authored component keeps its name, so no request can displace one.
        """
        by_name = _by_name(expand(_components(), _include(), _items({"name": name})))

        assert set(by_name) == {
            "git_unshallow",
            "developer_agent",
            _catalog_name(name),
        }

    def test_a_name_that_collides_after_namespacing_raises(self, expand):
        """The collision check still guards the one name an item can reach.

        Namespacing clears ordinary component names, but not the namespace itself.
        """
        components = _components()
        components.append(
            {"name": _catalog_name("tester"), "type": "DeterministicStepComponent"}
        )

        with pytest.raises(CatalogItemsError, match="collides with a component name"):
            expand(components, _include(), _items({"name": "tester"}))

    @pytest.mark.parametrize(
        "toolset", [["run_command"], []], ids=["with_tools", "toolless"]
    )
    def test_toolsets_are_resolved_through_the_registry(
        self, toolset, expand, mock_tools_registry
    ):
        """The registry decides, so an empty toolset is resolved rather than bypassed.

        An agent that only reasons is a valid delegation target.
        """
        expanded = expand(
            _components(),
            _include(),
            _items({"name": "summariser", "toolset": toolset}),
        )
        subagent = _by_name(expanded)[_catalog_name("summariser")]

        assert subagent["toolset"] == toolset
        mock_tools_registry.toolset.assert_called_once_with(toolset)

    def test_unknown_tool_names_are_rejected(self, expand, mock_tools_registry):
        """The registry's error is re-raised naming the item that declared the tool."""
        mock_tools_registry.toolset.side_effect = ValueError("unknown tool 'nope'")

        with pytest.raises(
            CatalogItemsError,
            match=r"'workspace/agents/tester'.*unknown tool 'nope'",
        ):
            expand(
                _components(),
                _include(),
                _items({"name": "tester", "toolset": ["nope"]}),
            )
