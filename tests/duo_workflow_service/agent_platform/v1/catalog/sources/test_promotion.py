import pytest

from duo_workflow_service.agent_platform.v1.catalog.sources.promotion import (
    ITEM_PROMPT_VARIABLE,
    AgentComponentTemplate,
    item_prompt_config,
    literal_input,
    promote_claimant,
    synthesized_agent_config,
)

_PROMPT_ID = "workspace_agent_template_prompt"


class _Item:
    name = "workspace/tester"
    description = "Runs tests."
    toolset = ["run_tests"]
    prompt = "Be terse."


class TestTheDefaultTemplate:
    """The component config every workspace agent template is built from."""

    def test_each_call_returns_a_fresh_template(self):
        """Built per item, so a rewrite of one cannot reach the next."""
        first = AgentComponentTemplate.default(_PROMPT_ID)

        assert AgentComponentTemplate.default(_PROMPT_ID) is not first

    def test_it_names_the_prompt_it_was_given(self):
        assert AgentComponentTemplate.default(_PROMPT_ID).prompt_id == _PROMPT_ID

    def test_it_declares_only_the_delegated_goal(self):
        """A subagent reads nothing else from state; its own prompt is injected per item.

        Declared even though ``bind_to_supervisor`` later swaps it, because prompt-variable coverage is
        checked at construction.
        """
        assert AgentComponentTemplate.default(_PROMPT_ID).inputs == [
            {"from": "context:goal", "as": "goal"}
        ]

    def test_ui_log_events_are_forwarded_rather_than_declared(self):
        """``ui_log_events`` is the component's field, so the template only carries it."""
        assert AgentComponentTemplate.default(_PROMPT_ID).model_dump(
            exclude_unset=True
        )["ui_log_events"] == [
            "on_agent_reasoning",
            "on_tool_execution_success",
            "on_tool_execution_failed",
            "on_agent_final_answer",
        ]


class TestAgentComponentTemplate:
    """The template validates what binding touches and forwards the rest verbatim.

    It does not mirror ``AgentComponent``'s config surface, so anything the component accepts must
    reach it unchanged, including keys this model has never heard of.
    """

    MINIMAL = {"type": "AgentComponent", "prompt_id": "workspace_agent_template_prompt"}

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"type": None}, "type"),
            ({"type": "DeterministicStepComponent"}, "AgentComponent"),
            ({"prompt_id": None}, "prompt_id"),
            # `inputs` is the one key binding rewrites, so its shape is checked
            # here rather than assumed.
            ({"inputs": "context:goal"}, "inputs"),
        ],
        ids=["missing_type", "non_agent_type", "missing_prompt_id", "malformed_inputs"],
    )
    def test_invalid_templates_raise(self, overrides, expected):
        template = {
            key: value
            for key, value in {**self.MINIMAL, **overrides}.items()
            if value is not None
        }

        with pytest.raises(ValueError, match=expected):
            AgentComponentTemplate.model_validate(template)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_cycles", 42),
            ("require_tool_approval", True),
            # Falsy values must survive too: they are explicitly set, and dropping
            # one would silently restore the component's default.
            ("require_tool_approval", False),
            ("model_tags", ["fast"]),
            # A field this model has no knowledge of whatsoever.
            ("some_future_agent_field", "value"),
        ],
    )
    def test_component_fields_are_forwarded(self, field, value):
        template = AgentComponentTemplate.model_validate({**self.MINIMAL, field: value})

        assert template.model_dump(exclude_unset=True)[field] == value

    def test_unset_fields_are_left_to_the_component(self):
        """The template invents no defaults for keys it was not given."""
        template = AgentComponentTemplate.model_validate(self.MINIMAL)

        assert set(template.model_dump(exclude_unset=True)) == {"type", "prompt_id"}


class TestLiteralInput:
    def test_it_is_a_literal_with_the_alias(self):
        assert literal_input("Be terse.", "item_prompt") == {
            "from": "Be terse.",
            "as": "item_prompt",
            "literal": True,
        }

    def test_optional_is_set_only_when_asked(self):
        assert literal_input("x", "y", optional=True)["optional"] is True
        assert "optional" not in literal_input("x", "y")


class TestPromoteClaimant:
    def test_it_adds_one_subagent_entry_per_name(self):
        rewritten = {"name": "coordinator", "subagents": [{"name": "static_helper"}]}

        promoted = promote_claimant(rewritten, {"name": "coordinator"}, ["a", "b"])

        assert promoted is rewritten
        assert promoted["subagents"] == [
            {"name": "static_helper"},
            {"name": "a"},
            {"name": "b"},
        ]

    def test_it_adds_the_delegation_events_after_the_declared_ones(self):
        claimant = {"ui_log_events": ["on_agent_final_answer", "on_delegation"]}

        promoted = promote_claimant({}, claimant, ["a"])

        assert promoted["ui_log_events"] == [
            "on_agent_final_answer",
            "on_delegation",
            "on_delegation_returns",
            "on_delegation_error",
        ]


class TestSynthesizedAgentConfig:
    def test_it_fills_in_what_the_item_owns(self):
        config = synthesized_agent_config(
            AgentComponentTemplate.default(_PROMPT_ID), _Item(), {}
        )

        assert config["name"] == "workspace/tester"
        assert config["description"] == "Runs tests."
        assert config["toolset"] == ["run_tests"]
        assert config["prompt_id"] == _PROMPT_ID
        assert config["inputs"][-1] == literal_input("Be terse.", ITEM_PROMPT_VARIABLE)

    @pytest.mark.parametrize(
        "field,template_value",
        [
            ("name", "template"),
            ("description", "From the template."),
            ("toolset", ["other"]),
        ],
    )
    def test_the_item_wins_over_the_template(self, field, template_value):
        template = AgentComponentTemplate.model_validate(
            {"type": "AgentComponent", "prompt_id": _PROMPT_ID, field: template_value}
        )

        config = synthesized_agent_config(template, _Item(), {})

        assert config[field] == getattr(_Item, field)

    @pytest.mark.parametrize("strict", [True, False], ids=["strict", "lenient"])
    def test_strict_validation_is_inherited_from_the_claimant(self, strict):
        config = synthesized_agent_config(
            AgentComponentTemplate.default(_PROMPT_ID),
            _Item(),
            {"strict_validation": strict},
        )

        assert config.get("strict_validation", False) is strict


class TestItemPromptConfig:
    def test_each_call_returns_a_fresh_prompt(self):
        first = item_prompt_config(_PROMPT_ID, "Workspace Agent")

        assert item_prompt_config(_PROMPT_ID, "Workspace Agent") is not first

    def test_it_reads_the_item_prompt_and_the_goal(self):
        prompt = item_prompt_config(_PROMPT_ID, "Workspace Agent")

        assert prompt.prompt_id == _PROMPT_ID
        assert prompt.name == "Workspace Agent"
        assert ITEM_PROMPT_VARIABLE in prompt.prompt_template["system"]
        assert "goal" in prompt.prompt_template["user"]
