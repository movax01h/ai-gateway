"""Unit tests for the Flow Registry benchmark's deterministic scoring logic.

Every ``check_rule_N`` function in
``agent_tests/flow_creator/helpers.py`` is a pure function over a
parsed flow config, so it can be exercised without an LLM, an API key, or a
network. That matters because the benchmark itself only runs in a manual,
``allow_failure: true`` CI job: without these tests a regression in a check would
silently change every future score with nothing to catch it.

Each rule is tested from both directions - a config that satisfies it reports no
violations, and a config broken in exactly one way reports exactly that
violation.
"""

import textwrap
from typing import Any

import pytest
import yaml

from agent_tests.flow_creator import helpers

# A flow that satisfies every rule the suite checks. Each test mutates a copy of
# this to break exactly one thing, so an assertion failure points at the rule
# under test rather than at incidental breakage.
VALID_FLOW = """\
version: "v1"
environment: ambient
components:
  - name: "reviewer"
    type: AgentComponent
    prompt_id: "reviewer_prompt"
    inputs:
      - from: "context:goal"
        as: "goal"
      - from: "context:project_id"
        as: "project_id"
    toolset:
      - get_merge_request
    ui_log_events:
      - "on_agent_final_answer"
  - name: "review_gate"
    type: HumanInputComponent
    interaction_type: "approval"
    sends_response_to: "reviewer"
    ui_log_events:
      - "on_user_input_prompt"
      - "on_user_response"
  - name: "publisher"
    type: AgentComponent
    prompt_id: "publisher_prompt"
    inputs:
      - from: "context:reviewer.final_answer"
        as: "review_summary"
      - from: "context:project_id"
        as: "project_id"
    toolset:
      - create_merge_request_note
    ui_log_events:
      - "on_agent_final_answer"
routers:
  - from: "reviewer"
    to: "review_gate"
  - from: "review_gate"
    condition:
      input: "context:review_gate.approval"
      routes:
        "approve": "publisher"
        "modify": "reviewer"
        "reject": "end"
        "default_route": "end"
  - from: "publisher"
    to: "end"
flow:
  entry_point: "reviewer"
prompts:
  - name: Reviewer
    prompt_id: "reviewer_prompt"
    unit_primitives:
      - duo_agent_platform
    prompt_template:
      system: |
        You review merge requests.
        When the summary is written, that is your final answer.
      user: |
        Project ID: {{project_id}}
        Task: {{goal}}
      placeholder: history
  - name: Publisher
    prompt_id: "publisher_prompt"
    unit_primitives:
      - duo_agent_platform
    prompt_template:
      system: |
        You post a summary as a merge request comment.
        When posted, your final answer is the comment URL.
      user: |
        Project ID: {{project_id}}
        Summary: {{review_summary}}
      placeholder: history
"""

# Rule 4 is absent because it has no check_rule_4 function to call. Whether a
# closing sentence is a genuine stopping instruction is a judgement about natural
# language, so it is validated by an LLM in test_rule_4_stopping_instruction
# rather than by a deterministic check.
ALL_DETERMINISTIC_CHECKS = (
    helpers.check_rule_1_project_id_threaded,
    helpers.check_rule_2_flow_inputs_declared,
    helpers.check_rule_3_human_input_wiring,
    helpers.check_rule_5_unit_primitives,
    helpers.check_rule_6_history_placeholder,
    helpers.check_rule_7_aliases_match_placeholders,
    # Not a numbered rule but scored the same way: test_hard_rules fails a case
    # when this reports anything, and rules 4 to 7 silently skip a component whose
    # prompt it cannot resolve.
    helpers.components_missing_inline_prompts,
)

# Sentinel for a parametrized case that removes a field rather than setting one.
DELETE = object()


@pytest.fixture(name="valid_config")
def valid_config_fixture() -> dict[str, Any]:
    return yaml.safe_load(VALID_FLOW)


def component_named(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the component mapping with the given name."""
    for component in config["components"]:
        if component["name"] == name:
            return component
    raise KeyError(name)


def prompt_named(config: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    """Return the inline prompt mapping with the given prompt_id."""
    for prompt in config["prompts"]:
        if prompt["prompt_id"] == prompt_id:
            return prompt
    raise KeyError(prompt_id)


def router_from(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the router leaving the given component."""
    for router in config["routers"]:
        if router["from"] == name:
            return router
    raise KeyError(name)


class TestBaseline:
    @pytest.mark.parametrize(
        "check", ALL_DETERMINISTIC_CHECKS, ids=lambda c: c.__name__
    )
    def test_valid_flow_has_no_violations(self, valid_config, check):
        """The fixture must satisfy every rule, or the tests below prove nothing."""
        assert check(valid_config) == []

    def test_valid_flow_has_every_required_top_level_key(self, valid_config):
        assert helpers.missing_top_level_keys(valid_config) == []


class TestRule1ProjectIdThreaded:
    def test_flags_component_missing_project_id_input(self, valid_config):
        component = component_named(valid_config, "reviewer")
        component["inputs"] = [{"from": "context:goal", "as": "goal"}]

        violations = helpers.check_rule_1_project_id_threaded(valid_config)

        assert len(violations) == 1
        assert "no 'context:project_id' input" in violations[0]
        assert "reviewer" in violations[0]

    def test_flags_prompt_not_referencing_project_id(self, valid_config):
        prompt = prompt_named(valid_config, "reviewer_prompt")
        prompt["prompt_template"]["user"] = "Task: {{goal}}\n"

        violations = helpers.check_rule_1_project_id_threaded(valid_config)

        assert any("does not reference the project ID" in v for v in violations)

    def test_flags_a_component_whose_prompt_is_not_declared(self, valid_config):
        """An unresolvable prompt hides the rule; it is reported, not skipped."""
        component_named(valid_config, "reviewer")["prompt_id"] = "absent_prompt"

        violations = helpers.check_rule_1_project_id_threaded(valid_config)

        assert len(violations) == 1
        assert "is not defined in 'prompts'" in violations[0]
        assert "reviewer" in violations[0]

    @pytest.mark.parametrize(
        "toolset",
        [["read_file", "grep"], ["list_epics"]],
        ids=["no-gitlab-tools", "group-scoped-only"],
    )
    def test_ignores_component_needing_no_project_id(self, valid_config, toolset):
        """Neither toolset intersects GITLAB_API_TOOLS, so the rule must not apply."""
        component = component_named(valid_config, "reviewer")
        component["toolset"] = toolset
        component["inputs"] = [{"from": "context:goal", "as": "goal"}]
        prompt = prompt_named(valid_config, "reviewer_prompt")
        prompt["prompt_template"]["user"] = "Task: {{goal}}\n"

        assert helpers.check_rule_1_project_id_threaded(valid_config) == []

    # One tool per branch of helpers._project_scoped_gitlab_tool_names; more names
    # of the same kind reach no further branch.
    @pytest.mark.parametrize(
        "tool_name,is_project_scoped",
        [
            ("get_merge_request", True),  # explicit project argument
            ("gitlab_api_get", True),  # in _ALWAYS_PROJECT_SCOPED
            ("list_epics", False),  # group-scoped: `id` identifies a group
            ("gitlab__user_search", False),  # instance-scoped
            ("get_current_user", False),  # in _NEVER_PROJECT_SCOPED
        ],
    )
    def test_tool_project_scoping_is_classified(self, tool_name, is_project_scoped):
        assert (tool_name in helpers.GITLAB_API_TOOLS) is is_project_scoped


class TestRule2FlowInputsDeclared:
    BRANCH_SOURCE = "context:inputs.agent_platform_standard_context.primary_branch"

    def use_branch_context(
        self,
        config: dict[str, Any],
        declares: str | None = None,
        category: str = helpers.STANDARD_CONTEXT_CATEGORY,
    ) -> None:
        """Read ``primary_branch``, optionally declaring an input schema for it.

        Shared by every test below so that a change to how branch context is read cannot be applied to the passing case
        alone, leaving a failing case to pass for a reason it was never meant to test.
        """
        component_named(config, "reviewer")["inputs"].append(
            {"from": self.BRANCH_SOURCE, "as": "primary_branch"}
        )
        if declares is None:
            return
        config["flow"]["inputs"] = [
            {"category": category, "input_schema": {declares: {"type": "string"}}}
        ]

    def test_flags_branch_context_without_flow_inputs(self, valid_config):
        self.use_branch_context(valid_config)

        violations = helpers.check_rule_2_flow_inputs_declared(valid_config)

        assert len(violations) == 1
        assert "declares no 'flow.inputs' stanza" in violations[0]

    def test_accepts_branch_context_with_matching_declaration(self, valid_config):
        self.use_branch_context(valid_config, declares="primary_branch")

        assert helpers.check_rule_2_flow_inputs_declared(valid_config) == []

    def test_flags_a_declaration_under_the_wrong_category(self, valid_config):
        """A schema the framework never populates is as absent as no schema."""
        self.use_branch_context(
            valid_config, declares="primary_branch", category="project_context"
        )

        violations = helpers.check_rule_2_flow_inputs_declared(valid_config)

        assert len(violations) == 1
        assert "category with an input_schema" in violations[0]

    def test_flags_declaration_missing_the_referenced_key(self, valid_config):
        self.use_branch_context(valid_config, declares="workload_branch")

        violations = helpers.check_rule_2_flow_inputs_declared(valid_config)

        assert len(violations) == 1
        assert "input_schema only declares" in violations[0]
        assert "primary_branch" in violations[0]

    def test_prose_mentioning_a_branch_key_is_not_an_access(self, valid_config):
        """Only component inputs count; naming a key in prose must not trigger."""
        prompt = prompt_named(valid_config, "reviewer_prompt")
        prompt["prompt_template"]["system"] += "\nAlways target the primary_branch.\n"

        assert helpers.check_rule_2_flow_inputs_declared(valid_config) == []


class TestRule3HumanInputWiring:
    def test_flags_missing_interaction_type(self, valid_config):
        del component_named(valid_config, "review_gate")["interaction_type"]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert any("does not set 'interaction_type'" in v for v in violations)

    def test_flags_missing_ui_log_events(self, valid_config):
        component_named(valid_config, "review_gate")["ui_log_events"] = [
            "on_user_input_prompt"
        ]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert any("is missing ui_log_events" in v for v in violations)
        assert any("on_user_response" in v for v in violations)

    def test_flags_missing_modify_route(self, valid_config):
        del router_from(valid_config, "review_gate")["condition"]["routes"]["modify"]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert len(violations) == 1
        assert "is missing routes" in violations[0]
        assert "modify" in violations[0]

    def test_flags_a_modify_route_to_a_terminal_component(self, valid_config):
        """'modify' means 'go and change it', so it cannot end the flow."""
        router = router_from(valid_config, "review_gate")
        router["condition"]["routes"]["modify"] = "end"

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert len(violations) == 1
        assert "must route back to an agent" in violations[0]

    def test_flags_simple_router_after_a_gate(self, valid_config):
        router = router_from(valid_config, "review_gate")
        del router["condition"]
        router["to"] = "publisher"

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert any("a conditional router is required" in v for v in violations)

    def test_flags_a_gate_with_no_outgoing_router(self, valid_config):
        """A gate the flow cannot leave collects a response and strands it."""
        valid_config["routers"] = [
            r for r in valid_config["routers"] if r["from"] != "review_gate"
        ]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert len(violations) == 1
        assert "has no outgoing router" in violations[0]

    def test_flags_a_gate_without_sends_response_to(self, valid_config):
        del component_named(valid_config, "review_gate")["sends_response_to"]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert len(violations) == 1
        assert "does not set 'sends_response_to'" in violations[0]

    def test_flags_sends_response_to_a_component_that_has_not_run(self, valid_config):
        component_named(valid_config, "review_gate")["sends_response_to"] = "publisher"

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert len(violations) == 1
        assert "sends_response_to" in violations[0]
        assert "publisher" in violations[0]

    def test_reports_entry_point_gate_without_blaming_the_entry_point(
        self, valid_config
    ):
        """A gate that is itself the entry point is a real error, reported as one."""
        valid_config["flow"]["entry_point"] = "review_gate"

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert any("no component runs before the gate" in v for v in violations)
        assert not any("no reachable entry point" in v for v in violations)

    def test_reports_missing_entry_point_distinctly(self, valid_config):
        del valid_config["flow"]["entry_point"]

        violations = helpers.check_rule_3_human_input_wiring(valid_config)

        assert any("declares no 'flow.entry_point'" in v for v in violations)

    def test_flow_without_a_gate_is_not_flagged(self, valid_config):
        valid_config["components"] = [
            c for c in valid_config["components"] if c["name"] != "review_gate"
        ]
        valid_config["routers"] = [
            r for r in valid_config["routers"] if r["from"] != "review_gate"
        ]
        router_from(valid_config, "reviewer")["to"] = "publisher"

        assert helpers.check_rule_3_human_input_wiring(valid_config) == []


class TestRule5UnitPrimitives:
    def test_flags_prompt_without_unit_primitives(self, valid_config):
        del prompt_named(valid_config, "reviewer_prompt")["unit_primitives"]

        violations = helpers.check_rule_5_unit_primitives(valid_config)

        assert len(violations) == 1
        assert "has no 'unit_primitives' key" in violations[0]

    def test_accepts_an_empty_list(self, valid_config):
        prompt_named(valid_config, "reviewer_prompt")["unit_primitives"] = []

        assert helpers.check_rule_5_unit_primitives(valid_config) == []

    def test_flags_non_list_unit_primitives(self, valid_config):
        prompt_named(valid_config, "reviewer_prompt")["unit_primitives"] = "duo"

        violations = helpers.check_rule_5_unit_primitives(valid_config)

        assert len(violations) == 1
        assert "expected a list" in violations[0]


class TestRule6HistoryPlaceholder:
    def test_flags_prompt_without_history_placeholder(self, valid_config):
        del prompt_named(valid_config, "reviewer_prompt")["prompt_template"][
            "placeholder"
        ]

        violations = helpers.check_rule_6_history_placeholder(valid_config)

        assert len(violations) == 1
        assert "does not declare 'placeholder: history'" in violations[0]

    def test_accepts_history_declared_in_a_list(self, valid_config):
        prompt_named(valid_config, "reviewer_prompt")["prompt_template"][
            "placeholder"
        ] = ["history"]

        assert helpers.check_rule_6_history_placeholder(valid_config) == []

    def test_flags_a_prompt_without_a_template_block(self, valid_config):
        """No template is reported, rather than read as an absent placeholder."""
        del prompt_named(valid_config, "reviewer_prompt")["prompt_template"]

        violations = helpers.check_rule_6_history_placeholder(valid_config)

        assert len(violations) == 1
        assert "has no 'prompt_template' block" in violations[0]


class TestRule7AliasesMatchPlaceholders:
    def test_flags_alias_with_no_placeholder(self, valid_config):
        prompt_named(valid_config, "reviewer_prompt")["prompt_template"]["user"] = (
            "Project ID: {{project_id}}\n"
        )

        violations = helpers.check_rule_7_aliases_match_placeholders(valid_config)

        assert len(violations) == 1
        assert "has no '{{goal}}' placeholder" in violations[0]

    def test_flags_inert_angle_placeholder_with_an_explicit_hint(self, valid_config):
        prompt_named(valid_config, "reviewer_prompt")["prompt_template"]["user"] = (
            "Project ID: <<project_id>>\nTask: {{goal}}\n"
        )

        violations = helpers.check_rule_7_aliases_match_placeholders(valid_config)

        assert len(violations) == 1
        assert "does not render" in violations[0]
        assert "{{project_id}}" in violations[0]


class TestComponentsMissingInlinePrompts:
    """TestBaseline covers the pass direction, including that a gate is exempt."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("prompt_id", DELETE, "declares no 'prompt_id'"),
            ("prompt_version", "1.0.0", "must define the prompt inline"),
            ("prompt_id", "absent_prompt", "no matching inline prompt"),
        ],
        ids=["no-prompt-id", "resolves-from-registry", "dangling-prompt-id"],
    )
    def test_flags_a_component_without_a_usable_inline_prompt(
        self, valid_config, field, value, expected
    ):
        component = component_named(valid_config, "publisher")
        if value is DELETE:
            del component[field]
        else:
            component[field] = value

        violations = helpers.components_missing_inline_prompts(valid_config)

        assert len(violations) == 1
        assert "publisher" in violations[0]
        assert expected in violations[0]


class TestSystemPromptsToValidate:
    """Rule 4 skips when this returns nothing, so an empty result passes silently."""

    def test_returns_a_label_and_system_prompt_for_each_inline_prompt(
        self, valid_config
    ):
        prompts = helpers.system_prompts_to_validate(valid_config)

        # prompt_label prefers prompt_id over name, so these are not "Reviewer".
        assert [label for label, _ in prompts] == [
            "reviewer_prompt",
            "publisher_prompt",
        ]
        assert "You review merge requests." in prompts[0][1]

    def test_omits_a_prompt_whose_system_block_is_blank(self, valid_config):
        template = prompt_named(valid_config, "reviewer_prompt")["prompt_template"]
        template["system"] = "   \n"

        prompts = helpers.system_prompts_to_validate(valid_config)

        assert [label for label, _ in prompts] == ["publisher_prompt"]

    def test_returns_nothing_when_the_flow_declares_no_prompts(self, valid_config):
        del valid_config["prompts"]

        assert helpers.system_prompts_to_validate(valid_config) == []


class TestYamlExtraction:
    def test_ignores_a_yaml_snippet_that_is_not_a_flow(self):
        """A flow block is found; an illustrative fragment is not a second flow."""
        response = textwrap.dedent(
            """\
            ```yaml
            version: "v1"
            environment: ambient
            components:
              - name: "a"
            ```

            To add an input, extend that component:

            ```yaml
            inputs:
              - from: "context:goal"
                as: "goal"
            ```
            """
        )

        assert len(helpers.yaml_code_blocks(response)) == 2
        assert len(helpers.flow_config_blocks(response)) == 1

    def test_detects_an_unterminated_fence(self):
        response = '```yaml\nversion: "v1"\ncomponents:\n'

        assert helpers.has_unterminated_code_fence(response)

    @pytest.mark.parametrize(
        "text",
        [
            "components:\n  # ... rest of the components\n",
            "components:\n  # (unchanged)\n",
        ],
    )
    def test_finds_truncation_markers(self, text):
        assert helpers.find_truncation_markers(text)

    def test_clean_yaml_has_no_truncation_markers(self):
        assert helpers.find_truncation_markers(VALID_FLOW) == []

    def test_finds_no_blocks_in_a_response_without_any_fence(self):
        """Real model output sometimes answers in prose, with no fenced block."""
        response = "I need the project ID before I can write the flow."

        assert helpers.yaml_code_blocks(response) == []
        assert helpers.flow_config_blocks(response) == []

    def test_reports_a_yaml_parse_error(self):
        config, error = helpers.parse_flow_yaml('version: "v1"\n  bad: indent\n')

        assert config is None
        assert error is not None

    def test_parses_valid_yaml_without_an_error(self):
        config, error = helpers.parse_flow_yaml(VALID_FLOW)

        assert error is None
        assert config is not None
        assert config["version"] == "v1"
        assert config["environment"] == "ambient"
        assert [c["name"] for c in config["components"]] == [
            "reviewer",
            "review_gate",
            "publisher",
        ]


class TestMissingTopLevelKeys:
    @pytest.mark.parametrize("key", helpers.REQUIRED_TOP_LEVEL_KEYS)
    def test_each_required_key_is_reported_when_absent(self, valid_config, key):
        del valid_config[key]

        assert helpers.missing_top_level_keys(valid_config) == [key]

    def test_every_structural_key_is_required(self):
        """The test above iterates this tuple, so it cannot catch a key dropped from it."""
        assert set(helpers.REQUIRED_TOP_LEVEL_KEYS) == {
            "version",
            "environment",
            "components",
            "routers",
            "flow",
            "prompts",
        }
