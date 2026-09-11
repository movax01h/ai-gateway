"""Hard rule checks on the flow YAML the agent generates.

One test per rule from
https://gitlab.com/gitlab-org/gitlab/-/work_items/604709, scored against the
same generated YAML the smoke tests use. Every rule except Rule 4 is a
deterministic check on the parsed YAML structure; Rule 4 (an explicit stopping
instruction) cannot be verified structurally, so it is the only one delegated to
the LLM judge.

A rule that does not apply to a case is skipped rather than scored, so the pass
rate is not diluted. When the matrix says a case must exercise a construct - a
HITL gate, GitLab API tools, branch context - its absence is a failure, not a
skip. The rules that read prompts (4 to 7) apply the same principle to prompt
resolution: a generated custom flow cannot resolve prompts from the AI Gateway
registry, so a component whose prompt is not defined inline is scored as a
failure rather than skipped.
"""

import pytest

from agent_tests.flow_creator.cases import CASES
from agent_tests.flow_creator.helpers import (
    HUMAN_INPUT_TYPE,
    branch_context_inputs,
    check_rule_1_project_id_threaded,
    check_rule_2_flow_inputs_declared,
    check_rule_3_human_input_wiring,
    check_rule_5_unit_primitives,
    check_rule_6_history_placeholder,
    check_rule_7_aliases_match_placeholders,
    components,
    components_missing_inline_prompts,
    components_of_type,
    gitlab_api_tools_in,
    inline_prompts,
    input_variables,
    rendered_template_for,
    system_prompts_to_validate,
)
from agent_tests.llm_validator import validate_with_llm

CASE_PARAMS = [
    pytest.param(case, id=case.case_id, marks=pytest.mark.xdist_group(case.case_id))
    for case in CASES
]

STOPPING_INSTRUCTION_CRITERION = (
    "The prompt ends with an explicit stopping instruction: a sentence stating "
    "when the agent is finished and what its final answer should be, for example "
    "'When the summary is posted, your final answer is the comment URL. No "
    "further steps are needed after that.' A prompt that only describes the task "
    "or lists steps, without stating a termination condition, does NOT satisfy "
    "this criterion."
)


async def _flow_and_config(generated_flow, case):
    """Return the generated flow and its parsed config, skipping if there is none."""
    flow = await generated_flow(case)
    config = flow.config
    if config is None:
        pytest.skip(f"no parseable flow YAML to score: {flow.parse_error}")
    return flow, config


def _failure(case, rule, violations, flow):
    return (
        f"{rule} violated by the flow generated for '{case.title}':\n"
        + "\n".join(f"  - {violation}" for violation in violations)
        + f"\n\nYAML:\n{flow.yaml_text}"
    )


def _require_inline_prompts(case, flow, config):
    """Fail when a prompt-bearing component has no inline prompt to check.

    The rules below read prompts, so without this they would skip such a component and report nothing. A generated
    custom flow cannot resolve prompts from the AI Gateway registry, so a component whose prompt is not inline is broken
    output rather than a case the rule does not apply to.
    """
    missing = components_missing_inline_prompts(config)
    if missing:
        pytest.fail(_failure(case, "Inline prompts", missing, flow))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_1_project_id_threaded(generated_flow, case):
    """Rule 1: components calling GitLab API tools must receive ``project_id``."""
    flow, config = await _flow_and_config(generated_flow, case)

    with_api_tools = [
        component for component in components(config) if gitlab_api_tools_in(component)
    ]
    if not with_api_tools:
        if case.expects_gitlab_api_tools:
            pytest.fail(
                f"The flow generated for '{case.title}' calls no GitLab API tools, "
                f"but the case requires them.\n\nYAML:\n{flow.yaml_text}"
            )
        pytest.skip("no component calls a GitLab API tool")

    violations = check_rule_1_project_id_threaded(config)
    assert not violations, _failure(case, "Rule 1", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_2_flow_inputs_declared(generated_flow, case):
    """Rule 2: reading branch context requires a ``flow.inputs`` stanza."""
    flow, config = await _flow_and_config(generated_flow, case)

    if not branch_context_inputs(config):
        if case.expects_branch_context:
            pytest.fail(
                f"The flow generated for '{case.title}' reads no branch context, "
                f"but the case creates a branch and so needs "
                f"'primary_branch' from agent_platform_standard_context.\n\n"
                f"YAML:\n{flow.yaml_text}"
            )
        pytest.skip("flow does not read primary_branch or workload_branch")

    violations = check_rule_2_flow_inputs_declared(config)
    assert not violations, _failure(case, "Rule 2", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_3_human_input_wiring(generated_flow, case):
    """Rule 3: every HumanInputComponent must be completely wired."""
    flow, config = await _flow_and_config(generated_flow, case)

    if not components_of_type(config, HUMAN_INPUT_TYPE):
        if case.expects_hitl:
            pytest.fail(
                f"The flow generated for '{case.title}' has no "
                f"{HUMAN_INPUT_TYPE}, but the case asks for an approval gate.\n\n"
                f"YAML:\n{flow.yaml_text}"
            )
        pytest.skip(f"flow declares no {HUMAN_INPUT_TYPE}")

    violations = check_rule_3_human_input_wiring(config)
    assert not violations, _failure(case, "Rule 3", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_4_stopping_instruction(generated_flow, case, validation_model):
    """Rule 4: every inline system prompt must end with a stopping instruction.

    The only LLM-validated rule: whether a closing sentence is a genuine
    termination condition is a judgement about natural language, not structure.
    """
    flow, config = await _flow_and_config(generated_flow, case)
    _require_inline_prompts(case, flow, config)

    prompts = system_prompts_to_validate(config)
    if not prompts:
        pytest.skip("flow declares no inline system prompts")

    violations = []
    for label, system_prompt in prompts:
        summary = await validate_with_llm(
            system_prompt,
            [STOPPING_INSTRUCTION_CRITERION],
            model=validation_model,
        )
        if not summary.all_passed:
            explanation = "; ".join(
                result.explanation or "no explanation"
                for result in summary.results
                if not result.passed
            )
            violations.append(
                f"prompt '{label}' has no explicit stopping instruction ({explanation})"
            )

    assert not violations, _failure(case, "Rule 4", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_5_unit_primitives(generated_flow, case):
    """Rule 5: every inline prompt must declare ``unit_primitives``."""
    flow, config = await _flow_and_config(generated_flow, case)
    _require_inline_prompts(case, flow, config)

    if not inline_prompts(config):
        pytest.skip("flow declares no inline prompts")

    violations = check_rule_5_unit_primitives(config)
    assert not violations, _failure(case, "Rule 5", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_6_history_placeholder(generated_flow, case):
    """Rule 6: every inline prompt template must declare ``placeholder: history``."""
    flow, config = await _flow_and_config(generated_flow, case)
    _require_inline_prompts(case, flow, config)

    if not inline_prompts(config):
        pytest.skip("flow declares no inline prompts")

    violations = check_rule_6_history_placeholder(config)
    assert not violations, _failure(case, "Rule 6", violations, flow)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_rule_7_aliases_match_placeholders(generated_flow, case):
    """Rule 7: every input ``as:`` alias must appear as a ``{{alias}}`` placeholder.

    Strictly Jinja2: the framework renders templates with Jinja2 and nothing else,
    so the ``<<alias>>`` form the agent's own prompt documents is a violation. It
    substitutes nothing at runtime and produces no error, which is exactly the
    silent failure Rule 7 exists to prevent.
    """
    flow, config = await _flow_and_config(generated_flow, case)
    _require_inline_prompts(case, flow, config)

    scoreable = [
        component
        for component in components(config)
        if input_variables(component) and rendered_template_for(config, component)
    ]
    if not scoreable:
        pytest.skip("no component declares both inputs and an inline template")

    violations = check_rule_7_aliases_match_placeholders(config)
    assert not violations, _failure(case, "Rule 7", violations, flow)
