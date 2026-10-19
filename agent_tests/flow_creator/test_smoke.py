"""Smoke tests for the Flow Creator agent.

One test per prompt in the six-case matrix from
https://gitlab.com/gitlab-org/gitlab/-/work_items/604710. Each asserts the agent
answers with exactly one complete flow YAML document that a YAML parser accepts:
no partial snippets, no elided sections.

A second test parses that YAML with the framework's own config schema, which
catches invalid enum values, missing required component fields, and unknown
prompt fields that a plain YAML parse would accept.
"""

import pytest
from pydantic import ValidationError

from agent_tests.flow_creator.cases import CASES
from agent_tests.flow_creator.helpers import (
    agent_components,
    find_truncation_markers,
    missing_top_level_keys,
    supervisor_components,
)

CASE_PARAMS = [
    pytest.param(case, id=case.case_id, marks=pytest.mark.xdist_group(case.case_id))
    for case in CASES
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_produces_one_complete_flow_yaml(generated_flow, case):
    """The agent must answer with exactly one complete, parseable flow YAML."""
    flow = await generated_flow(case)

    # Counted over flow config blocks, not every YAML fence: the prompt asks the
    # agent to state what it verified, so a correct answer may pair one complete
    # flow with a short illustrative snippet. Two *flows* is the real failure.
    blocks = flow.flow_blocks
    assert len(blocks) == 1, (
        f"Expected exactly one flow config document for '{case.title}', found "
        f"{len(blocks)} after {flow.user_turns} user turn(s). A flow must be "
        f"emitted whole, not as snippets the user has to assemble.\n\n"
        f"Response:\n{flow.response[:2000]}"
    )

    assert flow.parse_error is None, (
        f"Generated YAML for '{case.title}' is not a usable flow config: "
        f"{flow.parse_error}\n\nResponse:\n{flow.response[:2000]}"
    )

    config = flow.config
    assert config is not None

    missing = missing_top_level_keys(config)
    assert not missing, (
        f"Generated flow for '{case.title}' is missing required top-level "
        f"section(s) {missing}; the agent must always emit version, environment, "
        f"components, routers, flow and prompts.\n\nYAML:\n{flow.yaml_text}"
    )

    yaml_text = flow.yaml_text
    assert yaml_text is not None
    truncation = find_truncation_markers(yaml_text)
    assert not truncation, (
        f"Generated flow for '{case.title}' elides part of the configuration "
        f"instead of writing it out: {truncation}\n\nYAML:\n{yaml_text}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_generated_yaml_matches_flow_config_schema(generated_flow, case):
    """The generated YAML must satisfy the framework's own config schema."""
    from duo_workflow_service.agent_platform.v1.flows.flow_config import (
        FlowConfig,
        PartialFlowConfig,
    )

    flow = await generated_flow(case)
    config = flow.config
    if config is None:
        pytest.skip(f"no parseable flow YAML to validate: {flow.parse_error}")

    # A chat-partial flow is embedded in Duo Chat, which owns the graph, so its
    # `flow` and `routers` sections are optional.
    config_class = (
        PartialFlowConfig if config.get("environment") == "chat-partial" else FlowConfig
    )

    try:
        config_class(**config)
    except ValidationError as exc:
        pytest.fail(
            f"Generated flow for '{case.title}' does not satisfy "
            f"{config_class.__name__}:\n{exc}\n\nYAML:\n{flow.yaml_text}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASE_PARAMS)
async def test_declares_the_structure_the_case_requires(generated_flow, case):
    """A case that asks for several agents, or for delegation, must get them.

    The numbered hard rules never read either construct, so without this the two hardest cases in the matrix score the
    same whether the agent delegates or answers with a single agent.
    """
    flow = await generated_flow(case)
    config = flow.config
    if config is None:
        pytest.skip(f"no parseable flow YAML to score: {flow.parse_error}")

    if not (case.expects_multiple_agents or case.expects_supervisor):
        pytest.skip("case requires no particular agent structure")

    missing = []
    agents = agent_components(config)
    if case.expects_multiple_agents and len(agents) < 2:
        missing.append(
            "the case asks for several agents, but the flow declares "
            f"{len(agents)} AgentComponent(s)"
        )
    if case.expects_supervisor and not supervisor_components(config):
        missing.append(
            "the case asks for a supervisor, but no AgentComponent declares a "
            "'subagents' list, so nothing can delegate"
        )

    assert not missing, (
        f"The flow generated for '{case.title}' is missing structure the case "
        f"requires:\n"
        + "\n".join(f"  - {item}" for item in missing)
        + f"\n\nYAML:\n{flow.yaml_text}"
    )
