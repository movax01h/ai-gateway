"""Guard for the shipped developer/2.0.0-goal flow config.

Only the non-obvious invariant is tested here; everything visible in the YAML itself is not re-asserted (loading and
dry-run compilation are covered generically in test_registry.py and test_configs.py).
"""

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


def test_verifier_receives_raw_goal_outside_the_delegation_channel():
    """DoD integrity: the verifier must see the user's task verbatim.

    At bind time the supervisor replaces the subagent input aliased ``goal``
    with the delegation prompt, so the raw task must ride on a different
    alias — and the prompt must actually render it, otherwise the verifier
    only ever sees what the developer chooses to tell it and the adversarial
    design is silently defeated.
    """
    config = FlowConfig.from_yaml_config("developer", "2.0.0-goal")
    verifier = next(c for c in config.components if c["name"] == "verifier")

    raw_goal_inputs = {
        inp["as"]
        for inp in verifier["inputs"]
        if inp["from"] == "context:goal" and inp["as"] != "goal"
    }
    assert raw_goal_inputs == {"original_goal"}

    assert config.prompts is not None
    prompt = next(p for p in config.prompts if p.prompt_id == "verifier_prompt")
    assert "{{ original_goal }}" in prompt.prompt_template["system"]
