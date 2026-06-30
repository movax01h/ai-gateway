# pylint: disable=file-naming-for-tests,redefined-outer-name
"""Tests for the distill_agents_md/1.0.0 flow config.

Covers:
- YAML loads without error via FlowConfig
- Graph structure: entry point, component types, router chain
- Envelope schema: version_constraint and required/optional fields
- Agent toolset: glab-only (no gitlab_api_get)
- Prompt timeout raised for multi-step workload
"""

import pytest
import yaml

from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig

FLOW_ID = "distill_agents_md"
FLOW_VERSION = "1.0.0"


@pytest.fixture(scope="module")
def config() -> FlowConfig:
    return FlowConfig.from_yaml_config(FLOW_ID, FLOW_VERSION)


@pytest.fixture(scope="module")
def raw_yaml_config() -> dict:
    config_path = FlowConfig.DIRECTORY_PATH / FLOW_ID / f"{FLOW_VERSION}.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def distill_agent_component(config):
    return config.components[1]


@pytest.fixture(scope="module")
def agent_platform_envelope(config):
    inputs = config.flow.inputs or []
    matches = [i for i in inputs if i.category == "agent_platform_standard_context"]
    assert matches, "agent_platform_standard_context envelope not found in flow.inputs"
    return matches[0]


@pytest.fixture(scope="module")
def distill_prompt(config):
    prompts = config.prompts or []
    matches = [p for p in prompts if p.prompt_id == "distill_agent_prompt"]
    assert matches, "distill_agent_prompt not found in flow prompts"
    return matches[0]


class TestDistillAgentsMdFlowLoads:
    def test_loads_without_error(self, config):  # pylint: disable=redefined-outer-name
        assert config is not None

    def test_schema_version_is_v1(self, config):  # pylint: disable=redefined-outer-name
        assert config.version == "v1"

    def test_environment_is_ambient(self, config):  # pylint: disable=redefined-outer-name
        assert config.environment == "ambient"

    def test_resolved_version(self, config):  # pylint: disable=redefined-outer-name
        assert config.resolved_version == FLOW_VERSION


class TestDistillAgentsMdGraphStructure:
    def test_entry_point_is_git_unshallow(self, config):  # pylint: disable=redefined-outer-name
        assert config.flow.entry_point == "git_unshallow"

    def test_has_exactly_two_components(self, config):  # pylint: disable=redefined-outer-name
        assert len(config.components) == 2

    def test_first_component_is_deterministic_step(self, config):  # pylint: disable=redefined-outer-name
        git_unshallow = config.components[0]
        assert git_unshallow["name"] == "git_unshallow"
        assert git_unshallow["type"] == "DeterministicStepComponent"
        assert git_unshallow["tool_name"] == "run_command"

    def test_second_component_is_agent(self, distill_agent_component):
        assert distill_agent_component["name"] == "distill_agent"
        assert distill_agent_component["type"] == "AgentComponent"

    def test_router_chain_git_unshallow_to_distill_agent(self, config):  # pylint: disable=redefined-outer-name
        router_map = {r["from"]: r["to"] for r in config.routers}
        assert router_map["git_unshallow"] == "distill_agent"

    def test_router_chain_distill_agent_to_end(self, config):  # pylint: disable=redefined-outer-name
        router_map = {r["from"]: r["to"] for r in config.routers}
        assert router_map["distill_agent"] == "end"

    def test_exactly_two_routers(self, config):  # pylint: disable=redefined-outer-name
        assert len(config.routers) == 2


class TestDistillAgentsMdEnvelopeSchema:
    def test_version_constraint_is_set(self, raw_yaml_config):
        inputs = raw_yaml_config.get("flow", {}).get("inputs", [])
        envelope = next(
            (
                i
                for i in inputs
                if i.get("category") == "agent_platform_standard_context"
            ),
            None,
        )
        assert envelope is not None
        assert envelope.get("version_constraint") == "^1.0.0"

    def test_required_fields_present(self, agent_platform_envelope):
        schema = agent_platform_envelope.input_schema
        assert "workload_branch" in schema
        assert "primary_branch" in schema
        assert "session_owner_id" in schema

    def test_service_account_name_is_optional(self, agent_platform_envelope):
        schema = agent_platform_envelope.input_schema
        assert "service_account_name" in schema
        assert schema["service_account_name"].optional is True

    def test_required_fields_are_not_optional(self, agent_platform_envelope):
        schema = agent_platform_envelope.input_schema
        for field in ("workload_branch", "primary_branch", "session_owner_id"):
            assert not schema[field].optional, (
                f"{field} should be required (not optional)"
            )


class TestDistillAgentsMdToolset:
    def test_gitlab_api_get_not_in_toolset(self, distill_agent_component):
        toolset = distill_agent_component.get("toolset", [])
        assert "gitlab_api_get" not in toolset, (
            "gitlab_api_get must not be in the toolset; use glab via run_command instead"
        )

    def test_run_command_in_toolset(self, distill_agent_component):
        toolset = distill_agent_component.get("toolset", [])
        assert "run_command" in toolset

    def test_filesystem_tools_present(self, distill_agent_component):
        toolset = distill_agent_component.get("toolset", [])
        for tool in (
            "read_file",
            "find_files",
            "list_dir",
            "grep",
            "create_file_with_contents",
            "edit_file",
            "mkdir",
        ):
            assert tool in toolset, f"expected filesystem tool '{tool}' in toolset"

    def test_todo_write_in_toolset(self, distill_agent_component):
        toolset = distill_agent_component.get("toolset", [])
        assert "todo_write" in toolset


class TestDistillAgentsMdPrompt:
    def test_prompt_timeout_is_sufficient_for_workload(self, distill_prompt):
        timeout = distill_prompt.params.timeout if distill_prompt.params else None
        assert timeout is not None
        assert timeout >= 300, (
            f"Prompt timeout {timeout}s is too low for a multi-step trace-fetch workload; "
            "expected >= 300s"
        )

    def test_unit_primitive_is_duo_agent_platform(self, distill_prompt):
        assert "duo_agent_platform" in (distill_prompt.unit_primitives or [])

    def test_system_prompt_extends_common_developer(self, distill_prompt):
        system = (distill_prompt.prompt_template or {}).get("system", "")
        assert "common/developer/system/1.0.0.jinja" in system

    def test_user_prompt_contains_goal_variable(self, distill_prompt):
        user = (distill_prompt.prompt_template or {}).get("user", "")
        assert "{{ goal }}" in user
