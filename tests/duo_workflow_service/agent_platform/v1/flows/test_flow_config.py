# pylint: disable=too-many-lines
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import patch

import pytest
import yaml
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.components import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    FlowConfigMetadata,
    PartialFlowConfig,
    _safe_resolve,
    list_configs,
    load_component_class,
)
from duo_workflow_service.agent_platform.v1.routers.base import BaseRouter


class TestFlowConfig:
    """Test FlowConfig class functionality."""

    def test_input_json_schemas_by_category_with_no_inputs(self):
        """Test input_json_schemas_by_category returns empty dict when no inputs defined."""
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        assert not result

    def test_input_json_schemas_by_category_with_inputs_none(self):
        """Test input_json_schemas_by_category returns empty dict when inputs is None."""
        config_data = {
            "flow": {"entry_point": "agent", "inputs": None},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        assert not result

    def test_input_json_schemas_by_category_single_category_single_field(self):
        """Test input_json_schemas_by_category with single category and single field."""
        config_data = {
            "flow": {
                "entry_point": "agent",
                "inputs": [
                    {
                        "category": "user_input",
                        "input_schema": {
                            "message": {"type": "string", "description": "User message"}
                        },
                    }
                ],
            },
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": True,
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "User message"}
                },
                "required": ["message"],
            }
        }

        assert result == expected

    def test_input_json_schemas_by_category_multiple_categories_multiple_fields(self):
        """Test input_json_schemas_by_category with multiple categories."""
        config_data = {
            "flow": {
                "entry_point": "agent",
                "inputs": [
                    {
                        "category": "user_input",
                        "input_schema": {
                            "message": {"type": "string", "description": "User message"}
                        },
                    },
                    {
                        "category": "system_config",
                        "input_schema": {
                            "timeout": {
                                "type": "number",
                                "format": "float",
                                "description": "Request timeout in seconds",
                            },
                            "debug_mode": {
                                "type": "boolean",
                                "description": "Enable debug logging",
                            },
                        },
                    },
                ],
            },
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": True,
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "User message"}
                },
                "required": ["message"],
            },
            "system_config": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": True,
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "format": "float",
                        "description": "Request timeout in seconds",
                    },
                    "debug_mode": {
                        "type": "boolean",
                        "description": "Enable debug logging",
                    },
                },
                "required": ["timeout", "debug_mode"],
            },
        }

        assert result == expected

    def test_input_json_schemas_by_category_excludes_none_values(self):
        """Test that None values are excluded from the schema properties."""
        config_data = {
            "flow": {
                "entry_point": "agent",
                "inputs": [
                    {
                        "category": "user_input",
                        "input_schema": {
                            "message": {
                                "type": "string",
                                "description": "User message",
                                "format": None,  # This should be excluded
                            },
                            "optional_field": {
                                "type": "string"
                                # description and format are None by default, should be excluded
                            },
                        },
                    }
                ],
            },
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": True,
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "User message"},
                    "optional_field": {"type": "string"},
                },
                "required": ["message", "optional_field"],
            }
        }

        assert result == expected

    def test_input_json_schemas_by_category_empty_input_list(self):
        """Test input_json_schemas_by_category with empty inputs list."""
        config_data = {
            "flow": {"entry_point": "agent", "inputs": []},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        assert not result

    def test_flowconfig_creation_valid_data(self):
        """Test creating FlowConfig with valid data."""
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [
                {
                    "name": "agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config = FlowConfig(**config_data)

        assert config.flow.entry_point == "agent"
        assert len(config.components) == 1
        assert config.components[0]["name"] == "agent"
        assert len(config.routers) == 1
        assert config.environment == "ambient"
        assert config.version == "v1"

    def test_flowconfig_creation_missing_required_fields(self):
        """Test FlowConfig creation fails with missing required fields."""
        incomplete_data = {
            "flow": {"entry_point": "agent"},
            "components": [],
            # Missing routers, environment, version
        }

        with pytest.raises(ValidationError):
            FlowConfig(**incomplete_data)

    def test_flowconfig_from_yaml_config_success(self, tmp_path):
        """Test loading YAML config from file successfully."""
        config_data = {
            "flow": {"entry_point": "test_agent"},
            "components": [
                {
                    "name": "test_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "test_agent", "to": "end"}],
            "environment": "chat",
            "version": "v1",
        }

        config_file = tmp_path / "config" / "1.0.0.yml"
        config_file.parent.mkdir()

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config("config")

        assert config.flow.entry_point == "test_agent"
        assert config.environment == "chat"
        assert config.version == "v1"
        assert config.resolved_version == "1.0.0"

    def test_flowconfig_from_yaml_config_file_not_found(self):
        """Test loading YAML config raises ValueError for missing flow."""
        with pytest.raises(ValueError, match="No version matching"):
            FlowConfig.from_yaml_config("nonexistent")

    def test_flowconfig_from_yaml_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises YAMLError."""
        config_file = tmp_path / "invalid_config" / "1.0.0.yml"
        config_file.parent.mkdir()

        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            with pytest.raises(yaml.YAMLError) as exc_info:
                FlowConfig.from_yaml_config("invalid_config")

        assert "Error parsing YAML file" in str(exc_info.value)

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "config/../../etc/passwd",
            "/etc/config/absolute",
        ],
    )
    def test_flowconfig_from_yaml_config_path_traversal_protection(
        self, tmp_path, malicious_path
    ):
        """Test that path traversal attempts are blocked."""
        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            with pytest.raises(ValueError, match="Path traversal detected"):
                FlowConfig.from_yaml_config(malicious_path)

    @pytest.mark.parametrize(
        "obfuscated_path",
        [
            "%2e%2e/etc/passwd",  # URL-encoded ../
            "..%2fetc%2fpasswd",  # URL-encoded /
        ],
    )
    def test_flowconfig_from_yaml_config_obfuscated_paths_cannot_load(
        self, tmp_path, obfuscated_path
    ):
        """Test that obfuscated path traversal attempts cannot load config data."""
        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            with pytest.raises((ValueError, FileNotFoundError)):
                FlowConfig.from_yaml_config(obfuscated_path)

    def test_flowconfig_from_yaml_config_symlink_outside_base_rejected(self, tmp_path):
        """Test that a symlink pointing outside the base directory is rejected."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()
        config_dir = base_dir / "my_flow"
        config_dir.mkdir()
        # real_file is outside base_dir — this must be rejected
        real_file = tmp_path / "secret.yml"
        real_file.write_text("secret: content")
        symlink_yml = config_dir / "1.0.0.yml"
        symlink_yml.symlink_to(real_file)

        with patch.object(FlowConfig, "DIRECTORY_PATH", base_dir):
            with pytest.raises(ValueError, match="No version matching"):
                FlowConfig.from_yaml_config("my_flow")

    def test_flowconfig_from_yaml_config_circular_symlink_raises_value_error(
        self, tmp_path
    ):
        """Test that a circular symlink raises ValueError, not OSError.

        Path.resolve() raises OSError (errno 40: Too many levels of symbolic links) for circular symlinks.
        _safe_resolve() must convert this to ValueError so callers see a consistent exception type and the error does
        not propagate as an unhandled 500 to the API layer.
        """
        base_dir = tmp_path / "configs"
        base_dir.mkdir()
        config_dir = base_dir / "my_flow"
        config_dir.mkdir()
        # Create a circular symlink: a.yml -> b.yml -> a.yml
        symlink_a = config_dir / "a.yml"
        symlink_b = config_dir / "b.yml"
        symlink_a.symlink_to(symlink_b)
        symlink_b.symlink_to(symlink_a)

        with pytest.raises(ValueError, match="Symlink resolution failed"):
            _safe_resolve(symlink_a, base_dir)

    def test_flowconfig_from_yaml_config_circular_symlink_in_flow_dir_skipped(
        self, tmp_path
    ):
        """Test that from_yaml_config skips circular symlinks when building available list.

        A circular symlink within base_path must not cause an unhandled OSError; it should be silently excluded from the
        available versions list.
        """
        config_data = {
            "flow": {"entry_point": "test_agent"},
            "components": [
                {
                    "name": "test_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "test_agent", "to": "end"}],
            "environment": "chat",
            "version": "v1",
        }
        base_dir = tmp_path / "configs"
        base_dir.mkdir()
        config_dir = base_dir / "my_flow"
        config_dir.mkdir()

        # Create a valid config file
        valid_file = config_dir / "1.0.0.yml"
        valid_file.write_text(yaml.dump(config_data))

        # Create a circular symlink alongside the valid file
        symlink_a = config_dir / "circular_a.yml"
        symlink_b = config_dir / "circular_b.yml"
        symlink_a.symlink_to(symlink_b)
        symlink_b.symlink_to(symlink_a)

        with patch.object(FlowConfig, "DIRECTORY_PATH", base_dir):
            # Should succeed, loading the valid 1.0.0.yml and ignoring the circular symlinks
            config = FlowConfig.from_yaml_config("my_flow")
            assert config.flow.entry_point == "test_agent"

    def test_flowconfig_from_yaml_config_symlink_within_base_allowed(self, tmp_path):
        """Test that a symlink pointing within the base directory is allowed."""
        config_data = {
            "flow": {"entry_point": "test_agent"},
            "components": [
                {
                    "name": "test_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "test_agent", "to": "end"}],
            "environment": "chat-partial",
            "version": "v1",
        }
        # Create the canonical config file
        canonical_dir = tmp_path / "canonical_flow"
        canonical_dir.mkdir()
        canonical_file = canonical_dir / "2.0.0.yml"
        with open(canonical_file, "w") as f:
            yaml.dump(config_data, f)

        # Create a symlink from another flow dir pointing to the canonical file
        alias_dir = tmp_path / "alias_flow"
        alias_dir.mkdir()
        symlink_yml = alias_dir / "1.0.0.yml"
        symlink_yml.symlink_to(canonical_file)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config("alias_flow")
            assert config.flow.entry_point == "test_agent"

    @pytest.mark.parametrize(
        "safe_path",
        [
            "valid_config",
            "config_name",
            "test-config",
            "config_123",
        ],
    )
    def test_flowconfig_from_yaml_config_safe_paths_allowed(self, tmp_path, safe_path):
        """Test that legitimate flow names are allowed through security checks."""
        config_data = {
            "flow": {"entry_point": "test_agent"},
            "components": [
                {
                    "name": "test_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "test_agent", "to": "end"}],
            "environment": "chat-partial",
            "version": "v1",
        }

        config_path = tmp_path / safe_path / "1.0.0.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config(safe_path)
            assert config.flow.entry_point == "test_agent"


class TestToConfig:
    """Test FlowConfig.to_config() and the completion PartialFlowConfig overrides it with."""

    @staticmethod
    def _make_partial_config(**overrides) -> PartialFlowConfig:
        return PartialFlowConfig(
            version="v1",
            environment="chat-partial",
            components=[{"name": "chat_agent", "type": "AgentComponent"}],
            **overrides,
        )

    def test_a_full_config_is_already_complete(self):
        config = FlowConfig(
            flow=FlowConfigMetadata(entry_point="agent"),
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[{"from": "agent", "to": "end"}],
            environment="ambient",
            version="v1",
        )

        assert config.to_config() is config

    def test_a_partial_config_completes_into_a_full_one(self):
        assert type(self._make_partial_config().to_config()) is FlowConfig

    @pytest.mark.parametrize(
        "overrides", [{}, {"routers": []}], ids=["omitted", "declared empty"]
    )
    def test_routers_default_to_the_entry_component_routing_to_end(self, overrides):
        completed = self._make_partial_config(**overrides).to_config()

        assert completed.routers == [{"from": "chat_agent", "to": "end"}]

    def test_declared_routers_are_kept(self):
        routers = [{"from": "chat_agent", "to": "abort"}]

        assert self._make_partial_config(routers=routers).to_config().routers == routers

    def test_entry_point_defaults_to_the_single_component(self):
        completed = self._make_partial_config().to_config()

        assert completed.flow == FlowConfigMetadata(
            entry_point="chat_agent", inputs=None
        )

    def test_entry_point_default_keeps_declared_inputs(self):
        flow = FlowConfigMetadata(
            inputs=[{"category": "file", "input_schema": {"path": {"type": "string"}}}]
        )

        completed = self._make_partial_config(flow=flow).to_config()

        assert completed.flow.entry_point == "chat_agent"
        assert completed.flow.inputs == flow.inputs

    def test_declared_entry_point_is_kept(self):
        flow = FlowConfigMetadata(entry_point="chat_agent")

        assert self._make_partial_config(flow=flow).to_config().flow is flow

    def test_components_pass_through_untouched(self):
        component = {
            "name": "chat_agent",
            "type": "AgentComponent",
            "pre_approved_tools": ["read_file"],
        }
        config = PartialFlowConfig(
            version="v1", environment="chat-partial", components=[dict(component)]
        )

        assert config.to_config().components == [component]

    def test_the_partial_config_is_not_mutated(self):
        config = self._make_partial_config()

        config.to_config()

        assert config.flow is None
        assert config.routers is None


class TestLoadComponentClass:
    """Test load_component_class function with ComponentRegistry."""

    def test_load_component_class_success(self, component_registry_instance_type):
        """Test loading existing component class successfully from registry."""

        class TestComponent(BaseComponent):
            def attach(self, graph: StateGraph, router: RouterProtocol) -> None: ...

            def __entry_hook__(self) -> Annotated[str, "Components entry node name"]:
                return "mock"

        registry = component_registry_instance_type()
        mock_component_class = TestComponent
        registry.register(mock_component_class, decorators=[])

        result = load_component_class("TestComponent")

        assert result is mock_component_class

    @pytest.mark.usefixtures("component_registry_instance_type")
    def test_load_component_class_not_found_raises_error(self):
        """Test loading non-existent component class raises TypeError."""
        with pytest.raises(KeyError):
            load_component_class("NonExistentComponent")


class TestVersionConstraintsByCategory:
    """Test BaseFlowConfig.version_constraints_by_category()."""

    @staticmethod
    def _make_config(**flow_kwargs):
        return FlowConfig(
            flow={"entry_point": "agent", **flow_kwargs},
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[{"from": "agent", "to": "end"}],
            environment="ambient",
            version="v1",
        )

    def test_no_inputs_returns_empty_dict(self):
        """version_constraints_by_category returns empty dict when no inputs defined."""
        config = self._make_config()
        assert config.version_constraints_by_category() == {}

    def test_with_constraint(self):
        """version_constraints_by_category returns the declared constraint."""
        config = self._make_config(
            inputs=[
                {
                    "category": "agent_platform_standard_context",
                    "version_constraint": "^1.0.0",
                    "input_schema": {"primary_branch": {"type": "string"}},
                }
            ]
        )
        assert config.version_constraints_by_category() == {
            "agent_platform_standard_context": "^1.0.0"
        }

    def test_without_constraint_returns_none(self):
        """version_constraints_by_category returns None for inputs without a constraint."""
        config = self._make_config(
            inputs=[
                {
                    "category": "file",
                    "input_schema": {
                        "contents": {"type": "string"},
                        "file_name": {"type": "string"},
                    },
                }
            ]
        )
        assert config.version_constraints_by_category() == {"file": None}

    def test_mixed_constrained_and_unconstrained(self):
        """version_constraints_by_category handles a mix of constrained and unconstrained inputs."""
        config = self._make_config(
            inputs=[
                {
                    "category": "agent_platform_standard_context",
                    "version_constraint": "^1.1.0",
                    "input_schema": {"primary_branch": {"type": "string"}},
                },
                {
                    "category": "file",
                    "input_schema": {
                        "contents": {"type": "string"},
                        "file_name": {"type": "string"},
                    },
                },
            ]
        )
        assert config.version_constraints_by_category() == {
            "agent_platform_standard_context": "^1.1.0",
            "file": None,
        }


class TestShouldAutoInjectMcpTools:
    """Test BaseFlowConfig.should_auto_inject_mcp_tools()."""

    @staticmethod
    def _make_config(environment: Literal["ambient", "chat", "chat-partial"]):
        return FlowConfig(
            flow=FlowConfigMetadata(entry_point="agent"),
            components=[{"name": "agent", "type": "AgentComponent"}],
            routers=[{"from": "agent", "to": "end"}],
            environment=environment,
            version="v1",
        )

    @pytest.mark.parametrize(
        "environment,expected",
        [
            ("chat", True),
            ("ambient", False),
            ("chat-partial", True),
        ],
    )
    def test_should_auto_inject_mcp_tools_by_environment(self, environment, expected):
        """should_auto_inject_mcp_tools returns True for chat and chat-partial environments."""
        config = self._make_config(environment)
        assert config.should_auto_inject_mcp_tools() == expected


class TestListConfigs:
    """Test list_configs function functionality."""

    @pytest.fixture
    def sample_config_data(self):
        """Sample config data for testing."""
        return {
            "flow": {"entry_point": "test_agent"},
            "components": [
                {
                    "name": "test_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                }
            ],
            "routers": [{"from": "test_agent", "to": "end"}],
            "environment": "chat",
            "version": "v1",
        }

    def test_list_configs_empty_directory(self, tmp_path):
        """Test list_configs returns empty list when no config files exist."""
        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()
            assert not result

    def test_list_configs_single_valid_config(self, tmp_path, sample_config_data):
        """Test list_configs returns single config when one valid file exists."""
        config_file = tmp_path / "test_config" / "1.0.0.yml"
        config_file.parent.mkdir()
        with open(config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == "test_config"
        assert result[0]["flow_version"] == "1.0.0"
        assert result[0]["version"] == "v1"
        assert result[0]["environment"] == "chat"
        assert "config" in result[0]
        config_data = json.loads(result[0]["config"])
        assert config_data["version"] == "v1"
        assert config_data["environment"] == "chat"

    @pytest.mark.parametrize(
        "flow_name",
        [
            "simple",
            "complex-name",
            "config_123",
            "nested_config_file",
        ],
    )
    def test_list_configs_various_flow_names(
        self, tmp_path, sample_config_data, flow_name
    ):
        """Test list_configs handles various valid flow directory names."""
        config_file = tmp_path / flow_name / "1.0.0.yml"
        config_file.parent.mkdir()
        with open(config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == flow_name

    def test_list_configs_multiple_valid_configs(self, tmp_path):
        """Test list_configs returns multiple configs when multiple valid files exist."""
        configs_data = [
            {
                "flow": {"entry_point": "agent1"},
                "components": [{"name": "agent1", "type": "AgentComponent"}],
                "routers": [{"from": "agent1", "to": "end"}],
                "environment": "chat",
                "version": "v1",
            },
            {
                "flow": {"entry_point": "agent2"},
                "components": [{"name": "agent2", "type": "AgentComponent"}],
                "routers": [{"from": "agent2", "to": "end"}],
                "environment": "chat-partial",
                "version": "v1",
            },
            {
                "flow": {"entry_point": "agent3"},
                "components": [{"name": "agent3", "type": "AgentComponent"}],
                "routers": [{"from": "agent3", "to": "end"}],
                "environment": "ambient",
                "version": "v1",
            },
        ]

        for i, config_data in enumerate(configs_data):
            config_file = tmp_path / f"config_{i}" / "1.0.0.yml"
            config_file.parent.mkdir()
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 3
        names = {config["flow_identifier"] for config in result}
        assert names == {"config_0", "config_1", "config_2"}

        versions = {config["version"] for config in result}
        assert versions == {"v1"}

        environments = {config["environment"] for config in result}
        assert environments == {"chat", "chat-partial", "ambient"}

    @pytest.mark.parametrize(
        "invalid_content",
        [
            # Invalid YAML syntax
            "invalid: yaml: content: [unclosed",
            # Malformed YAML with unmatched brackets
            "flow:\n  - entry_point: test\n    missing_bracket: [",
            # Invalid YAML structure
            "- invalid\n  - structure\n    - with: mixed types",
        ],
    )
    def test_list_configs_skips_invalid_yaml_files(
        self, tmp_path, sample_config_data, invalid_content
    ):
        """Test list_configs skips files with invalid YAML and continues processing."""
        # Create one valid config
        valid_config_file = tmp_path / "valid_config" / "1.0.0.yml"
        valid_config_file.parent.mkdir()
        with open(valid_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Create one invalid config
        invalid_config_file = tmp_path / "invalid_config" / "1.0.0.yml"
        invalid_config_file.parent.mkdir()
        with open(invalid_config_file, "w") as f:
            f.write(invalid_content)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        # Should only return the valid config, skipping the invalid one
        assert len(result) == 1
        assert result[0]["flow_identifier"] == "valid_config"

    def test_list_configs_skips_files_with_io_errors(
        self, tmp_path, sample_config_data
    ):
        """Test list_configs skips files that cause IO errors and continues processing."""
        # Create one valid config
        valid_config_file = tmp_path / "valid_config" / "1.0.0.yml"
        valid_config_file.parent.mkdir()
        with open(valid_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Create another valid config
        another_config_file = tmp_path / "another_config" / "1.0.0.yml"
        another_config_file.parent.mkdir()
        with open(another_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Mock IOError for one specific file
        original_open = open

        def mock_open(file, *args, **kwargs):
            if "another_config" in str(file) and "r" in args:
                raise IOError("Mocked IO error")
            return original_open(file, *args, **kwargs)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            with patch("builtins.open", side_effect=mock_open):
                result = list_configs()

        # Should only return the config that didn't have IO error
        assert len(result) == 1
        assert result[0]["flow_identifier"] == "valid_config"

    def test_list_configs_ignores_non_yml_files(self, tmp_path, sample_config_data):
        """Test list_configs only processes versioned 1.0.0.yml files, ignoring other file types."""
        # Create valid YAML config in versioned subdirectory
        yml_config = tmp_path / "config" / "1.0.0.yml"
        yml_config.parent.mkdir()
        with open(yml_config, "w") as f:
            yaml.dump(sample_config_data, f)

        # Create files with other extensions
        other_files = [
            ("config.yaml", yaml.dump(sample_config_data, default_flow_style=False)),
            ("config.json", json.dumps(sample_config_data)),
            ("config.txt", "some text content"),
            ("README.md", "# README"),
            ("config.py", "config = {}"),
        ]

        for filename, content in other_files:
            file_path = tmp_path / filename
            with open(file_path, "w") as f:
                f.write(content)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        # Should only return the .yml file
        assert len(result) == 1
        assert result[0]["flow_identifier"] == "config"

    def test_list_configs_json_serialization(self, tmp_path):
        """Test that list_configs properly serializes complex config structures to JSON."""
        complex_config = {
            "flow": {"entry_point": "complex_agent"},
            "components": [
                {
                    "name": "complex_agent",
                    "type": "AgentComponent",
                    "inputs": ["context:goal"],
                    "nested_config": {
                        "params": {"value": 42, "enabled": True},
                        "list_param": [1, 2, "string", {"nested": "object"}],
                    },
                }
            ],
            "routers": [
                {
                    "from": "complex_agent",
                    "to": "end",
                    "conditions": ["param1", "param2"],
                }
            ],
            "environment": "ambient",
            "version": "v1",
        }

        config_file = tmp_path / "complex_config" / "1.0.0.yml"
        config_file.parent.mkdir()
        with open(config_file, "w") as f:
            yaml.dump(complex_config, f)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        config_result = result[0]

        # Verify that the config JSON is valid and contains expected data
        parsed_config = json.loads(config_result["config"])
        assert parsed_config["components"][0]["nested_config"]["params"]["value"] == 42
        assert (
            parsed_config["components"][0]["nested_config"]["params"]["enabled"] is True
        )
        assert parsed_config["components"][0]["nested_config"]["list_param"] == [
            1,
            2,
            "string",
            {"nested": "object"},
        ]
        assert parsed_config["routers"][0]["conditions"] == ["param1", "param2"]

    def test_list_configs_handles_missing_optional_fields(self, tmp_path):
        """Test list_configs works with configs that have only required fields."""
        minimal_config = {
            "flow": {"entry_point": "minimal_agent"},
            "components": [{"name": "minimal_agent", "type": "AgentComponent"}],
            "routers": [{"from": "minimal_agent", "to": "end"}],
            "environment": "ambient",
            "version": "v1",
        }

        config_file = tmp_path / "minimal_config" / "1.0.0.yml"
        config_file.parent.mkdir()
        with open(config_file, "w") as f:
            yaml.dump(minimal_config, f)

        with (
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == "minimal_config"
        assert result[0]["version"] == "v1"
        assert result[0]["environment"] == "ambient"

        # Verify JSON config is valid
        parsed_config = json.loads(result[0]["config"])
        assert parsed_config == minimal_config

    def test_list_configs_multiple_versions_of_same_flow(self, tmp_path):
        """Test list_configs discovers all versions of the same flow."""
        config_v1 = {
            "flow": {"entry_point": "agent"},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "chat",
            "version": "v1",
        }
        config_v2 = {
            "flow": {"entry_point": "agent_v2"},
            "components": [{"name": "agent_v2", "type": "AgentComponent"}],
            "routers": [{"from": "agent_v2", "to": "end"}],
            "environment": "chat",
            "version": "v1",
        }

        flow_dir = tmp_path / "my_flow"
        flow_dir.mkdir()
        (flow_dir / "1.0.0.yml").write_text(yaml.dump(config_v1))
        (flow_dir / "2.0.0.yml").write_text(yaml.dump(config_v2))

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            result = list_configs()

        assert len(result) == 2
        versions = {r["flow_version"] for r in result}
        assert versions == {"1.0.0", "2.0.0"}
        assert all(r["flow_identifier"] == "my_flow" for r in result)

    def test_list_configs_ignores_non_yml_in_flow_dirs(
        self, tmp_path, sample_config_data
    ):
        """Test list_configs only picks up .yml files inside flow directories."""
        flow_dir = tmp_path / "my_flow"
        flow_dir.mkdir()
        (flow_dir / "1.0.0.yml").write_text(yaml.dump(sample_config_data))
        (flow_dir / "README.md").write_text("# Notes")
        (flow_dir / "config.json").write_text("{}")

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_version"] == "1.0.0"


class TestFromYamlConfigVersionResolution:
    """Test from_yaml_config with version constraints (semver resolution)."""

    @pytest.fixture()
    def flow_dir(self, tmp_path):
        """Create a flow directory with multiple version files."""
        d = tmp_path / "myflow"
        d.mkdir()
        return d

    @staticmethod
    def _write_config(path, environment="ambient"):
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": environment,
            "version": "v1",
        }
        path.write_text(yaml.dump(config_data))

    def test_exact_version(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "2.0.0.yml")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow", "2.0.0")
        assert result.environment == "ambient"
        assert result.resolved_version == "2.0.0"

    def test_caret_constraint_picks_highest(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.0.0.yml", environment="ambient")
        self._write_config(flow_dir / "1.2.0.yml", environment="chat")
        self._write_config(flow_dir / "2.0.0.yml", environment="chat-partial")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow", "^1.0.0")
        assert result.environment == "chat"
        assert result.resolved_version == "1.2.0"

    def test_tilde_constraint(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.0.0.yml", environment="ambient")
        self._write_config(flow_dir / "1.0.5.yml", environment="chat")
        self._write_config(flow_dir / "1.1.0.yml", environment="chat-partial")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow", "~1.0.0")
        assert result.environment == "chat"
        assert result.resolved_version == "1.0.5"

    def test_no_compatible_version_raises(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.0.0.yml")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            with pytest.raises(ValueError, match="No version matching"):
                FlowConfig.from_yaml_config("myflow", "^2.0.0")

    def test_range_constraint_excludes_prerelease(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.0.0.yml", environment="ambient")
        self._write_config(flow_dir / "1.1.0-rc1.yml", environment="chat")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow", "^1.0.0")
        assert result.environment == "ambient"
        assert result.resolved_version == "1.0.0"

    def test_exact_prerelease_still_works(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.1.0-rc1.yml", environment="chat")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow", "1.1.0-rc1")
        assert result.environment == "chat"
        assert result.resolved_version == "1.1.0-rc1"

    def test_default_version_used_when_none(self, tmp_path, flow_dir):
        self._write_config(flow_dir / "1.0.0.yml")
        with patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path):
            result = FlowConfig.from_yaml_config("myflow")
        assert result.environment == "ambient"
        assert result.resolved_version == "1.0.0"


class TestShippedConfigRouterIndentation:
    """Regression guard for the 2026-06-23 resolve_sast_vulnerability incident.

    A ``default_route`` indented as a sibling of ``routes:`` (a child of
    ``condition`` rather than an entry inside ``routes``) is silently dropped:
    the parser in ``flows/base.py`` only iterates ``condition["routes"]``. The
    affected router then has no default branch and raises ``KeyError`` when the
    routing input matches none of the explicit routes. This walks every shipped
    v1 flow config and fails if any router reintroduces that misindentation.
    """

    def test_no_router_defines_default_route_outside_routes(self):
        config_dir = FlowConfig.DIRECTORY_PATH
        offenders = []

        for config_file in sorted(config_dir.glob("*/*.yml")):
            config = yaml.safe_load(config_file.read_text())
            for router in (config or {}).get("routers", []) or []:
                condition = router.get("condition")
                if (
                    isinstance(condition, dict)
                    and BaseRouter.DEFAULT_ROUTE in condition
                ):
                    offenders.append(
                        f"{config_file.relative_to(config_dir)} "
                        f"(router from '{router.get('from')}'): "
                        f"'{BaseRouter.DEFAULT_ROUTE}' is a sibling of 'routes:' and "
                        "will be ignored — nest it inside 'routes:'."
                    )

        assert not offenders, "Misindented default_route(s) found:\n" + "\n".join(
            offenders
        )


class TestShippedConfigOptionalStateLookups:
    """Regression guard for the no-MR ``KeyError`` in the fix_pipeline flows.

    ``IOKey.value_from_state`` resolves non-optional subkeys with ``current[key]``,
    which raises when the key is absent. Collectors omit whole context envelopes
    (rather than sending them with blank values), so any config that reads a
    subkey of an envelope that is not always present must mark the lookup
    ``optional: True``. Without it the flow raises ``KeyError`` on the outer
    envelope name — for fix_pipeline this happened in the first router, before
    any agent ran, surfacing only as a generic error message.

    ``merge_request`` is the known-omitted envelope: a pipeline can fail with no
    associated merge request. This walks every shipped v1 flow config and fails
    if a router condition or component input reads into it without ``optional``.
    """

    ALWAYS_OPTIONAL_PREFIX = "context:inputs.merge_request."

    def test_router_conditions_reading_merge_request_are_optional(self):
        """Every router condition reading into merge_request must be optional."""
        offenders = []

        for config_file in sorted(FlowConfig.DIRECTORY_PATH.glob("*/*.yml")):
            config = yaml.safe_load(config_file.read_text()) or {}
            for router in config.get("routers") or []:
                condition = router.get("condition")
                if not isinstance(condition, dict):
                    continue

                spec = condition.get("input")
                # The plain-string form has no way to express optionality, so it
                # is always unsafe for a possibly-absent envelope.
                if isinstance(spec, str):
                    if spec.startswith(self.ALWAYS_OPTIONAL_PREFIX):
                        offenders.append(
                            f"{config_file.parent.name}/{config_file.stem} (router from "
                            f"'{router.get('from')}'): reads '{spec}' as a plain "
                            "string; use the mapping form with optional: True"
                        )
                elif isinstance(spec, dict):
                    from_ = spec.get("from", "")
                    if from_.startswith(self.ALWAYS_OPTIONAL_PREFIX) and not spec.get(
                        "optional"
                    ):
                        offenders.append(
                            f"{config_file.parent.name}/{config_file.stem} (router from "
                            f"'{router.get('from')}'): reads '{from_}' without "
                            "optional: True"
                        )

        assert not offenders, (
            "Router condition(s) may raise KeyError when the merge_request "
            "context is absent:\n" + "\n".join(offenders)
        )

    def test_component_inputs_reading_merge_request_are_optional(self):
        """Every component input reading into merge_request must be optional."""
        offenders = []

        for config_file in sorted(FlowConfig.DIRECTORY_PATH.glob("*/*.yml")):
            config = yaml.safe_load(config_file.read_text()) or {}
            for component in config.get("components") or []:
                for input_spec in component.get("inputs") or []:
                    if not isinstance(input_spec, dict):
                        continue
                    from_ = input_spec.get("from", "")
                    if from_.startswith(
                        self.ALWAYS_OPTIONAL_PREFIX
                    ) and not input_spec.get("optional"):
                        offenders.append(
                            f"{config_file.parent.name}/{config_file.stem} "
                            f"(component '{component.get('name')}'): reads "
                            f"'{from_}' without optional: True"
                        )

        assert not offenders, (
            "Component input(s) may raise KeyError when the merge_request "
            "context is absent:\n" + "\n".join(offenders)
        )


class TestSastFixValidationIsDeterministic:
    """Regression guard for https://gitlab.com/gitlab-com/request-for-help/-/work_items/5216.

    ``validate_fix_has_changes`` is the only thing standing between an agent that did not
    write a fix and a merge request with an empty diff. It used to be an AgentComponent
    whose prompt read the agent's self-reported ``files_modified`` before consulting git,
    so an agent that claimed success without calling ``edit_file`` was waved through. The
    question it answers has one exact answer, so it must stay a deterministic step whose
    only input is the literal git command, never anything the agent produced.
    """

    FLOW = "resolve_sast_vulnerability"
    COMPONENT = "validate_fix_has_changes"
    PROCEED_ROUTE = "Exit code: 0\nproceed"

    def _shipped_versions(self):
        config_dir = FlowConfig.DIRECTORY_PATH / self.FLOW
        return sorted(path.stem for path in config_dir.glob("*.yml"))

    def _component(self, version):
        config = FlowConfig.from_yaml_config(self.FLOW, version)
        return next(c for c in config.components if c["name"] == self.COMPONENT)

    def test_every_shipped_version_validates_deterministically(self):
        assert self._shipped_versions(), "no shipped configs found for the flow"

        for version in self._shipped_versions():
            component = self._component(version)

            assert component["type"] == "DeterministicStepComponent", version
            assert component["tool_name"] == "run_command", version

            for step_input in component.get("inputs", []):
                assert step_input.get("literal"), (
                    f"{version}: the check must read git and nothing else; "
                    "an agent self-report is not evidence that a fix reached disk"
                )

    def test_every_shipped_version_checks_the_working_tree(self):
        for version in self._shipped_versions():
            command = next(
                step_input["from"]
                for step_input in self._component(version)["inputs"]
                if step_input["as"] == "command"
            )

            assert "git status --porcelain" in command, version
            # printf, not echo: a trailing newline would not match the route key below.
            assert "printf proceed" in command, version
            assert "printf no_changes" in command, version

    def test_every_shipped_version_routes_anything_but_proceed_to_end(self):
        for version in self._shipped_versions():
            config = FlowConfig.from_yaml_config(self.FLOW, version)
            router = next(r for r in config.routers if r["from"] == self.COMPONENT)
            routes = router["condition"]["routes"]

            assert router["condition"]["input"] == (
                f"context:{self.COMPONENT}.tool_responses"
            ), version
            assert routes[self.PROCEED_ROUTE] == "commit_changes", version
            assert routes[BaseRouter.DEFAULT_ROUTE] == "end", version
            assert set(routes) == {self.PROCEED_ROUTE, BaseRouter.DEFAULT_ROUTE}, (
                f"{version}: only an exact clean-exit 'proceed' may reach commit_changes"
            )


class TestSastPushPrecedesMergeRequestCreation:
    """Regression guard for https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2814.

    ``push_and_create_mr`` used to push and call ``create_merge_request`` in one response,
    so a push rejected by a server-side push rule still produced a merge request — one
    pointing at a branch that never received the commit. The push now owns a step of its
    own, and only its reported success reaches the merge request step.
    """

    FLOW = "resolve_sast_vulnerability"
    PUSH = "push_commits"
    CREATE_MR = "push_and_create_mr"
    # Anything that can run a command can push, which is what has to stay out of the
    # merge request step.
    COMMAND_TOOLS = {"run_command", "run_git_command"}

    def _shipped_versions(self):
        config_dir = FlowConfig.DIRECTORY_PATH / self.FLOW
        return sorted(path.stem for path in config_dir.glob("*.yml"))

    def _config(self, version):
        return FlowConfig.from_yaml_config(self.FLOW, version)

    def _component(self, version, name):
        return next(c for c in self._config(version).components if c["name"] == name)

    def test_every_shipped_version_pushes_in_its_own_agent_step(self):
        assert self._shipped_versions(), "no shipped configs found for the flow"

        for version in self._shipped_versions():
            component = self._component(version, self.PUSH)

            # An agent, not a deterministic step: recovering from a rejected push means
            # reading the remote's error and amending the commit message.
            assert component["type"] == "AgentComponent", version
            assert component["prompt_id"] == "resolve_sast_vulnerability_push", version
            assert self.COMMAND_TOOLS & set(component["toolset"]), version

    def test_every_shipped_version_creates_the_mr_only_after_a_successful_push(self):
        for version in self._shipped_versions():
            config = self._config(version)
            routers = {r["from"]: r for r in config.routers}

            assert (
                routers["commit_changes"]["condition"]["routes"]["success"] == self.PUSH
            ), version

            push_router = routers[self.PUSH]["condition"]
            assert push_router["input"] == f"context:{self.PUSH}.final_answer.status", (
                version
            )
            assert push_router["routes"]["success"] == self.CREATE_MR, version
            assert push_router["routes"][BaseRouter.DEFAULT_ROUTE] == "end", version

    def test_every_shipped_version_cannot_push_from_the_mr_step(self):
        """Left in the toolset, the prompt's own history of pushing there would repeat itself."""
        for version in self._shipped_versions():
            toolset = set(self._component(version, self.CREATE_MR)["toolset"])

            assert "create_merge_request" in toolset, version
            assert not self.COMMAND_TOOLS & toolset, (
                f"{version}: {self.CREATE_MR} can still run git, so it can still push "
                "a branch the flow has not confirmed"
            )


class TestSastResolvesTargetBranchBeforeGitOperations:
    """Regression guard for tracked-ref target branch wiring in version 1.0.2."""

    FLOW = "resolve_sast_vulnerability"
    VERSION = "1.0.2"
    COMPONENT = "resolve_target_branch"
    BRANCH_SOURCE = f"context:{COMPONENT}.tool_responses"

    def _config(self):
        return FlowConfig.from_yaml_config(self.FLOW, self.VERSION)

    def _component(self, name):
        return next(c for c in self._config().components if c["name"] == name)

    def test_target_branch_is_resolved_deterministically(self):
        component = self._component(self.COMPONENT)

        assert component["type"] == "DeterministicStepComponent"
        assert component["tool_name"] == "resolve_vulnerability_target_branch"

    def test_target_branch_resolution_must_succeed_before_git_operations(self):
        router = next(r for r in self._config().routers if r["from"] == self.COMPONENT)
        condition = router["condition"]

        assert condition["input"] == f"context:{self.COMPONENT}.execution_result"
        assert condition["routes"]["success"] == "ensure_clean_git_state"
        assert condition["routes"][BaseRouter.DEFAULT_ROUTE] == "end"

    @pytest.mark.parametrize(
        "component_name,input_alias",
        [
            ("ensure_clean_git_state", "default_branch"),
            ("create_repository_branch", "ref"),
            ("push_and_create_mr", "default_branch"),
        ],
    )
    def test_resolved_branch_is_used_consistently(self, component_name, input_alias):
        component = self._component(component_name)

        assert {
            "from": self.BRANCH_SOURCE,
            "as": input_alias,
        } in component["inputs"]


class TestSastHidesTheSandboxsOwnFiles:
    """Regression guard for https://gitlab.com/gitlab-org/gitlab/-/work_items/629525.

    The flow runs under ``@anthropic-ai/sandbox-runtime``, which protects a set of
    sensitive filenames by mounting read-only ``/dev/null`` or empty directories over
    them. The paths are relative and the working directory is the checkout, so they
    land in the repository: untracked, unremovable ("Device or resource busy") and
    unstageable ("can only add regular files"). ``git add .`` exited 128 on one, the
    commit exited 1 with the fix left unstaged, and because a non-zero exit returns as
    ordinary text rather than raising, both were recorded as successes. The flow
    pushed an empty branch and opened a merge request asserting the fix.

    Taking them out of git's view is what makes every later step's reading of the
    working tree honest again, ``validate_fix_has_changes`` included.
    """

    FLOW = "resolve_sast_vulnerability"
    COMPONENT = "exclude_pre_existing_files"
    # The executor wraps output as "Exit code: N\n<output>", so a clean exit and the
    # step's own marker are both required before the flow may continue.
    ROUTE_KEY = "Exit code: 0\nexcluded"
    # Named only so the tests can assert the flow does NOT hardcode them.
    SANDBOX_FILENAMES = (
        ".bashrc",
        ".bash_profile",
        ".zshrc",
        ".gitconfig",
        ".gitmodules",
        ".ripgreprc",
        ".mcp.json",
        ".vscode",
        ".idea",
        ".claude",
    )

    def _versions(self):
        """Versions carrying the step, discovered rather than listed.

        A hardcoded list would silently stop covering a version added later.
        """
        config_dir = FlowConfig.DIRECTORY_PATH / self.FLOW
        return [
            version
            for version in sorted(path.stem for path in config_dir.glob("*.yml"))
            if any(
                c["name"] == self.COMPONENT
                for c in FlowConfig.from_yaml_config(self.FLOW, version).components
            )
        ]

    def _config(self, version):
        return FlowConfig.from_yaml_config(self.FLOW, version)

    def _component(self, version):
        return next(
            c for c in self._config(version).components if c["name"] == self.COMPONENT
        )

    @staticmethod
    def _targets(router):
        if "to" in router:
            return {router["to"]}
        return set(router["condition"]["routes"].values())

    def _command(self, version):
        return next(
            step_input["from"]
            for step_input in self._component(version)["inputs"]
            if step_input["as"] == "command"
        )

    def test_at_least_one_version_hides_them(self):
        """Stops every guard below from passing by having nothing to check."""
        assert self._versions(), (
            "no shipped version hides the sandbox's files, so nothing stops them "
            "reaching `git add` and leaving the fix unstaged"
        )

    def test_it_runs_before_the_fix_and_cannot_fail_open(self):
        """Recorded after the fix, the fix itself would count as pre-existing.

        And everything downstream reads git's view of the tree, so a failure here has to stop the flow rather than let
        the sandbox's files reach a commit.
        """
        for version in self._versions():
            config = self._config(version)

            # Component order in the YAML is declaration order, not execution order,
            # so reachability has to come off the routers: nothing may enter
            # `execute_fix` except this step.
            feeders = {
                router["from"]
                for router in config.routers
                if "execute_fix" in self._targets(router)
            }
            assert feeders == {self.COMPONENT}, (
                f"{version}: {sorted(feeders)} can reach execute_fix, so the fix can "
                "run while the sandbox's files are still visible to git"
            )

            component = self._component(version)
            assert component["type"] == "DeterministicStepComponent", version
            assert component["tool_name"] == "run_command", version
            for step_input in component.get("inputs", []):
                assert step_input.get("literal"), version

            router = next(r for r in config.routers if r["from"] == self.COMPONENT)
            assert router["condition"]["input"] == (
                f"context:{self.COMPONENT}.tool_responses"
            ), (
                f"{version}: routing on execution_result would gate on whether the "
                "tool raised, and DeterministicStepNode reports success for a command "
                "that merely exits non-zero, which is this issue's own failure mode"
            )
            assert router["condition"]["routes"][self.ROUTE_KEY] == "execute_fix", (
                version
            )
            assert router["condition"]["routes"][BaseRouter.DEFAULT_ROUTE] == "end", (
                version
            )

    def test_it_hides_them_from_git_not_from_the_project(self):
        """``.gitignore`` is a tracked file and would land in the commit.

        The list is derived rather than written down: the sandbox's own list is
        versioned in the runtime image and was bumped three times in one month, so a
        hardcoded copy goes stale silently and fails the way the original bug failed.
        The block is delimited and rewritten because local clones are reused between
        runs, and a blind append would grow the file every time.
        """
        for version in self._versions():
            command = self._command(version)

            assert ".git/info/exclude" in command, version
            assert ".gitignore" not in command, (
                f"{version}: writing .gitignore would commit the sandbox's filenames "
                "into the user's repository"
            )
            assert "ls-files --others --exclude-standard" in command, (
                f"{version}: the ignore list is not derived from the working tree"
            )
            for filename in self.SANDBOX_FILENAMES:
                assert filename not in command, (
                    f"{version}: {filename} is hardcoded; the next sandbox release "
                    "that adds a filename silently stops being handled"
                )
            assert "/^# duo-flow-begin$/,/^# duo-flow-end$/d" in command, (
                f"{version}: the previous block is never stripped, so every run on a "
                "reused clone appends another copy"
            )
            assert command.startswith("set -e;"), (
                f"{version}: without it the command runs on past a failure and still "
                "reaches its final printf, reporting the route key either way"
            )
            assert command.count("ls-files --others --exclude-standard") >= 2, (
                f"{version}: the step never re-reads the tree, so an entry that was "
                "written but does not match still reports success"
            )

    @staticmethod
    def _git(repo, *args):
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=30,
        )

    def _repo(self, tmp_path):
        """A checkout holding a tracked dotfile, as a real project would."""
        repo = tmp_path / "checkout"
        repo.mkdir(parents=True)
        self._git(repo, "init", "--quiet", ".")
        self._git(repo, "config", "user.email", "test@example.com")
        self._git(repo, "config", "user.name", "test")
        (repo / "app.py").write_text("vulnerable\n")
        (repo / ".gitlab-ci.yml").write_text("stages: [test]\n")
        self._git(repo, "add", "--all")
        self._git(repo, "commit", "--quiet", "--message", "base")
        return repo

    # Built from the real binaries' locations rather than the inherited PATH, so the
    # shim takes precedence over git without the test reading the environment.
    _COREUTILS = ("sh", "sed", "mkdir", "touch", "mv")

    def _run(self, command, repo, shim=None):
        env = None
        if shim is not None:
            real = sorted(
                {str(Path(shutil.which(tool)).parent) for tool in self._COREUTILS}
            )
            env = {"PATH": os.pathsep.join([str(shim), *real]), "HOME": str(repo)}
        return subprocess.run(
            ["sh", "-c", command],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=env,
        )

    def _untracked(self, repo):
        return subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
    def test_it_leaves_git_blind_to_everything_that_was_already_there(self, tmp_path):
        """The outcome, not the wiring: `git add .` must stage the fix and nothing else.

        ``.weird[1].json`` is here because the paths are written into a gitignore
        file, where ``[`` is a character class. An unescaped entry is still written
        successfully and still leaves the file visible, so only checking the
        outcome catches it.
        """
        for version in self._versions():
            repo = self._repo(tmp_path / version)
            for name in (".bashrc", ".zshrc", ".weird[1].json"):
                (repo / name).write_text("sandbox\n")
            (repo / ".claude").mkdir()
            (repo / ".claude" / "agents").write_text("sandbox\n")

            result = self._run(self._command(version), repo)

            assert result.returncode == 0, f"{version}: {result.stderr}"
            assert result.stdout == "excluded", version
            assert self._untracked(repo) == [], (
                f"{version}: the sandbox's files are still visible to git, so "
                "`git add .` will fail on them and leave the fix unstaged"
            )

            (repo / "app.py").write_text("fixed\n")
            (repo / ".gitlab-ci.yml").write_text("stages: [test, sast]\n")
            (repo / "helper.py").write_text("new\n")
            self._git(repo, "add", ".")
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.split()
            assert sorted(staged) == [".gitlab-ci.yml", "app.py", "helper.py"], version

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
    def test_it_reports_failure_rather_than_claiming_success(self, tmp_path):
        """A command that prints its marker anyway is indistinguishable from one that worked.

        The router can only be as honest as the exit code and the output it reads, and
        ``DeterministicStepNode`` will not notice a non-zero exit on its own.
        """
        for version in self._versions():
            repo = self._repo(tmp_path / version)
            failing_bin = tmp_path / version / "bin"
            failing_bin.mkdir(parents=True)
            broken_git = failing_bin / "git"
            broken_git.write_text("#!/bin/sh\nexit 7\n")
            broken_git.chmod(0o755)

            result = self._run(self._command(version), repo, shim=str(failing_bin))

            assert result.returncode != 0, (
                f"{version}: a broken git still exits 0, so the router sees a clean "
                "run and the flow continues with the sandbox's files in the tree"
            )
            assert "excluded" not in result.stdout, (
                f"{version}: the step printed its route key despite failing"
            )
