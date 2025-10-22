import json
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import pytest
import yaml
from langgraph.graph import StateGraph
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components import (
    BaseComponent,
    RouterProtocol,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
    list_configs,
    load_component_class,
)


class TestFlowConfig:
    """Test FlowConfig class functionality."""

    def test_input_json_schemas_by_category_with_no_inputs(self):
        """Test input_json_schemas_by_category returns empty dict when no inputs defined."""
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "remote",
            "version": "experimental",
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
            "environment": "remote",
            "version": "experimental",
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
            "environment": "remote",
            "version": "experimental",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": False,
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
            "environment": "remote",
            "version": "experimental",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": False,
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "User message"}
                },
                "required": ["message"],
            },
            "system_config": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": False,
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
            "environment": "remote",
            "version": "experimental",
        }

        config = FlowConfig(**config_data)
        result = config.input_json_schemas_by_category()

        expected = {
            "user_input": {
                "$schema": "https://json-schema.org/draft/2020-12/schema#",
                "additionalProperties": False,
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
            "environment": "remote",
            "version": "experimental",
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
            "environment": "remote",
            "version": "experimental",
        }

        config = FlowConfig(**config_data)

        assert config.flow.entry_point == "agent"
        assert len(config.components) == 1
        assert config.components[0]["name"] == "agent"
        assert len(config.routers) == 1
        assert config.environment == "remote"
        assert config.version == "experimental"

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
            "environment": "local",
            "version": "experimental",
        }

        config_file = tmp_path / "config.yml"

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config("config")

        assert config.flow.entry_point == "test_agent"
        assert config.environment == "local"
        assert config.version == "experimental"

    def test_flowconfig_from_yaml_config_file_not_found(self):
        """Test loading YAML config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            FlowConfig.from_yaml_config("nonexistent")

        assert "nonexistent file not found" in str(exc_info.value)

    def test_flowconfig_from_yaml_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises YAMLError."""
        config_file = tmp_path / "invalid_config.yml"

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
            "flows/.../.../.../config.yml",
            r"configs\…..\\…..\\system.yml",
            "templates%00../../../../../etc/passwd",
            "flows%2e%2e%2fconfig.yml",
            "configs%252e%252e%252fsystem.yml",
            "templates%c0%ae%c0%ae%c0%afconfig.yml",
            "flows%uff0e%uff0e%u2215config.yml",
            "configs%uff0e%uff0e%u2216system.yml",
            "/etc/config/absolute.yml",
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
        "safe_path",
        [
            "valid_config",
            "config_name",
            "test-config",
            "config_123",
            "nested/config",
            "deeply/nested/config",
        ],
    )
    def test_flowconfig_from_yaml_config_safe_paths_allowed(self, tmp_path, safe_path):
        """Test that legitimate paths are allowed through security checks."""
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
            "environment": "local",
            "version": "experimental",
        }

        # Create nested directory structure if needed
        config_path = tmp_path / f"{safe_path}.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config(safe_path)
            assert config.flow.entry_point == "test_agent"


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

    def test_load_component_class_not_found_raises_error(
        self, component_registry_instance_type  # pylint: disable=unused-argument
    ):
        """Test loading non-existent component class raises TypeError."""
        with pytest.raises(KeyError):
            load_component_class("NonExistentComponent")


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
            "environment": "test",
            "version": "1.0",
        }

    def test_list_configs_empty_directory(self, tmp_path):
        """Test list_configs returns empty list when no config files exist."""
        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()
            assert not result

    def test_list_configs_single_valid_config(self, tmp_path, sample_config_data):
        """Test list_configs returns single config when one valid file exists."""
        config_file = tmp_path / "test_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == "test_config"
        assert result[0]["version"] == "1.0"
        assert result[0]["environment"] == "test"
        assert "config" in result[0]
        config_data = json.loads(result[0]["config"])
        assert config_data["version"] == "1.0"
        assert config_data["environment"] == "test"

    @pytest.mark.parametrize(
        "filename,expected_name",
        [
            ("simple.yml", "simple"),
            ("complex-name.yml", "complex-name"),
            ("config_123.yml", "config_123"),
            ("test.config.yml", "test.config"),
            ("nested_config_file.yml", "nested_config_file"),
        ],
    )
    def test_list_configs_various_filenames(
        self, tmp_path, sample_config_data, filename, expected_name
    ):
        """Test list_configs handles various valid filename patterns."""
        config_file = tmp_path / filename
        with open(config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == expected_name

    def test_list_configs_multiple_valid_configs(self, tmp_path):
        """Test list_configs returns multiple configs when multiple valid files exist."""
        configs_data = [
            {
                "flow": {"entry_point": "agent1"},
                "components": [{"name": "agent1", "type": "AgentComponent"}],
                "routers": [{"from": "agent1", "to": "end"}],
                "environment": "dev",
                "version": "1.0",
            },
            {
                "flow": {"entry_point": "agent2"},
                "components": [{"name": "agent2", "type": "AgentComponent"}],
                "routers": [{"from": "agent2", "to": "end"}],
                "environment": "prod",
                "version": "2.0",
            },
            {
                "flow": {"entry_point": "agent3"},
                "components": [{"name": "agent3", "type": "AgentComponent"}],
                "routers": [{"from": "agent3", "to": "end"}],
                "environment": "staging",
                "version": "1.5",
            },
        ]

        for i, config_data in enumerate(configs_data):
            config_file = tmp_path / f"config_{i}.yml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 3
        names = {config["flow_identifier"] for config in result}
        assert names == {"config_0", "config_1", "config_2"}

        versions = {config["version"] for config in result}
        assert versions == {"1.0", "2.0", "1.5"}

        environments = {config["environment"] for config in result}
        assert environments == {"dev", "prod", "staging"}

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
        valid_config_file = tmp_path / "valid_config.yml"
        with open(valid_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Create one invalid config
        invalid_config_file = tmp_path / "invalid_config.yml"
        with open(invalid_config_file, "w") as f:
            f.write(invalid_content)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
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
        valid_config_file = tmp_path / "valid_config.yml"
        with open(valid_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Create another valid config
        another_config_file = tmp_path / "another_config.yml"
        with open(another_config_file, "w") as f:
            yaml.dump(sample_config_data, f)

        # Mock IOError for one specific file
        original_open = open

        def mock_open(file, *args, **kwargs):
            if str(file).endswith("another_config.yml") and "r" in args:
                raise IOError("Mocked IO error")
            return original_open(file, *args, **kwargs)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            with patch("builtins.open", side_effect=mock_open):
                result = list_configs()

        # Should only return the config that didn't have IO error
        assert len(result) == 1
        assert result[0]["flow_identifier"] == "valid_config"

    def test_list_configs_ignores_non_yml_files(self, tmp_path, sample_config_data):
        """Test list_configs only processes .yml files, ignoring other file types."""
        # Create valid YAML config
        yml_config = tmp_path / "config.yml"
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
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
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
            "environment": "test",
            "version": "1.0",
        }

        config_file = tmp_path / "complex_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(complex_config, f)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
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
            "environment": "test",
            "version": "1.0",
        }

        config_file = tmp_path / "minimal_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(minimal_config, f)

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.flows.flow_config._DIRECTORY_PATH",
                tmp_path,
            ),
            patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)),
        ):
            result = list_configs()

        assert len(result) == 1
        assert result[0]["flow_identifier"] == "minimal_config"
        assert result[0]["version"] == "1.0"
        assert result[0]["environment"] == "test"

        # Verify JSON config is valid
        parsed_config = json.loads(result[0]["config"])
        assert parsed_config == minimal_config
