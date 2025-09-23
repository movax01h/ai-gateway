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
