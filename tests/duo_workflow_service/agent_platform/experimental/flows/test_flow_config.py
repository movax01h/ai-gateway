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
    load_component_class,
)


class TestFlowConfig:
    """Test FlowConfig class functionality."""

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

        assert config.flow == {"entry_point": "agent"}
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

        assert config.flow["entry_point"] == "test_agent"
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
            assert config.flow["entry_point"] == "test_agent"


class TestValidateAdditionalContextSchema:
    """Test validate_additional_context_schema field validator."""

    def test_validate_additional_context_schema_none_input(self):
        """Test that None input returns None."""
        result = FlowConfig.validate_additional_context_schema(None)
        assert result is None

    @pytest.mark.parametrize(
        "valid_schema_str",
        [
            # Basic valid schema with properties
            '{"properties": {"name": {"type": "string"}}}',
            # Schema with multiple properties
            '{"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
            # Schema with nested properties
            '{"properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}}}}',
            # Schema with additional fields beyond properties
            '{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}',
            # Schema with empty properties
            '{"properties": {}}',
            # Complex schema with various types
            '{"properties": {"items": {"type": "array", "items": {"type": "string"}}, "config": {"type": "object"}}}',
        ],
    )
    def test_validate_additional_context_schema_valid_cases(self, valid_schema_str):
        """Test validation passes for valid JSON schemas with properties field."""
        result = FlowConfig.validate_additional_context_schema(valid_schema_str)
        assert result == valid_schema_str

    @pytest.mark.parametrize(
        "invalid_json_str,expected_error_pattern",
        [
            # Malformed JSON - missing closing brace
            (
                '{"properties": {"name": {"type": "string"}',
                "Invalid JSON in additional_context_schema",
            ),
            # Malformed JSON - invalid syntax
            (
                '{"properties": {"name": {"type": "string"}}',
                "Invalid JSON in additional_context_schema",
            ),
            # Malformed JSON - trailing comma
            (
                '{"properties": {"name": {"type": "string"},}}',
                "Invalid JSON in additional_context_schema",
            ),
            # Malformed JSON - unquoted keys
            (
                '{properties: {"name": {"type": "string"}}}',
                "Invalid JSON in additional_context_schema",
            ),
            # Completely invalid JSON
            ("not json at all", "Invalid JSON in additional_context_schema"),
            # Empty string
            ("", "Invalid JSON in additional_context_schema"),
        ],
    )
    def test_validate_additional_context_schema_invalid_json(
        self, invalid_json_str, expected_error_pattern
    ):
        """Test validation fails for malformed JSON strings."""
        with pytest.raises(ValueError, match=expected_error_pattern):
            FlowConfig.validate_additional_context_schema(invalid_json_str)

    @pytest.mark.parametrize(
        "non_dict_json_str,expected_type_name",
        [
            # Array instead of dict
            ('["properties"]', "list"),
            # String instead of dict
            ('"properties"', "str"),
            # Number instead of dict
            ("42", "int"),
            # Boolean instead of dict
            ("true", "bool"),
            # Null instead of dict
            ("null", "NoneType"),
        ],
    )
    def test_validate_additional_context_schema_non_dict_json(
        self, non_dict_json_str, expected_type_name
    ):
        """Test validation fails when JSON is not a dictionary."""
        with pytest.raises(
            ValueError,
            match=f"additional_context_schema must be a dict, found {expected_type_name}",
        ):
            FlowConfig.validate_additional_context_schema(non_dict_json_str)

    @pytest.mark.parametrize(
        "missing_properties_schema_str",
        [
            # Empty dict
            "{}",
            # Dict with other fields but no properties
            '{"type": "object", "required": ["name"]}',
            # Dict with similar field name but not properties
            '{"property": {"name": {"type": "string"}}}',
            # Dict with nested structure but no top-level properties
            '{"schema": {"properties": {"name": {"type": "string"}}}}',
        ],
    )
    def test_validate_additional_context_schema_missing_properties_field(
        self, missing_properties_schema_str
    ):
        """Test validation fails when schema doesn't have a 'properties' field."""
        with pytest.raises(
            ValueError, match="additional_context_schema must have a 'properties' field"
        ):
            FlowConfig.validate_additional_context_schema(missing_properties_schema_str)

    def test_validate_additional_context_schema_integration_with_flowconfig(self):
        """Test that the validator works correctly when creating FlowConfig instances."""
        # Test with valid additional_context_schema
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [
                {"name": "agent", "type": "AgentComponent", "inputs": ["context:goal"]}
            ],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "remote",
            "version": "experimental",
            "additional_context_schema": '{"properties": {"user_input": {"type": "string"}}}',
        }

        config = FlowConfig(**config_data)
        assert (
            config.additional_context_schema
            == '{"properties": {"user_input": {"type": "string"}}}'
        )

    def test_validate_additional_context_schema_integration_invalid_schema(self):
        """Test that FlowConfig creation fails with invalid additional_context_schema."""
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [
                {"name": "agent", "type": "AgentComponent", "inputs": ["context:goal"]}
            ],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "remote",
            "version": "experimental",
            "additional_context_schema": '{"no_properties_field": true}',
        }

        with pytest.raises(ValidationError) as exc_info:
            FlowConfig(**config_data)

        assert "additional_context_schema must have a 'properties' field" in str(
            exc_info.value
        )

    def test_validate_additional_context_schema_integration_none_value(self):
        """Test that FlowConfig works correctly with None additional_context_schema."""
        config_data = {
            "flow": {"entry_point": "agent"},
            "components": [
                {"name": "agent", "type": "AgentComponent", "inputs": ["context:goal"]}
            ],
            "routers": [{"from": "agent", "to": "end"}],
            "environment": "remote",
            "version": "experimental",
            "additional_context_schema": None,
        }

        config = FlowConfig(**config_data)
        assert config.additional_context_schema is None

    def test_validate_additional_context_schema_edge_case_large_schema(self):
        """Test validation with a large, complex schema."""
        large_schema = {
            "properties": {
                f"field_{i}": {
                    "type": "object",
                    "properties": {
                        "nested_field": {"type": "string"},
                        "nested_array": {"type": "array", "items": {"type": "integer"}},
                    },
                }
                for i in range(10)
            }
        }
        large_schema_str = json.dumps(large_schema)

        result = FlowConfig.validate_additional_context_schema(large_schema_str)
        assert result == large_schema_str

    def test_validate_additional_context_schema_edge_case_deeply_nested(self):
        """Test validation with deeply nested schema structure."""
        nested_schema = {
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {"deep_field": {"type": "string"}},
                                }
                            },
                        }
                    },
                }
            }
        }
        nested_schema_str = json.dumps(nested_schema)

        result = FlowConfig.validate_additional_context_schema(nested_schema_str)
        assert result == nested_schema_str


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
