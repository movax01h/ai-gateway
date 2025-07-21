from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

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

        config_file = tmp_path / "config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            config = FlowConfig.from_yaml_config("config.yaml")

        assert config.flow["entry_point"] == "test_agent"
        assert config.environment == "local"
        assert config.version == "experimental"

    def test_flowconfig_from_yaml_config_file_not_found(self):
        """Test loading YAML config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            FlowConfig.from_yaml_config("nonexistent.yaml")

        assert "nonexistent.yaml file not found" in str(exc_info.value)

    def test_flowconfig_from_yaml_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises YAMLError."""
        config_file = tmp_path / "invalid_config.yaml"

        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with patch.object(FlowConfig, "DIRECTORY_PATH", Path(tmp_path)):
            with pytest.raises(yaml.YAMLError) as exc_info:
                FlowConfig.from_yaml_config("invalid_config.yaml")

        assert "Error parsing YAML file" in str(exc_info.value)


class TestLoadComponentClass:
    """Test load_component_class function."""

    def test_load_component_class_success(self):
        """Test loading existing component class successfully."""
        # Mock the components module to have a test class
        mock_component_class = type("TestComponent", (), {})

        with patch(
            "duo_workflow_service.agent_platform.experimental.flows.flow_config.components"
        ) as mock_components:
            mock_components.TestComponent = mock_component_class

            result = load_component_class("TestComponent")

            assert result == mock_component_class

    def test_load_component_class_not_found(self):
        """Test loading non-existent component class raises AttributeError."""
        with patch(
            "duo_workflow_service.agent_platform.experimental.flows.flow_config.components"
        ) as mock_components:
            # Remove the attribute to simulate it not existing
            del mock_components.NonExistentComponent

            with pytest.raises(AttributeError) as exc_info:
                load_component_class("NonExistentComponent")

            assert "Component class 'NonExistentComponent' not found" in str(
                exc_info.value
            )

    def test_load_component_class_not_a_class(self):
        """Test loading non-class object raises TypeError."""
        with patch(
            "duo_workflow_service.agent_platform.experimental.flows.flow_config.components"
        ) as mock_components:
            # Set attribute to a function instead of a class
            mock_components.NotAClass = lambda: None

            with pytest.raises(TypeError) as exc_info:
                load_component_class("NotAClass")

            assert "'NotAClass' is not a class" in str(exc_info.value)
