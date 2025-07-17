from functools import partial
from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import resolve_workflow_class
from duo_workflow_service.workflows.software_development import Workflow


def test_registry_resolve():
    # Test resolving default workflow
    assert resolve_workflow_class(None) == Workflow

    # Test resolving a non-existent workflow
    with pytest.raises(ValueError, match="Unknown Flow"):
        resolve_workflow_class("non_existent_workflow")

    # Test that resolved class is a subclass of AbstractWorkflow
    resolved_class = resolve_workflow_class("software_development")
    assert issubclass(resolved_class, AbstractWorkflow)
    assert resolved_class == Workflow


def test_registry_resolve_experimental_flow():
    """Test resolving experimental flow with config path."""
    mock_config = Mock()
    mock_flow_cls = Mock()
    mock_flow_config_cls = Mock()
    mock_flow_config_cls.from_yaml_config.return_value = mock_config

    with patch(
        "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
        {"experimental": (mock_flow_config_cls, mock_flow_cls)},
    ):
        result = resolve_workflow_class("prototype/experimental")

        # Should return a partial function
        assert isinstance(result, partial)
        mock_flow_config_cls.from_yaml_config.assert_called_once_with("prototype")


def test_registry_resolve_unknown_flow_version():
    """Test resolving flow with unknown version raises ValueError."""
    with pytest.raises(ValueError, match="Unknown Flow version: unknown_version"):
        resolve_workflow_class("prototype/unknown_version")


def test_registry_resolve_flow_config_error():
    """Test that config loading errors are handled properly."""
    mock_flow_config_cls = Mock()
    mock_flow_config_cls.from_yaml_config.side_effect = FileNotFoundError(
        "Config not found"
    )

    with patch(
        "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
        {"experimental": (mock_flow_config_cls, Mock())},
    ):
        with pytest.raises(ValueError, match="Unknown Flow"):
            resolve_workflow_class("nonexistent/experimental")
