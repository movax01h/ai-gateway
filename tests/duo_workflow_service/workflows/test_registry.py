from functools import partial
from unittest.mock import Mock, patch

import pytest
from google.protobuf import struct_pb2

from duo_workflow_service.workflows import chat
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import resolve_workflow_class
from duo_workflow_service.workflows.software_development import Workflow


@pytest.fixture
def simple_flow_config():
    mock_flow_config_cls = Mock()
    mock_config_instance = Mock()
    mock_flow_config_cls.return_value = mock_config_instance

    # Create mock flow class
    mock_flow_cls = Mock()

    struct = struct_pb2.Struct()
    struct.update(
        {
            "version": "1.0",
            "environment": "test",
            "components": [{"name": "test_agent", "type": "AgentComponent"}],
            "flow": {"entry_point": "test_agent"},
        }
    )

    expected_dict = {
        "version": "experimental",
        "environment": "test",
        "components": [{"name": "test_agent", "type": "AgentComponent"}],
        "flow": {"entry_point": "test_agent"},
    }

    return {
        "flow_config_cls": mock_flow_config_cls,
        "flow_cls": mock_flow_cls,
        "struct": struct,
        "config_instance": mock_config_instance,
        "expected_dict": expected_dict,
    }


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


def test_resolve_workflow_class_with_flow_config(simple_flow_config):
    """Test resolving workflow class with flow config protobuf."""
    mocks = simple_flow_config

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"experimental": (mocks["flow_config_cls"], mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_workflow_class(
            workflow_definition=None,
            flow_config=mocks["struct"],
            flow_config_schema_version="experimental",
        )

        assert isinstance(result, partial)
        assert result.func == mocks["flow_cls"]
        assert result.keywords == {"config": mocks["config_instance"]}

        mocks["flow_config_cls"].assert_called_once_with(
            version="experimental",
            environment="test",
            components=[{"name": "test_agent", "type": "AgentComponent"}],
            flow={"entry_point": "test_agent"},
        )


def test_resolve_workflow_class_with_chat_flow_config():
    mock_flow_config_cls = Mock()
    mock_config_instance = Mock()
    mock_config_instance.environment = "chat-partial"
    mock_config_instance.components = [
        {
            "type": "AgentComponent",
            "toolset": ["tool1", "tool2"],
            "prompt_id": "custom/prompt",
        }
    ]
    mock_config_instance.prompts = [
        {"prompt_id": "custom/prompt", "content": "test prompt"}
    ]
    mock_flow_config_cls.return_value = mock_config_instance

    mock_flow_cls = Mock()

    struct = struct_pb2.Struct()
    struct.update(
        {
            "version": "1.0",
            "environment": "chat-partial",
            "components": [
                {
                    "type": "AgentComponent",
                    "toolset": ["tool1", "tool2"],
                    "prompt_id": "custom/prompt",
                }
            ],
            "prompts": [{"prompt_id": "custom/prompt", "content": "test prompt"}],
        }
    )

    expected_dict = {
        "version": "experimental",
        "environment": "chat-partial",
        "components": [
            {
                "type": "AgentComponent",
                "toolset": ["tool1", "tool2"],
                "prompt_id": "custom/prompt",
            }
        ],
        "prompts": [{"prompt_id": "custom/prompt", "content": "test prompt"}],
    }

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"experimental": (mock_flow_config_cls, mock_flow_cls)},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=expected_dict,
        ),
    ):
        result = resolve_workflow_class(
            workflow_definition=None,
            flow_config=struct,
            flow_config_schema_version="experimental",
        )

        assert isinstance(result, partial)
        assert result.func == chat.Workflow

        expected_kwargs = {
            "tools_override": ["tool1", "tool2"],
            "prompt_template_id_override": "custom/prompt",
            "prompt_template_version_override": None,
            "prompt_template_override": {
                "prompt_id": "custom/prompt",
                "content": "test prompt",
            },
        }
        assert result.keywords == expected_kwargs

        mock_flow_config_cls.assert_called_once_with(
            version="experimental",
            environment="chat-partial",
            components=[
                {
                    "type": "AgentComponent",
                    "toolset": ["tool1", "tool2"],
                    "prompt_id": "custom/prompt",
                }
            ],
            prompts=[{"prompt_id": "custom/prompt", "content": "test prompt"}],
        )


def test_resolve_workflow_class_with_chat_flow_config_invalid_component_count():
    mock_flow_config_cls = Mock()
    mock_config_instance = Mock()
    mock_config_instance.environment = "chat-partial"
    mock_config_instance.components = [
        {"type": "AgentComponent"},
        {"type": "AgentComponent"},
    ]  # Too many components
    mock_flow_config_cls.return_value = mock_config_instance

    struct = struct_pb2.Struct()
    struct.update(
        {
            "environment": "chat-partial",
            "components": [{"type": "AgentComponent"}, {"type": "AgentComponent"}],
        }
    )

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"experimental": (mock_flow_config_cls, Mock())},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value={
                "version": "experimental",
                "environment": "chat-partial",
                "components": [{"type": "AgentComponent"}, {"type": "AgentComponent"}],
            },
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Chat-partial environment allows exactly one component, but received 2",
        ):
            resolve_workflow_class(
                workflow_definition=None,
                flow_config=struct,
                flow_config_schema_version="experimental",
            )


def test_resolve_workflow_class_with_chat_flow_config_invalid_component_type():
    mock_flow_config_cls = Mock()
    mock_config_instance = Mock()
    mock_config_instance.environment = "chat-partial"
    mock_config_instance.components = [{"type": "InvalidComponent"}]
    mock_flow_config_cls.return_value = mock_config_instance

    struct = struct_pb2.Struct()
    struct.update(
        {"environment": "chat-partial", "components": [{"type": "InvalidComponent"}]}
    )

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"experimental": (mock_flow_config_cls, Mock())},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value={
                "version": "experimental",
                "environment": "chat-partial",
                "components": [{"type": "InvalidComponent"}],
            },
        ),
    ):
        with pytest.raises(
            ValueError, match="Invalid component type: InvalidComponent"
        ):
            resolve_workflow_class(
                workflow_definition=None,
                flow_config=struct,
                flow_config_schema_version="experimental",
            )
