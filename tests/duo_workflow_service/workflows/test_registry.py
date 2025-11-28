from functools import partial
from unittest.mock import Mock, patch

import pytest
from google.protobuf import struct_pb2

from duo_workflow_service.agent_platform import experimental, v1
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.workflows import chat
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import (
    CHAT_AGENT_COMPONENT_ENVIRONMENT,
    list_configs,
    resolve_workflow_class,
)
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


def build_chat_flow_config(
    components=None, prompts=None, version=None, routers=None, flow=None
):
    mock_flow_cls = Mock()

    if components is None:
        components = [
            {
                "type": "AgentComponent",
                "toolset": ["tool1", "tool2"],
                "prompt_id": "custom/prompt",
            }
        ]
    if prompts is None:
        prompts = [
            {
                "prompt_id": "custom/prompt",
                "name": "test prompt 1",
                "unit_primitives": [],
                "prompt_template": {"user": "test"},
            }
        ]
    if routers is None:
        routers = []
    if flow is None:
        flow = {}

    struct_data = {
        "environment": CHAT_AGENT_COMPONENT_ENVIRONMENT,
        "components": components,
        "routers": routers,
        "flow": flow,
    }

    expected_data = {
        "version": "v1",
        "environment": CHAT_AGENT_COMPONENT_ENVIRONMENT,
        "components": components,
        "routers": routers,
        "flow": flow,
    }

    if version:
        struct_data["version"] = version
    if prompts:
        struct_data["prompts"] = prompts
        expected_data["prompts"] = prompts

    struct = struct_pb2.Struct()
    struct.update(struct_data)

    return {
        "flow_config_cls": FlowConfig,
        "flow_cls": mock_flow_cls,
        "struct": struct,
        "expected_dict": expected_data,
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


@pytest.mark.parametrize("version", ["experimental", "v1"])
def test_registry_flow_versions_return_correct_classes(version):
    """Test that registry can resolve both experimental and v1 versions with correct flow classes."""

    # Create a minimal flow config for testing with all required fields
    struct = struct_pb2.Struct()
    struct.update(
        {
            "version": version,
            "environment": "ambient",
            "components": [{"name": "test_agent", "type": "AgentComponent"}],
            "flow": {"entry_point": "test_agent"},
            "routers": [
                {"from": "test_agent", "to": "end"}
            ],  # Add required routers field
        }
    )

    result = resolve_workflow_class(
        workflow_definition=None,
        flow_config=struct,
        flow_config_schema_version=version,
    )
    # Should return a partial function
    assert isinstance(result, partial)

    # Verify the underlying flow class is from the correct version
    if version == "experimental":
        assert result.func == experimental.flows.Flow
    elif version == "v1":
        assert result.func == v1.flows.Flow


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


@pytest.mark.parametrize(
    "config_params",
    [
        ({"version": "1.0"}),
        ({}),
    ],
    ids=["basic", "no_entry_point"],
)
def test_resolve_workflow_class_with_chat_flow_config_success(config_params):
    mocks = build_chat_flow_config(**config_params)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": (mocks["flow_config_cls"], mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_workflow_class(
            workflow_definition=None,
            flow_config=mocks["struct"],
            flow_config_schema_version="v1",
        )

        assert isinstance(result, partial)
        assert result.func == chat.Workflow

        expected_kwargs = {
            "tools_override": ["tool1", "tool2"],
            "system_template_override": None,
        }
        assert result.keywords == expected_kwargs


@pytest.mark.parametrize(
    "config_params,expected_error",
    [
        (
            {"components": [{"type": "AgentComponent"}, {"type": "AgentComponent"}]},
            "Chat-partial environment allows exactly one component, but received 2",
        ),
        (
            {"components": [{"type": "InvalidComponent"}], "prompts": None},
            "Invalid component type: InvalidComponent",
        ),
        (
            {
                "components": [{"type": "AgentComponent"}],
                "prompts": [
                    {
                        "prompt_id": "prompt1",
                        "name": "test prompt 1",
                        "unit_primitives": [],
                        "prompt_template": {"user": "test"},
                    },
                    {
                        "prompt_id": "prompt2",
                        "name": "test prompt 2",
                        "unit_primitives": [],
                        "prompt_template": {"user": "test"},
                    },
                ],
            },
            "Chat-partial environment expects exactly one prompt in prompt configuration, but received 2",
        ),
        (
            {
                "components": [{"type": "AgentComponent", "prompt_version": "v1.0"}],
                "prompts": [
                    {
                        "prompt_id": "prompt1",
                        "name": "test prompt 2",
                        "unit_primitives": [],
                        "prompt_template": {"user": "test"},
                    }
                ],
            },
            "Chat-partial environment expects either inline or in repository prompt configuration, but received both",
        ),
    ],
    ids=[
        "multiple_components",
        "invalid_component_type",
        "multiple_prompts",
        "both_prompts_and_version",
    ],
)
def test_resolve_workflow_class_with_chat_flow_config_failure(
    config_params, expected_error
):
    mocks = build_chat_flow_config(**config_params)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": (mocks["flow_config_cls"], mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        with pytest.raises(ValueError, match=expected_error):
            resolve_workflow_class(
                workflow_definition=None,
                flow_config=mocks["struct"],
                flow_config_schema_version="v1",
            )


def test_list_configs():
    """Test list_configs function returns aggregated configs from all versions."""
    mock_experimental_configs = [
        {
            "name": "config1",
            "version": "experimental",
            "environment": "test",
            "config": '{"flow": {"entry_point": "agent"}}',
        },
        {
            "name": "config2",
            "version": "experimental",
            "environment": "prod",
            "config": '{"flow": {"entry_point": "router"}}',
        },
    ]
    mock_v1_configs = [
        {
            "name": "config3",
            "version": "v1",
            "environment": "chat",
            "config": '{"flow": {"entry_point": "agent"}}',
        },
        {
            "name": "config4",
            "version": "v1",
            "environment": "chat-partial",
            "config": '{"flow": {"entry_point": "router"}}',
        },
    ]

    with patch(
        "duo_workflow_service.workflows.registry._FLOW_CONFIGS_BY_VERSION",
        {
            "experimental": lambda: mock_experimental_configs,
            "v1": lambda: mock_v1_configs,
        },
    ):
        result = list_configs()

        assert result == (mock_experimental_configs + mock_v1_configs)
        assert len(result) == 4
        assert result[0]["name"] == "config1"
        assert result[0]["version"] == "experimental"
        assert result[1]["name"] == "config2"
        assert result[1]["version"] == "experimental"
        assert result[2]["name"] == "config3"
        assert result[2]["version"] == "v1"
        assert result[3]["name"] == "config4"
        assert result[3]["version"] == "v1"
