import copy
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock, patch

import pytest
import yaml
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from duo_workflow_service.agent_platform import experimental, v1
from duo_workflow_service.agent_platform.constants import (
    CHAT_PARTIAL_ENVIRONMENT,
    METADATA_ONLY_FLOWS,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    PartialFlowConfig as ExperimentalPartialFlowConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    config_from_data as experimental_config_from_data,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    load_config as experimental_load_config,
)
from duo_workflow_service.agent_platform.v1.chat_engine import ChatFlow
from duo_workflow_service.agent_platform.v1.flows import FetchedFlow, flow_config
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    PartialFlowConfig,
    config_from_data,
    load_config,
)
from duo_workflow_service.flow_request import (
    FetchedFlowRequest,
    InlineFlowRequest,
    LegacyWorkflowRequest,
    RegistryFlowRequest,
)
from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.workflows import chat
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import (
    FlowSchema,
    ResolvedFlow,
    _convert_struct_to_flow_config,
    _load_flow_from_inline_config,
    _load_flow_from_registry,
    _normalize_prompt_template_order,
    flow_config_from_fetched_definition,
    flow_factory,
    get_flow_schema,
    list_configs,
    load_config_file,
    resolve_flow,
)
from duo_workflow_service.workflows.software_development import Workflow


@pytest.fixture
def simple_flow_config():
    mock_flow_config_cls = Mock()
    mock_config_instance = Mock()
    # A config with no prompts of its own, as the struct above declares.
    mock_config_instance.prompts = None
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
        # Validates every config as the mock class, as the real schemas do for a full config.
        "config_from_data": lambda data, **fields: mock_flow_config_cls(
            **data, **fields
        ),
        "flow_cls": mock_flow_cls,
        "struct": struct,
        "config_instance": mock_config_instance,
        "expected_dict": expected_dict,
    }


def build_chat_flow_config(
    components=None,
    prompts=None,
    version=None,
    routers=None,
    flow=None,
    is_partial=False,
    environment=CHAT_PARTIAL_ENVIRONMENT,
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
    if routers is None and not is_partial:
        routers = []
    if flow is None and not is_partial:
        flow = {}

    struct_data = {
        "environment": environment,
        "components": components,
        "routers": routers,
        "flow": flow,
    }

    expected_data = {
        "version": "v1",
        "environment": environment,
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
        "config_from_data": config_from_data,
        "flow_cls": mock_flow_cls,
        "struct": struct,
        "expected_dict": expected_data,
    }


def test_registry_resolve():
    # Test that resolved class is a subclass of AbstractWorkflow
    resolved_class = resolve_flow(
        LegacyWorkflowRequest(workflow_definition="software_development")
    ).factory
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

    result = resolve_flow(
        InlineFlowRequest(config_struct=struct, schema_version=version)
    ).factory
    # Should return a partial function
    assert isinstance(result, partial)

    # Verify the underlying flow class is from the correct version
    if version == "experimental":
        assert result.func == experimental.flows.Flow
    elif version == "v1":
        assert result.func == v1.flows.Flow


@pytest.mark.parametrize(
    ("schema_version", "expected"),
    [
        ("v1", (config_from_data, load_config, v1.flows.Flow)),
        (
            "experimental",
            (
                experimental_config_from_data,
                experimental_load_config,
                experimental.flows.Flow,
            ),
        ),
    ],
)
def test_get_flow_schema_returns_the_schema_functions(schema_version, expected):
    assert get_flow_schema(schema_version) == expected


@pytest.mark.parametrize(
    ("schema_version", "environment", "expected_config_cls"),
    [
        ("v1", CHAT_PARTIAL_ENVIRONMENT, PartialFlowConfig),
        ("v1", "chat", FlowConfig),
        ("v1", "ambient", FlowConfig),
        ("experimental", CHAT_PARTIAL_ENVIRONMENT, ExperimentalPartialFlowConfig),
        ("experimental", "chat", ExperimentalFlowConfig),
        ("experimental", "ambient", ExperimentalFlowConfig),
    ],
)
def test_an_inline_config_validates_as_the_class_its_environment_selects(
    schema_version, environment, expected_config_cls
):
    struct = struct_pb2.Struct()
    struct.update(
        {
            "version": schema_version,
            "environment": environment,
            "components": [{"name": "agent", "type": "AgentComponent"}],
            "routers": [{"from": "agent", "to": "end"}],
            "flow": {"entry_point": "agent"},
        }
    )
    config = _convert_struct_to_flow_config(
        struct=struct,
        flow_config_schema_version=schema_version,
    )

    assert type(config) is expected_config_cls


def test_registry_resolve_unknown_flow_version():
    """Test resolving flow with unknown version raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid flowConfigSchemaVersion"):
        RegistryFlowRequest(
            config_id="prototype",
            schema_version="unknown_version",
            version="1.0.0",
        )


def test_registry_resolve_flow_config_error():
    """Test that config loading errors are handled properly."""
    mock_load_config = Mock(side_effect=FileNotFoundError("Config not found"))

    with patch(
        "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
        {"experimental": FlowSchema(Mock(), mock_load_config, Mock())},
    ):
        with pytest.raises(ValueError, match="Unknown flow"):
            resolve_flow(
                RegistryFlowRequest(
                    config_id="nonexistent",
                    schema_version="experimental",
                    version="1.0.0",
                )
            )


def test_resolve_flow_with_flow_config(
    simple_flow_config,
):  # pylint: disable=redefined-outer-name
    """Test resolving workflow class with flow config protobuf."""
    mocks = simple_flow_config

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {
                "experimental": FlowSchema(
                    mocks["config_from_data"], Mock(), mocks["flow_cls"]
                )
            },
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(
                config_struct=mocks["struct"], schema_version="experimental"
            )
        ).factory

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
def test_resolve_flow_with_chat_flow_config_success(config_params):
    mocks = build_chat_flow_config(**config_params)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(mocks["config_from_data"], Mock(), mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(config_struct=mocks["struct"], schema_version="v1")
        ).factory

        assert isinstance(result, partial)
        assert result.func == chat.Workflow

        expected_kwargs = {
            "tools_override": ["tool1", "tool2"],
            "system_template_override": None,
            "agent_name_override": None,  # Default component has no name
            "component_inputs_config": None,
        }
        assert result.keywords == expected_kwargs


def test_resolve_flow_with_chat_flow_config_extracts_agent_name():
    """Test that agent_name is extracted from chat-partial flow config."""
    components = [
        {
            "name": "348/0",  # Agent name for Duo Planner
            "type": "AgentComponent",
            "toolset": ["tool1", "tool2"],
            "prompt_id": "custom/prompt",
        }
    ]
    mocks = build_chat_flow_config(components=components)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(mocks["config_from_data"], Mock(), mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(config_struct=mocks["struct"], schema_version="v1")
        ).factory

        assert isinstance(result, partial)
        assert result.func == chat.Workflow

        # Verify agent_name_override is extracted from component name
        expected_kwargs = {
            "tools_override": ["tool1", "tool2"],
            "system_template_override": None,
            "agent_name_override": "348/0",
            "component_inputs_config": None,
        }
        assert result.keywords == expected_kwargs


@pytest.mark.parametrize(
    "toolset_config",
    [{}, {"toolset": []}],
    ids=["omitted", "empty"],
)
def test_resolve_flow_with_chat_flow_config_treats_omitted_toolset_as_empty(
    toolset_config,
):
    mocks = build_chat_flow_config(
        components=[{"type": "AgentComponent", **toolset_config}]
    )

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(mocks["config_from_data"], Mock(), mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(config_struct=mocks["struct"], schema_version="v1")
        ).factory

    assert result.keywords["tools_override"] == []


def test_resolve_flow_with_chat_flow_config_forwards_component_inputs():
    """Test that component inputs config is forwarded to chat.Workflow."""
    component_inputs = [
        {"from": "context:goal", "as": "goal"},
        {
            "from": "context:inputs.orbit_context.orbit_enabled",
            "as": "orbit_enabled",
            "optional": True,
        },
    ]
    components = [
        {
            "type": "AgentComponent",
            "toolset": ["tool1"],
            "inputs": component_inputs,
        }
    ]
    mocks = build_chat_flow_config(components=components)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(mocks["config_from_data"], Mock(), mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(config_struct=mocks["struct"], schema_version="v1")
        ).factory

    assert result.keywords["component_inputs_config"] == component_inputs


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
                "components": [
                    {
                        "type": "AgentComponent",
                        "prompt_id": "prompt1",
                        "prompt_version": "v1.0",
                    }
                ],
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
        (
            {"components": [{"type": "AgentComponent", "prompt_version": "v1.0"}]},
            "prompt_id",
        ),
    ],
    ids=[
        "multiple_components",
        "invalid_component_type",
        "multiple_prompts",
        "both_prompts_and_version",
        "version_without_prompt_id",
    ],
)
def test_resolve_flow_with_chat_flow_config_failure(config_params, expected_error):
    mocks = build_chat_flow_config(**config_params)

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(mocks["config_from_data"], Mock(), mocks["flow_cls"])},
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=mocks["expected_dict"],
        ),
    ):
        with pytest.raises(ValueError, match=expected_error):
            resolve_flow(
                InlineFlowRequest(config_struct=mocks["struct"], schema_version="v1")
            )


def test_routers_and_flow_required_for_chat_optional_for_chat_partial():
    """Test that routers and flow are required for 'chat' but optional for 'chat-partial'."""

    # Test 1: chat-partial with missing routers and flow should succeed
    chat_partial_config = build_chat_flow_config(
        is_partial=True, environment="chat-partial"
    )

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {
                "v1": FlowSchema(
                    chat_partial_config["config_from_data"],
                    Mock(),
                    chat_partial_config["flow_cls"],
                )
            },
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=chat_partial_config["expected_dict"],
        ),
    ):
        result = resolve_flow(
            InlineFlowRequest(
                config_struct=chat_partial_config["struct"], schema_version="v1"
            )
        ).factory
        assert isinstance(result, partial)

    # Test 2: chat environment with missing routers and flow should fail
    chat_config = build_chat_flow_config(is_partial=True, environment="chat")

    with (
        patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {
                "v1": FlowSchema(
                    chat_config["config_from_data"], Mock(), chat_config["flow_cls"]
                )
            },
        ),
        patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=chat_config["expected_dict"],
        ),
    ):
        # Should raise ValidationError because routers and flow are required for FlowConfig
        with pytest.raises((ValueError, ValidationError)):
            resolve_flow(
                InlineFlowRequest(
                    config_struct=chat_config["struct"], schema_version="v1"
                )
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


def test_all_flow_configs_are_valid():
    """Test that all flow config YAML files can be resolved to workflow classes.

    This validates that every built-in flow config YAML file in both v1 and experimental directories can be loaded and
    resolved without errors, catching schema validation issues (like missing required fields) at test time rather than
    at runtime.
    """
    validation_errors = []

    flow_config_directories = {
        "v1": FlowConfig.DIRECTORY_PATH,
        "experimental": ExperimentalFlowConfig.DIRECTORY_PATH,
    }

    for version, directory in flow_config_directories.items():
        for config_file in directory.glob("*/*.yml"):
            flow_name = config_file.parent.name
            flow_version = config_file.stem  # e.g. "1.0.0"
            try:
                resolve_flow(
                    RegistryFlowRequest(
                        config_id=flow_name,
                        schema_version=version,
                        version=flow_version,
                    )
                )
            except (ValueError, ValidationError) as e:
                validation_errors.append(
                    f"Validation failed for {flow_name}/{version} (v{flow_version}): {e}"
                )

    if validation_errors:
        error_message = "\n".join(validation_errors)
        pytest.fail(f"Flow config validation errors:\n{error_message}")


class TestLoadFlowFromRegistry:
    """Tests for _load_flow_from_registry (YAML-based flow loader)."""

    def test_explicit_version_passes_correct_path(self):
        """Verify that the version argument is forwarded verbatim to load_config."""
        mock_config = Mock()
        mock_config.environment = "ambient"
        mock_config.components = []
        mock_config.prompts = None
        mock_config.resolved_version = "2.0.0"
        mock_load_config = Mock(return_value=mock_config)

        with patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(Mock(), mock_load_config, Mock())},
        ):
            with patch("duo_workflow_service.workflows.registry.flow_factory"):
                _load_flow_from_registry("developer", "v1", "2.0.0")
                mock_load_config.assert_called_once_with("developer", "2.0.0")

    def test_returns_resolved_version_from_config(self):
        """The loaded flow's identity is surfaced: resolved semver plus flow_id/schema_version."""
        mock_config = Mock()
        mock_config.environment = "ambient"
        mock_config.components = []
        mock_config.prompts = None
        mock_config.resolved_version = "2.1.0"
        mock_load_config = Mock(return_value=mock_config)

        with patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(Mock(), mock_load_config, Mock())},
        ):
            with patch("duo_workflow_service.workflows.registry.flow_factory"):
                resolved = _load_flow_from_registry("developer", "v1", "^2.0.0")

        assert resolved.flow_id == "developer"
        assert resolved.schema_version == "v1"
        assert resolved.flow_version == "2.1.0"


class TestResolveFlowTrackingFields:
    """resolve_flow surfaces the identity of what was *resolved* for tracing."""

    def test_registry_flow_tracks_resolved_version(self):
        mock_config = Mock()
        mock_config.environment = "ambient"
        mock_config.components = []
        mock_config.prompts = None
        mock_config.resolved_version = "2.1.0"
        mock_load_config = Mock(return_value=mock_config)

        with patch(
            "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
            {"v1": FlowSchema(Mock(), mock_load_config, Mock())},
        ):
            with patch("duo_workflow_service.workflows.registry.flow_factory"):
                resolved = resolve_flow(
                    RegistryFlowRequest(
                        config_id="developer", schema_version="v1", version="^2.0.0"
                    )
                )

        assert isinstance(resolved, ResolvedFlow)
        assert resolved.tracking_fields() == {
            "flow_id": "developer",
            "schema_version": "v1",
            "flow_version": "2.1.0",
        }

    def test_inline_flow_tracks_only_schema_version(self, simple_flow_config):  # pylint: disable=redefined-outer-name
        mocks = simple_flow_config

        with (
            patch(
                "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
                {
                    "experimental": FlowSchema(
                        mocks["config_from_data"], Mock(), mocks["flow_cls"]
                    )
                },
            ),
            patch(
                "duo_workflow_service.workflows.registry.MessageToDict",
                return_value=mocks["expected_dict"],
            ),
        ):
            resolved = resolve_flow(
                InlineFlowRequest(
                    config_struct=mocks["struct"], schema_version="experimental"
                )
            )

        assert resolved.tracking_fields() == {"schema_version": "experimental"}

    def test_legacy_flow_tracks_nothing(self):
        resolved = resolve_flow(
            LegacyWorkflowRequest(workflow_definition="software_development")
        )
        assert resolved.tracking_fields() == {}

    def test_unknown_legacy_flow_falls_back_to_software_development(self):
        """An unrecognized legacy definition falls back to the default workflow and tracks nothing."""
        resolved = resolve_flow(
            LegacyWorkflowRequest(workflow_definition="not_a_known_workflow")
        )
        assert resolved.factory == Workflow
        assert resolved.tracking_fields() == {}

    def test_inline_flow_security_exception_is_propagated(self, simple_flow_config):  # pylint: disable=redefined-outer-name
        """SecurityException raised inside _load_flow_from_inline_config must propagate, not be wrapped."""
        mocks = simple_flow_config

        with (
            patch(
                "duo_workflow_service.workflows.registry._FLOW_BY_VERSIONS",
                {
                    "experimental": FlowSchema(
                        mocks["config_from_data"], Mock(), mocks["flow_cls"]
                    )
                },
            ),
            patch(
                "duo_workflow_service.workflows.registry.MessageToDict",
                return_value=mocks["expected_dict"],
            ),
            patch(
                "duo_workflow_service.workflows.registry.flow_factory",
                side_effect=SecurityException("dangerous content"),
            ),
        ):
            with pytest.raises(SecurityException):
                resolve_flow(
                    InlineFlowRequest(
                        config_struct=mocks["struct"], schema_version="experimental"
                    )
                )


def _experimental_inline_config(model_params):
    """Minimal experimental flow config with the given inline model params."""
    return {
        "version": "experimental",
        "environment": "ambient",
        "components": [
            {
                "name": "agent",
                "type": "AgentComponent",
                "prompt_id": "p",
                "toolset": [],
                "inputs": [{"from": "context:goal", "as": "goal"}],
            }
        ],
        "routers": [{"from": "agent", "to": "end"}],
        "flow": {"entry_point": "agent"},
        "prompts": [
            {
                "prompt_id": "p",
                "name": "p",
                "model": {"params": model_params},
                "prompt_template": {"user": "say hello"},
                "unit_primitives": ["duo_agent_platform"],
            }
        ],
    }


def test_load_flow_from_inline_config_rejects_untrusted_model_selection():
    """An inline experimental config that self-specifies a model (via model_class_provider) is rejected at flow-config
    validation, because ModelConfig.params (BaseModelParams) forbids unknown fields."""
    struct = struct_pb2.Struct()
    struct.update(
        _experimental_inline_config(
            {
                "model_class_provider": "litellm",
                "model": "vertex_ai/gemini-2.5-flash",
                "max_tokens": 1,
                "max_retries": 0,
                "extra_headers": {"Host": "example.test"},
            }
        )
    )

    with pytest.raises(ValueError, match="model_class_provider"):
        _load_flow_from_inline_config(struct, "experimental")


def test_load_flow_from_inline_config_accepts_valid_experimental_config():
    """A well-formed experimental config (no untrusted model selection) still loads."""
    struct = struct_pb2.Struct()
    struct.update(_experimental_inline_config({"max_tokens": 1}))

    factory = _load_flow_from_inline_config(struct, "experimental")

    assert factory is not None


class TestNormalizePromptTemplateOrder:
    """Canonical role ordering for struct-sourced prompt templates.

    flowConfig is a protobuf Struct (map<string, Value>), so prompt_template key order is unspecified and varies per
    process under the upb backend. The registry must impose a canonical role order on struct-sourced templates or
    message order (and the security block position) is nondeterministic.
    """

    @pytest.mark.parametrize(
        "template,expected_keys",
        [
            (
                {
                    "placeholder": "history",
                    "assistant": "A",
                    "user": "U",
                    "system": "S",
                },
                ["system", "user", "assistant", "placeholder"],
            ),
            (
                {"user": "U", "system_dynamic": "S2", "system_static": "S1"},
                ["system_static", "system_dynamic", "user"],
            ),
            (
                {"custom": "C", "user": "U"},
                ["user", "custom"],
            ),
            (
                {"human": "H", "ai": "A", "function": "F"},
                ["human", "ai", "function"],
            ),
        ],
        ids=[
            "canonical_roles",
            "split_system_static_before_dynamic",
            "unknown_roles_last",
            "role_aliases",
        ],
    )
    def test_orders_roles_canonically(self, template, expected_keys):
        config_dict = {"prompts": [{"prompt_id": "p", "prompt_template": template}]}

        _normalize_prompt_template_order(config_dict)

        normalized = config_dict["prompts"][0]["prompt_template"]
        assert list(normalized.keys()) == expected_keys
        assert normalized == template

    def test_preserves_values_including_lists_and_placeholder_names(self):
        template = {
            "placeholder": "history",
            "user": "U",
            "system": ["static part", "dynamic part"],
        }
        config_dict = {"prompts": [{"prompt_id": "p", "prompt_template": template}]}

        _normalize_prompt_template_order(config_dict)

        normalized = config_dict["prompts"][0]["prompt_template"]
        assert list(normalized.keys()) == ["system", "user", "placeholder"]
        assert normalized["system"] == ["static part", "dynamic part"]
        assert normalized["placeholder"] == "history"

    @pytest.mark.parametrize(
        "config_dict",
        [
            {},
            {"prompts": None},
            {"prompts": "oops"},
            {"prompts": ["oops"]},
            {"prompts": [{"prompt_id": "p"}]},
            {"prompts": [{"prompt_id": "p", "prompt_template": "oops"}]},
        ],
        ids=[
            "no_prompts",
            "null_prompts",
            "non_list_prompts",
            "non_dict_prompt_entry",
            "no_template",
            "non_dict_template",
        ],
    )
    def test_tolerates_malformed_configs(self, config_dict):
        original = copy.deepcopy(config_dict)
        _normalize_prompt_template_order(config_dict)
        assert config_dict == original

    def test_inline_config_message_order_is_deterministic(self):
        """End to end through the Struct: canonical order regardless of map order.

        A live Struct round-trip cannot pin the bug in a single process (the
        upb hash seed is fixed per process), so the adversarial order is
        injected by patching MessageToDict.
        """
        config = _experimental_inline_config({"max_tokens": 1})
        config["prompts"][0]["prompt_template"] = {"user": "U", "system": "S"}
        struct = struct_pb2.Struct()
        struct.update(config)

        adversarial = MessageToDict(struct)
        adversarial["prompts"][0]["prompt_template"] = {"user": "U", "system": "S"}

        with patch(
            "duo_workflow_service.workflows.registry.MessageToDict",
            return_value=adversarial,
        ):
            loaded = _convert_struct_to_flow_config(
                struct=struct,
                flow_config_schema_version="experimental",
            )

        assert list(loaded.prompts[0].prompt_template.keys()) == ["system", "user"]


def shipped_chat_partial_configs():
    legacy = (
        (config_file.parent.name, config_file)
        for config_file in FlowConfig.DIRECTORY_PATH.glob("*/*.yml")
    )
    features = (
        (config_id, config_file)
        for config_id, config_dir in FlowConfig.feature_config_roots().items()
        for config_file in config_dir.glob("*.yml")
    )
    for config_id, config_file in chain(legacy, features):
        environment = yaml.safe_load(config_file.read_text()).get("environment")
        if environment == CHAT_PARTIAL_ENVIRONMENT:
            yield config_id, config_file.stem


def _as_registry_config(config_cls, data):
    """Build a config the way the registry loader does, so it carries the identity of a flow shipped here."""
    with NamedTemporaryFile("w", suffix=".yml") as written:
        yaml.safe_dump(data, written)
        written.flush()

        return config_cls.from_registry_file(Path(written.name), "assistant", "1.0.0")


def engine_owned_chat_partial_config(config_cls=PartialFlowConfig, **overrides):
    """A chat-partial config that opts into the engine: its prompts declare `template`."""
    config = {
        "version": "v1",
        "environment": CHAT_PARTIAL_ENVIRONMENT,
        "components": [
            {
                "name": "chat_agent",
                "type": "AgentComponent",
                "toolset": [],
                "prompt_id": "chat_agent_prompt",
            }
        ],
        "prompts": [
            {
                "name": "chat_agent_prompt",
                "prompt_id": "chat_agent_prompt",
                "unit_primitives": ["duo_chat"],
                "template": {"system": "You are a chat agent."},
            }
        ],
    }
    config.update(overrides)

    return _as_registry_config(config_cls, config)


def engine_owned_factory(config):
    return flow_factory(Mock(), config)


def test_a_template_version_outside_chat_partial_is_refused():
    # Refused before the environment branch, so an ambient flow can't build with prompts nothing renders.
    config = engine_owned_chat_partial_config(
        config_cls=FlowConfig,
        environment="ambient",
        routers=[{"from": "chat_agent", "to": "end"}],
        flow={"entry_point": "chat_agent"},
    )

    with pytest.raises(ValueError, match="only in the 'chat-partial' environment"):
        flow_factory(Mock(), config)


def test_a_template_version_from_the_registry_takes_the_engine_path(tmp_path):
    # Loaded the way a session loads it: a chat-partial file declares no `routers`, and the engine completes them.
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "1.0.0.yml").write_text(
        (
            Path(__file__).parents[2]
            / "prompts"
            / "config"
            / "chat_partial_template.yml"
        ).read_text()
    )

    with (
        patch.object(FlowConfig, "DIRECTORY_PATH", tmp_path),
        patch.object(FlowConfig, "feature_config_roots", return_value={}),
    ):
        resolved = _load_flow_from_registry("assistant", "v1", "^1.0.0")

    assert resolved.factory.func is ChatFlow
    assert resolved.factory.keywords["config"].config_id == "assistant"


@pytest.mark.parametrize("schema_version", ["experimental", "v1"])
def test_a_known_registry_file_loads_as_its_environment_class_with_its_identity(
    tmp_path, schema_version
):
    path = tmp_path / "1.0.0.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": schema_version,
                "environment": CHAT_PARTIAL_ENVIRONMENT,
                "components": [
                    {"name": "agent", "type": "AgentComponent", "prompt_id": "p"}
                ],
            }
        )
    )

    config = load_config_file(schema_version, path, "assistant", "1.0.0")

    # Chat-partial, so the partial class, which may leave out `routers`.
    assert type(config).__name__ == "PartialFlowConfig"
    assert (config.config_id, config.config_version) == ("assistant", "1.0.0")


def test_every_shipped_chat_partial_config_stays_on_legacy_chat():
    with patch.dict(flow_config._FEATURE_FLOW_ROOTS):
        flow_config.discover_feature_flow_configs()
        configs = list(shipped_chat_partial_configs())

        assert ("duo_permissions_assistant", "2.0.0") in configs

        for config_id, version in configs:
            factory = _load_flow_from_registry(config_id, "v1", version).factory

            assert factory.func is chat.Workflow, f"{config_id}/{version}"


def test_an_inline_template_flow_never_reaches_the_engine():
    # An AI Catalog flow arrives inline, with no registry identity, so it may not use `template` yet.
    struct = struct_pb2.Struct()
    struct.update(
        {
            "version": "v1",
            "environment": CHAT_PARTIAL_ENVIRONMENT,
            "components": [
                {
                    "name": "chat_agent",
                    "type": "AgentComponent",
                    "prompt_id": "p",
                    "toolset": [],
                }
            ],
            "prompts": [
                {
                    "name": "p",
                    "prompt_id": "p",
                    "unit_primitives": ["duo_chat"],
                    "template": {"system": "You are a chat agent."},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="outside the AI Gateway repository"):
        _load_flow_from_inline_config(struct, "v1")


def test_engine_owned_config_is_normalized_before_it_reaches_chat_flow():
    factory = engine_owned_factory(engine_owned_chat_partial_config())

    assert factory.func is ChatFlow
    config = factory.keywords["config"]
    assert config.routers == [{"from": "chat_agent", "to": "end"}]
    assert config.flow.entry_point == "chat_agent"
    assert config.components == [
        {
            "name": "chat_agent",
            "type": "AgentComponent",
            "toolset": [],
            "prompt_id": "chat_agent_prompt",
        }
    ]


def test_engine_owned_config_rejects_declared_routers():
    config = engine_owned_chat_partial_config(
        routers=[{"from": "chat_agent", "to": "end"}]
    )

    with pytest.raises(ValueError, match="reject declared routers"):
        engine_owned_factory(config)


def test_engine_owned_config_requires_a_component_name():
    config = engine_owned_chat_partial_config(
        components=[{"type": "AgentComponent", "toolset": []}]
    )

    with pytest.raises(ValueError, match="declare a name"):
        engine_owned_factory(config)


def test_engine_owned_config_keeps_the_chat_partial_validations():
    config = engine_owned_chat_partial_config(
        components=[
            {"name": "one", "type": "AgentComponent", "toolset": []},
            {"name": "two", "type": "AgentComponent", "toolset": []},
        ]
    )

    with pytest.raises(ValueError, match="exactly one component"):
        engine_owned_factory(config)


def test_engine_owned_config_requires_the_v1_schema():
    config = engine_owned_chat_partial_config(
        ExperimentalPartialFlowConfig, version="experimental"
    )

    with pytest.raises(ValueError, match="v1 config schema"):
        engine_owned_factory(config)


class TestRegistryIdentity:
    """Only a registry load sets the id that makes a flow metadata-only."""

    @pytest.mark.parametrize(
        "config_id", ["./bl_security", "bl_security/", "x/../bl_security", ".", ".."]
    )
    def test_the_registry_rejects_a_flow_id_that_is_not_a_bare_name(self, config_id):
        # Such an id could load bl_security under an id that is not in METADATA_ONLY_FLOWS.
        request = RegistryFlowRequest(
            config_id=config_id, schema_version="experimental", version="1.0.0"
        )

        with pytest.raises(ValueError, match="Failed to load flow"):
            resolve_flow(request)

    def test_the_registry_loads_bl_security_by_its_bare_id(self):
        resolved = resolve_flow(
            RegistryFlowRequest(
                config_id="bl_security", schema_version="experimental", version="1.0.0"
            )
        )

        assert resolved.factory.keywords["config"].config_id in METADATA_ONLY_FLOWS

    @pytest.mark.parametrize("schema_version", ["experimental", "v1"])
    @pytest.mark.parametrize("environment", ["ambient", CHAT_PARTIAL_ENVIRONMENT])
    def test_an_inline_config_has_no_registry_id(self, schema_version, environment):
        struct = struct_pb2.Struct()
        struct.update(
            {
                "version": schema_version,
                "environment": environment,
                "name": "bl_security",
                "config_id": "bl_security",
                "_config_id": "bl_security",
                "_config_version": "1.0.0",
                "components": [{"name": "a", "type": "AgentComponent"}],
                "routers": [],
                "flow": {"entry_point": "a"},
            }
        )

        config = _convert_struct_to_flow_config(struct, schema_version)

        assert config.config_id is None
        assert config.config_version is None


# Resolving a fetched flow request, and building a config from a fetched definition.


def _fetched_definition(**overrides) -> dict:
    return {
        "version": "v1",
        "environment": "ambient",
        "components": [
            {
                "name": "agent",
                "type": "AgentComponent",
                "prompt_id": "fetched_prompt",
                "toolset": [],
            }
        ],
        "routers": [{"from": "agent", "to": "end"}],
        "flow": {"entry_point": "agent"},
        "prompts": [
            {
                "prompt_id": "fetched_prompt",
                "name": "Fetched prompt",
                "unit_primitives": ["duo_agent_platform"],
                # A stored definition loses the author's key order, like an inline Struct.
                "prompt_template": {"user": "{{goal}}", "system": "You help."},
            }
        ],
        **overrides,
    }


def test_resolve_builds_a_fetched_flow_with_the_definition_builder():
    resolved = resolve_flow(FetchedFlowRequest(workflow_definition="my_flow/v1"))

    assert resolved.factory.func is FetchedFlow
    assert (
        resolved.factory.keywords["config_builder"]
        is flow_config_from_fetched_definition
    )
    assert resolved.tracking_fields() == {"schema_version": "v1"}


def test_a_v1_definition_builds_a_config():
    config = flow_config_from_fetched_definition(_fetched_definition())

    assert isinstance(config, FlowConfig)
    assert config.environment == "ambient"


def test_prompt_roles_get_the_canonical_order():
    config = flow_config_from_fetched_definition(_fetched_definition())

    template = config.prompts[0].prompt_template
    assert list(template) == ["system", "user"]


def test_the_served_definition_is_not_changed():
    definition = _fetched_definition()

    flow_config_from_fetched_definition(definition)

    assert list(definition["prompts"][0]["prompt_template"]) == ["user", "system"]


@pytest.mark.parametrize("version", ["experimental", None, "v2"])
def test_only_v1_definitions_are_accepted(version):
    with pytest.raises(ValueError, match="must have version 'v1'"):
        flow_config_from_fetched_definition(_fetched_definition(version=version))


def test_a_fetched_definition_with_a_template_prompt_is_refused():
    # Built here, it would fail later with an obscure error instead of this one.
    definition = _fetched_definition()
    definition["prompts"][0] = {
        "prompt_id": "fetched_prompt",
        "name": "Fetched prompt",
        "unit_primitives": ["duo_agent_platform"],
        "template": {"system": "You help."},
    }

    # A fetched definition is never chat-partial, so the environment rule is the first to refuse it.
    with pytest.raises(
        ValueError, match="declares 'template' only in the 'chat-partial'"
    ):
        flow_config_from_fetched_definition(definition)


def test_a_chat_partial_definition_is_rejected():
    with pytest.raises(ValueError, match=CHAT_PARTIAL_ENVIRONMENT):
        flow_config_from_fetched_definition(
            _fetched_definition(environment=CHAT_PARTIAL_ENVIRONMENT)
        )


def test_an_invalid_definition_raises_a_value_error():
    with pytest.raises(ValueError):
        flow_config_from_fetched_definition(
            _fetched_definition(components="not-a-list")
        )


def test_prompts_pass_the_security_check():
    with patch(
        "duo_workflow_service.workflows.registry._validate_flow_config_prompts",
        side_effect=SecurityException("dangerous content"),
    ) as validate:
        with pytest.raises(SecurityException):
            flow_config_from_fetched_definition(_fetched_definition())

    validate.assert_called_once()
