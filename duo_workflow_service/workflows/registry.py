import inspect
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    Union,
    overload,
)

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.experimental.flows import (
    Flow as ExperimentalFlow,
)
from duo_workflow_service.agent_platform.experimental.flows import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    list_configs as experimental_list_configs,
)
from duo_workflow_service.agent_platform.utils import parse_workflow_definition
from duo_workflow_service.agent_platform.v1 import list_configs as v1_list_configs
from duo_workflow_service.agent_platform.v1.flows import Flow as V1Flow
from duo_workflow_service.agent_platform.v1.flows import FlowConfig as V1FlowConfig
from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.prompt_security import PromptSecurity
from duo_workflow_service.workflows import (
    chat,
    convert_to_gitlab_ci,
    issue_to_merge_request,
    software_development,
)
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TypeWorkflow,
)

current_directory = Path(__file__).parent
_WORKFLOWS: list[TypeWorkflow] = [
    software_development.Workflow,
    convert_to_gitlab_ci.Workflow,
    chat.Workflow,
    issue_to_merge_request.Workflow,
]

# Eg: {
#         'workflow': Workflow,
#         '/software_development': software_development.workflow.Workflow,
#         '/software_development/v1': software_development.v1.workflow.Workflow,
#     }
_WORKFLOWS_LOOKUP = {
    f"{Path(inspect.getfile(workflow_cls)).relative_to(current_directory).parent.with_suffix('')}": workflow_cls
    for workflow_cls in _WORKFLOWS
}

CHAT_AGENT_COMPONENT_ENVIRONMENT = "chat-partial"

FlowFactory: TypeAlias = Callable[..., AbstractWorkflow]

_FLOW_BY_VERSIONS: Dict[
    str, Tuple[Type[Union[ExperimentalFlowConfig, V1FlowConfig]], Any]
] = {
    "experimental": (ExperimentalFlowConfig, ExperimentalFlow),
    "v1": (V1FlowConfig, V1Flow),
}

_FLOW_CONFIGS_BY_VERSION = {
    "experimental": experimental_list_configs,
    "v1": v1_list_configs,
}


@overload
def _convert_struct_to_flow_config(
    struct: struct_pb2.Struct,
    flow_config_schema_version: str,
    flow_config_cls: Type[ExperimentalFlowConfig],
) -> ExperimentalFlowConfig: ...


@overload
def _convert_struct_to_flow_config(
    struct: struct_pb2.Struct,
    flow_config_schema_version: str,
    flow_config_cls: Type[V1FlowConfig],
) -> V1FlowConfig: ...


def _convert_struct_to_flow_config(
    struct: struct_pb2.Struct,
    flow_config_schema_version: str,
    flow_config_cls: Type[Union[ExperimentalFlowConfig, V1FlowConfig]],
) -> Union[ExperimentalFlowConfig, V1FlowConfig]:
    try:
        _FLOW_BY_VERSIONS[flow_config_schema_version]
    except KeyError:
        raise ValueError(
            f"Unsupported schema version: {flow_config_schema_version}. "
            f"Supported versions: {list(_FLOW_BY_VERSIONS.keys())}"
        ) from None
    config_dict: Dict[str, Any] = MessageToDict(struct)

    if flow_config_schema_version != config_dict["version"]:
        raise ValueError(
            (
                f"Schema version mismatch, declared version: {flow_config_schema_version},"
                f"but received: {config_dict['version']}"
            )
        )

    return flow_config_cls(**config_dict)


def _validate_flow_config_prompts(
    config: Union[ExperimentalFlowConfig, V1FlowConfig],
) -> None:
    """Validate all prompts in flow config for security issues.

    Flow configs are developer-controlled and must be secure by design. This
    function validates (not sanitizes) prompt content and raises SecurityException
    if dangerous patterns are detected

    Args:
        config: The flow configuration to validate.

    Raises:
        SecurityException: If any prompt contains dangerous content that would
            require sanitization. The exception message includes the prompt_id
            and role for easier debugging.
    """
    # Early return if no prompts to validate
    if not hasattr(config, "prompts") or not config.prompts:
        return

    # Check if prompts is iterable (handles Mock objects in tests)
    try:
        prompts_iter = iter(config.prompts)
    except TypeError:
        return

    # Validate only actual prompt text roles, not metadata/placeholders
    # Roles that contain actual prompt content to validate
    prompt_text_roles = {"system", "user", "assistant", "function"}

    # Validate each prompt configuration
    for prompt_config in prompts_iter:
        # Extract prompt_id and prompt_template from various formats
        if isinstance(prompt_config, InMemoryPromptConfig):
            prompt_id = prompt_config.prompt_id
            prompt_template = prompt_config.prompt_template
        elif isinstance(prompt_config, dict):
            prompt_id = prompt_config.get("prompt_id", "unknown")
            prompt_template = prompt_config.get("prompt_template", {})
        else:
            # Handle objects with attributes (e.g., Pydantic models, test mocks)
            prompt_id = getattr(prompt_config, "prompt_id", "unknown")
            prompt_template = getattr(prompt_config, "prompt_template", {})

        for role, text in prompt_template.items():
            # Skip non-text roles (e.g., "placeholder: history" is metadata, not content)
            if role not in prompt_text_roles:
                continue

            if text and isinstance(text, str):
                try:
                    # Validate using PromptSecurity with validate_only=True.
                    # This raises SecurityException if the content contains
                    # dangerous patterns instead of sanitizing them.
                    PromptSecurity.apply_security_to_tool_response(
                        response=text,
                        tool_name="flow_config_prompts",
                        validate_only=True,
                    )
                except SecurityException as e:
                    # Re-raise with clearer error message
                    # Show examples in original form developers would recognize (not HTML entities)
                    raise SecurityException(
                        f"Flow config prompt '{prompt_id}' (role: '{role}') contains dangerous content. "
                        f"Developer-controlled prompts cannot contain:\n"
                        f"  - Dangerous tags: <system>, <goal>\n"
                        f"  - HTML comments: <!-- ... -->\n"
                        f"  - Hidden unicode characters\n"
                        f"Please remove these patterns from your prompt configuration."
                    ) from e


def _flow_factory(
    flow_cls: FlowFactory,
    config: Union[ExperimentalFlowConfig, V1FlowConfig],
) -> FlowFactory:
    # Validate all prompts for security issues before creating the flow
    _validate_flow_config_prompts(config)

    if config.environment != CHAT_AGENT_COMPONENT_ENVIRONMENT:
        return partial(flow_cls, config=config)

    if len(config.components) != 1:
        raise ValueError(
            f"Chat-partial environment allows exactly one component, but received {len(config.components)}"
        )

    agent_component = config.components[0]

    if agent_component["type"] != "AgentComponent":
        raise ValueError(f"Invalid component type: {agent_component['type']}")

    if config.prompts and len(config.prompts) > 1:
        raise ValueError(
            f"Chat-partial environment expects exactly one prompt in prompt configuration, "
            f"but received {len(config.prompts)}"
        )

    prompt_version = agent_component.get("prompt_version")

    if config.prompts and prompt_version:
        raise ValueError(
            "Chat-partial environment expects either inline or in repository prompt configuration, but received both"
        )

    # Extract agent name from component for proper event tracking
    # This ensures chat-partial agents (e.g., Duo Planner, analytics_agent)
    # are tracked with their actual names instead of generic "chat"
    agent_name = agent_component.get("name")

    args = {
        "tools_override": agent_component["toolset"],
        "agent_name_override": agent_name,
    }

    if prompt_template_override := (config.prompts[0] if config.prompts else None):
        if isinstance(prompt_template_override, InMemoryPromptConfig):
            prompt_template = prompt_template_override.prompt_template
        else:
            prompt_template = prompt_template_override.get("prompt_template", {})

        args["system_template_override"] = prompt_template.get("system")

    return partial(chat.Workflow, **args)


def resolve_workflow_class(
    workflow_definition: Optional[str],
    flow_config: Optional[struct_pb2.Struct] = None,
    flow_config_schema_version: Optional[str] = None,
) -> FlowFactory:
    """Resolve a workflow class based on definition or FlowConfig protobuf.

    Args:
        workflow_definition: The workflow definition string (legacy approach)
        flow_config: the protobuf Struct containing flow config data
        flow_config_schema_version: version of the flow that's provided
        by default it's "experimental"

    Returns:
        A FlowFactory callable that creates workflow instances

    Raises:
        ValueError: If workflow cannot be resolved or is invalid
    """
    if flow_config and flow_config_schema_version:
        try:
            flow_config_cls, flow_cls = _FLOW_BY_VERSIONS[flow_config_schema_version]
            config = _convert_struct_to_flow_config(
                struct=flow_config,
                flow_config_schema_version=flow_config_schema_version,
                flow_config_cls=flow_config_cls,
            )
            return _flow_factory(flow_cls, config)
        except Exception as e:
            raise ValueError(
                f"Failed to create flow from FlowConfig protobuf: {e}"
            ) from e

    if not workflow_definition:
        # backward compatibility for old GitLab instances
        return software_development.Workflow

    if workflow_definition in _WORKFLOWS_LOOKUP:
        return _WORKFLOWS_LOOKUP[workflow_definition]

    flow_version, flow_config_path = parse_workflow_definition(workflow_definition)

    if flow_version not in _FLOW_BY_VERSIONS:
        raise ValueError(f"Unknown Flow version: {flow_version}")

    try:
        flow_config_cls, flow_cls = _FLOW_BY_VERSIONS[flow_version]

        config = flow_config_cls.from_yaml_config(flow_config_path)

        return _flow_factory(flow_cls, config)
    except Exception:
        raise ValueError(f"Unknown Flow: {workflow_definition}")


def list_configs():
    configs = []
    for config_list in _FLOW_CONFIGS_BY_VERSION.values():
        configs.extend(config_list())

    return configs
