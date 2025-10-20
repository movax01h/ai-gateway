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

from duo_workflow_service.agent_platform.experimental.flows import (
    Flow as ExperimentalFlow,
)
from duo_workflow_service.agent_platform.experimental.flows import (
    FlowConfig as ExperimentalFlowConfig,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    list_configs as experimental_list_configs,
)
from duo_workflow_service.agent_platform.v1 import list_configs as v1_list_configs
from duo_workflow_service.agent_platform.v1.flows import Flow as V1Flow
from duo_workflow_service.agent_platform.v1.flows import FlowConfig as V1FlowConfig
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


def _flow_factory(
    flow_cls: FlowFactory,
    config: Union[ExperimentalFlowConfig, V1FlowConfig],
) -> FlowFactory:
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

    args = {
        "tools_override": agent_component["toolset"],
        "prompt_template_id_override": agent_component["prompt_id"],
        "prompt_template_version_override": agent_component.get("prompt_version"),
        "use_custom_adapter": True,
    }

    if prompt_template_override := (config.prompts[0] if config.prompts else None):
        args["prompt_template_override"] = prompt_template_override

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
        return software_development.Workflow  # for backwards compatibility

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


def parse_workflow_definition(
    workflow_definition: str,
) -> Tuple[str, str]:
    """Resolves a workflow definition string to its corresponding components."""
    flow_version = Path(workflow_definition).name
    flow_config_path = Path(workflow_definition).parent

    return flow_version, str(flow_config_path)


def list_configs():
    configs = []
    for config_list in _FLOW_CONFIGS_BY_VERSION.values():
        configs.extend(config_list())

    return configs
