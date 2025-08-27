import inspect
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeAlias, TypeVar

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from duo_workflow_service.agent_platform import experimental
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from duo_workflow_service.workflows import (
    chat,
    convert_to_gitlab_ci,
    issue_to_merge_request,
    search_and_replace,
    software_development,
)
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TypeWorkflow,
)

current_directory = Path(__file__).parent
_WORKFLOWS: list[TypeWorkflow] = [
    software_development.Workflow,
    search_and_replace.Workflow,
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

FlowFactory: TypeAlias = Callable[..., AbstractWorkflow]

FlowConfigT = TypeVar("FlowConfigT", bound=FlowConfig)

_FLOW_BY_VERSIONS = {
    "experimental": (experimental.flows.FlowConfig, experimental.flows.Flow),
}


def _convert_struct_to_flow_config(
    struct: struct_pb2.Struct,
    flow_config_schema_version: str,
    flow_config_cls: Type[FlowConfigT],
) -> FlowConfigT:
    try:
        _FLOW_BY_VERSIONS[flow_config_schema_version]
    except KeyError:
        raise ValueError(
            f"Unsupported schema version: {flow_config_schema_version}. "
            f"Supported versions: {list(_FLOW_BY_VERSIONS.keys())}"
        ) from None
    config_dict: Dict[str, Any] = MessageToDict(struct)
    config_dict["version"] = flow_config_schema_version

    return flow_config_cls(**config_dict)


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
            return partial(flow_cls, config=config)
        except Exception as e:
            raise ValueError(
                f"Failed to create flow from FlowConfig protobuf: {e}"
            ) from e

    if not workflow_definition:
        return software_development.Workflow  # for backwards compatibility

    if workflow_definition in _WORKFLOWS_LOOKUP:
        return _WORKFLOWS_LOOKUP[workflow_definition]

    flow_version = Path(workflow_definition).name
    flow_config_path = Path(workflow_definition).parent
    if flow_version not in _FLOW_BY_VERSIONS:
        raise ValueError(f"Unknown Flow version: {flow_version}")

    try:
        flow_config_cls, flow_cls = _FLOW_BY_VERSIONS[flow_version]

        config = flow_config_cls.from_yaml_config(str(flow_config_path))
        return partial(flow_cls, config=config)  # dynamic flow type
    except Exception:
        raise ValueError(f"Unknown Flow: {workflow_definition}")
