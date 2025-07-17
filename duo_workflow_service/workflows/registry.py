import inspect
from functools import partial
from pathlib import Path
from typing import Callable, Optional, TypeAlias

from duo_workflow_service.agent_platform import experimental
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

_FLOW_BY_VERSIONS = {
    "experimental": (experimental.flows.FlowConfig, experimental.flows.Flow),
}


def resolve_workflow_class(
    workflow_definition: Optional[str],
) -> FlowFactory:
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
