import inspect
from pathlib import Path
from typing import Optional

from duo_workflow_service.workflows import (
    chat,
    convert_to_gitlab_ci,
    search_and_replace,
    software_development,
)

from .abstract_workflow import TypeWorkflow

current_directory = Path(__file__).parent

_WORKFLOWS: list[TypeWorkflow] = [
    software_development.Workflow,
    search_and_replace.Workflow,
    convert_to_gitlab_ci.Workflow,
    chat.Workflow,
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


def resolve_workflow_class(workflow_definition: Optional[str]) -> TypeWorkflow:
    if workflow_definition:
        return _WORKFLOWS_LOOKUP[workflow_definition]
    return software_development.Workflow  # for backwards compatibility
