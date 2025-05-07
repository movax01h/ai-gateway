import pytest

from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.registry import resolve_workflow_class
from duo_workflow_service.workflows.software_development import Workflow


def test_registry_resolve():
    # Test resolving default workflow
    assert resolve_workflow_class(None) == Workflow

    # Test resolving a non-existent workflow
    with pytest.raises(KeyError):
        resolve_workflow_class("non_existent_workflow")

    # Test that resolved class is a subclass of AbstractWorkflow
    resolved_class = resolve_workflow_class("software_development")
    assert issubclass(resolved_class, AbstractWorkflow)
    assert resolved_class == Workflow
