"""The prompt a component names is part of the Flow Registry's public contract, so its fields are pinned here."""

import pytest
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
    AgentComponentBase,
)
from duo_workflow_service.agent_platform.v1.components.one_off.component import (
    OneOffComponent,
)
from duo_workflow_service.agent_platform.v1.components.prompt_reference import (
    PromptReference,
    prompt_reference_of,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.component import (
    SupervisorAgentComponentV2,
)


def _class_of(component):
    # The registry's decorator hands back a function for the components it registers, with the class behind it.
    return getattr(component, "__wrapped__", component)


_COMPONENTS = [
    _class_of(component)
    for component in (
        AgentComponentBase,
        AgentComponent,
        SupervisorAgentComponent,
        SupervisorAgentComponentV2,
        OneOffComponent,
    )
]


@pytest.mark.parametrize("component", _COMPONENTS, ids=lambda c: c.__name__)
def test_a_component_names_its_prompt_by_the_two_fields_flow_authors_write(component):
    # Renaming or retyping either one breaks every flow that writes it, so a change here has to be deliberate.
    fields = component.model_fields

    assert issubclass(component, PromptReference)
    assert fields["prompt_id"].annotation is str
    assert fields["prompt_id"].is_required()
    assert fields["prompt_version"].annotation == (str | None)
    assert fields["prompt_version"].default is None


@pytest.mark.parametrize(
    "version,expected",
    [(None, False), ("", False), ("^1.0.0", True)],
    ids=["inline", "empty", "stored"],
)
def test_a_reference_is_stored_only_when_it_names_a_version(version, expected):
    assert PromptReference(prompt_id="p", prompt_version=version).is_stored is expected


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"name": "a", "type": "AgentComponent", "prompt_id": "p"}, ("p", False)),
        (
            {
                "name": "a",
                "type": "AgentComponent",
                "prompt_id": "p",
                "prompt_version": "^1.0.0",
            },
            ("p", True),
        ),
    ],
    ids=["inline", "stored"],
)
def test_a_config_entry_is_read_without_knowing_its_component(entry, expected):
    reference = prompt_reference_of(entry)

    assert (reference.prompt_id, reference.is_stored) == expected


def test_a_config_entry_that_names_no_prompt_reads_as_none():
    assert prompt_reference_of({"name": "h", "type": "HumanInputComponent"}) is None


def test_a_config_entry_with_a_version_and_no_prompt_id_is_refused():
    with pytest.raises(ValidationError, match="prompt_id"):
        prompt_reference_of(
            {"name": "a", "type": "AgentComponent", "prompt_version": "^1.0.0"}
        )
