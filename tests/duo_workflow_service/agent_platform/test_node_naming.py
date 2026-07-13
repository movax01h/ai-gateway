import pytest

from duo_workflow_service.agent_platform.node_naming import component_name_from_node


@pytest.mark.parametrize(
    ("node_name", "expected"),
    [
        ("researcher#agent", "researcher"),
        ("researcher#tools", "researcher"),
        ("planner#final_response", "planner"),
        ("build_context", "build_context"),  # legacy node without a role suffix
        (
            "a#b#c",
            "a",
        ),  # only the first separator splits; component names never contain '#'
        (None, None),
        ("", None),
    ],
)
def test_component_name_from_node(node_name, expected):
    assert component_name_from_node(node_name) == expected
