from duo_workflow_service.server_capabilities import (
    CapabilityInfo,
    get_dws_capabilities,
    get_dws_capabilities_with_metadata,
)


def test_get_dws_capabilities():
    assert get_dws_capabilities() == [
        "tool_call_approval",
        "tool_call_pattern_approval",
        "flow_semantic_versioning",
    ]


def test_get_dws_capabilities_with_metadata():
    assert get_dws_capabilities_with_metadata() == [
        CapabilityInfo(name="tool_call_approval", metadata=""),
        CapabilityInfo(name="tool_call_pattern_approval", metadata=""),
        CapabilityInfo(name="flow_semantic_versioning", metadata=""),
    ]


def test_capabilities_are_consistent_between_functions():
    names_from_metadata = [
        capability.name for capability in get_dws_capabilities_with_metadata()
    ]

    assert names_from_metadata == get_dws_capabilities()
