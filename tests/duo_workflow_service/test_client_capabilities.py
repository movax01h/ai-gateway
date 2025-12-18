from duo_workflow_service.client_capabilities import (
    client_capabilities,
    is_client_capable,
)


def test_is_client_capable():
    """Test that is_client_capable returns True when the capability is in the set."""
    client_capabilities.set({"feature_a", "feature_b"})
    assert is_client_capable("feature_a") is True
    assert is_client_capable("feature_b") is True
    assert is_client_capable("feature_c") is False
