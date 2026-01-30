import pytest

from ai_gateway.instrumentators.model_requests import (
    client_capabilities,
    gitlab_version,
)
from duo_workflow_service.client_capabilities import is_client_capable


@pytest.fixture(name="gl_version_supported")
def gl_version_supported_fixture():
    """Set GitLab version to a supported version (18.7.0+)."""
    gitlab_version.set("18.7.0")
    yield
    gitlab_version.set(None)


@pytest.fixture(name="gl_version_unsupported")
def gl_version_unsupported_fixture():
    """Set GitLab version to an unsupported version (<18.7.0)."""
    gitlab_version.set("18.3.0")
    yield
    gitlab_version.set(None)


def test_is_client_capable_in_supported_gitlab_version(
    gl_version_supported,
):  # pylint: disable=unused-argument
    """Test that is_client_capable returns True when the capability is in the set."""
    client_capabilities.set({"feature_a", "feature_b"})
    assert is_client_capable("feature_a") is True
    assert is_client_capable("feature_b") is True
    assert is_client_capable("feature_c") is False


def test_is_client_capable_in_unsupported_gitlab_version(
    gl_version_unsupported,
):  # pylint: disable=unused-argument
    """Test that is_client_capable returns False when the GitLab version is below 18.7.0."""
    client_capabilities.set({"feature_a", "feature_b"})
    assert is_client_capable("feature_a") is False
    assert is_client_capable("feature_b") is False
    assert is_client_capable("feature_c") is False
