"""Fixtures shared by the tests/ tree and co-located feature tests.

Feature tests live under ai/features/<domain>/<feature>/tests/ and cannot see tests/conftest.py, so the genuinely global
fixtures are promoted here. Promote a fixture only when it is useful to both trees; everything tests/-tree-specific
stays in tests/conftest.py.
"""

from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.fixture(name="gl_http_client", scope="function")
def gl_http_client_fixture() -> AsyncMock:
    """Return an ``AsyncMock`` that follows the ``GitlabHttpClient`` spec."""
    return AsyncMock(spec=GitlabHttpClient)


@pytest.fixture(name="project_mock", scope="function")
def project_mock_fixture() -> Project:
    """Return a minimal ``Project`` payload for tests that need a project."""
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        default_branch=None,
        languages=[],
        exclusion_rules=None,
    )
