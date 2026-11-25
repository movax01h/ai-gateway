"""Fixtures shared by the tests/ tree and co-located feature tests.

Feature tests live under ai/features/<domain>/<feature>/tests/ and cannot see tests/conftest.py, so the genuinely global
fixtures are promoted here. Promote a fixture only when it is useful to both trees; everything tests/-tree-specific
stays in tests/conftest.py.

The file also holds one hook that makes each pytest process exit faster.
"""

import atexit
import gc
from unittest.mock import AsyncMock

import pytest

from ai_gateway.prompts.base import (
    _jinja2_variables_cache,
    _jinja2_variables_recursive_cache,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.hookimpl(trylast=True)
def pytest_unconfigure() -> None:
    """Skip the garbage collector passes when the pytest process exits.

    A test run leaves about 2 million objects on the heap of each pytest process, also in each xdist worker. When
    Python exits, the garbage collector scans the whole heap several times, and that takes seconds. ``gc.freeze()``
    moves all objects to a permanent generation that those scans skip.

    ``gc.freeze`` runs only when Python exits, so pytest's collection for unraisable exceptions runs before it. This
    hook runs after the other ``pytest_unconfigure`` hooks and registers the handler last. Thus the handler runs before
    the other exit handlers, and the other exit handlers still run.

    If this hook causes a problem, remove it. Nothing else depends on it.
    """
    atexit.register(gc.freeze)


@pytest.fixture(autouse=True)
def clear_jinja2_variables_caches() -> None:
    _jinja2_variables_cache.clear()
    _jinja2_variables_recursive_cache.clear()


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
