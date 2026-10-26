"""Fixtures shared by v1 flow tests."""

from unittest.mock import patch

import pytest

from ai_gateway.prompts import feature_roots


@pytest.fixture(name="feature_roots_registered", autouse=True, scope="module")
def feature_roots_registered_fixture():
    """Mirror application startup for tests that load feature Jinja templates directly.

    Restore the process-global registry after each test module so discovery does not leak into tests that exercise
    prompt-root isolation.
    """
    with patch.dict(feature_roots._FEATURE_PROMPT_ROOTS):
        feature_roots.discover_feature_prompts()
        yield
