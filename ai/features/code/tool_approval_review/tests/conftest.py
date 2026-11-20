from unittest.mock import patch

import pytest

from ai_gateway.prompts import feature_roots


@pytest.fixture(name="feature_roots_registered", autouse=True)
def feature_roots_registered_fixture():
    """Register the feature prompt roots, as the servers do at boot.

    The registry is a module global, so it's restored afterwards to keep other tests on the same worker isolated.
    """
    with patch.dict(feature_roots._FEATURE_PROMPT_ROOTS):
        feature_roots.discover_feature_prompts()
        yield
