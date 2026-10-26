from unittest.mock import patch

import pytest

from ai_gateway.prompts import feature_roots
from duo_workflow_service.agent_platform.v1.flows import flow_config


@pytest.fixture(name="feature_roots_registered", autouse=True)
def feature_roots_registered_fixture():
    """Register the feature roots, as the servers do at boot.

    The developer configs and their prompt fragments live under ai/features/, so
    ``FlowConfig.from_yaml_config("developer", ...)`` and the ``developer/common/``
    includes only resolve once discovery has run. The registries are module
    globals, so they are restored afterwards to keep other tests on the same
    worker isolated.
    """
    with (
        patch.dict(flow_config._FEATURE_FLOW_ROOTS),
        patch.dict(feature_roots._FEATURE_PROMPT_ROOTS),
    ):
        flow_config.discover_feature_flow_configs()
        feature_roots.discover_feature_prompts()
        yield
