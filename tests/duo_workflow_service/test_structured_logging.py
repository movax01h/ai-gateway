import os
from typing import Dict
from unittest.mock import patch

import pytest

from duo_workflow_service.structured_logging import LoggingConfig


@pytest.mark.parametrize(
    "env_vars, expected_config",
    [
        (
            {},
            {
                "level": "INFO",
                "json_format": True,
                "to_file": None,
                "environment": "development",
            },
        ),
        (
            {
                "DUO_WORKFLOW_LOGGING__LEVEL": "debug",
                "DUO_WORKFLOW_LOGGING__JSON_FORMAT": "false",
                "DUO_WORKFLOW_LOGGING__TO_FILE": "/ops/debug.log",
                "DUO_WORKFLOW_SERVICE_ENVIRONMENT": "production",
            },
            {
                "level": "DEBUG",
                "json_format": False,
                "to_file": "/ops/debug.log",
                "environment": "production",
            },
        ),
    ],
)
def test_logging_config(env_vars: Dict, expected_config: Dict):
    # pylint: disable=direct-environment-variable-reference
    with patch.dict(os.environ, env_vars, clear=True):
        config = LoggingConfig()
        assert config.model_dump() == expected_config
