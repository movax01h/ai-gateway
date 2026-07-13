import io
import json
import logging
import os
from typing import Dict
from unittest.mock import patch

import pytest
import structlog

from duo_workflow_service.structured_logging import LoggingConfig, setup_logging
from duo_workflow_service.tracking.monitoring_context import (
    MonitoringContext,
    current_monitoring_context,
)
from lib.context.request_metadata import gitlab_instance_id, gitlab_realm


@pytest.mark.parametrize(
    "env_vars, expected_config",
    [
        (
            {},
            {
                "level": "INFO",
                "json_format": False,
                "to_file": None,
                "environment": "development",
            },
        ),
        (
            {
                "DUO_WORKFLOW_SERVICE_ENVIRONMENT": "production",
            },
            {
                "level": "INFO",
                "json_format": True,
                "to_file": None,
                "environment": "production",
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


@pytest.fixture(name="json_logging_setup")
def json_logging_setup_fixture():
    """Set up JSON logging and capture output, restoring state after the test."""
    # pylint: disable=direct-environment-variable-reference
    stream = io.StringIO()
    with patch.dict(
        os.environ,
        {
            "DUO_WORKFLOW_SERVICE_ENVIRONMENT": "production",
            "DUO_WORKFLOW_LOGGING__JSON_FORMAT": "true",
        },
    ):
        setup_logging()

    # Replace the root logger's handler with one writing to our stream
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    stream_handler = logging.StreamHandler(stream)
    if root_logger.handlers:
        stream_handler.setFormatter(root_logger.handlers[0].formatter)
    root_logger.handlers = [stream_handler]

    yield stream

    # Restore original handlers
    root_logger.handlers = original_handlers
    gitlab_realm.set(None)
    gitlab_instance_id.set(None)
    current_monitoring_context.set(MonitoringContext())


def test_add_gitlab_realm_processor_with_value(json_logging_setup):
    """Test that add_gitlab_realm processor injects gitlab_realm when set."""
    gitlab_realm.set("saas")
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert log_entry.get("gitlab_realm") == "saas"


def test_add_gitlab_realm_processor_without_value(json_logging_setup):
    """Test that add_gitlab_realm processor omits gitlab_realm when not set."""
    gitlab_realm.set(None)
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert "gitlab_realm" not in log_entry


def test_add_gitlab_instance_id_processor_with_value(json_logging_setup):
    """Test that add_gitlab_instance_id processor injects gitlab_instance_id when set."""
    gitlab_instance_id.set("test-instance-123")
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert log_entry.get("gitlab_instance_id") == "test-instance-123"


def test_add_gitlab_instance_id_processor_without_value(json_logging_setup):
    """Test that add_gitlab_instance_id processor omits gitlab_instance_id when not set."""
    gitlab_instance_id.set(None)
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert "gitlab_instance_id" not in log_entry


def test_add_workflow_identity_processor_with_values(json_logging_setup):
    """Test that workflow identity fields are injected when set."""
    current_monitoring_context.set(
        MonitoringContext(
            workflow_definition="developer/v1",
            flow_id="developer",
            flow_version="1.2.3",
            schema_version="v1",
        )
    )
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert log_entry.get("workflow_definition") == "developer/v1"
    assert log_entry.get("flow_name") == "developer"
    assert log_entry.get("item_version") == "1.2.3"
    assert log_entry.get("schema_version") == "v1"


def test_add_workflow_identity_processor_without_values(json_logging_setup):
    """Test that workflow identity fields are omitted when not set."""
    current_monitoring_context.set(MonitoringContext())
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert "workflow_definition" not in log_entry
    assert "flow_name" not in log_entry
    assert "item_version" not in log_entry
    assert "schema_version" not in log_entry


def test_add_workflow_identity_processor_with_partial_values(json_logging_setup):
    """Test that each workflow identity field is omitted independently."""
    current_monitoring_context.set(
        MonitoringContext(
            workflow_definition="developer/v1",
            flow_id="",
            flow_version="1.2.3",
            schema_version=None,
        )
    )
    log = structlog.stdlib.get_logger("test")
    log.info("test event")

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert log_entry.get("workflow_definition") == "developer/v1"
    assert "flow_name" not in log_entry
    assert log_entry.get("item_version") == "1.2.3"
    assert "schema_version" not in log_entry


def test_add_workflow_identity_processor_does_not_overwrite_extra_values(
    json_logging_setup,
):
    """Test that explicit log extra fields are not overwritten by context values."""
    current_monitoring_context.set(
        MonitoringContext(
            workflow_definition="context-definition",
            flow_id="context-flow",
            flow_version="1.2.3",
            schema_version="v1",
        )
    )
    log = structlog.stdlib.get_logger("test")
    log.info(
        "test event",
        workflow_definition="extra-definition",
        item_version="9.9.9",
    )

    output = json_logging_setup.getvalue()
    assert output, "Expected log output but got none"
    log_entry = json.loads(output.strip().splitlines()[-1])
    assert log_entry.get("workflow_definition") == "extra-definition"
    assert log_entry.get("flow_name") == "context-flow"
    assert log_entry.get("item_version") == "9.9.9"
    assert log_entry.get("schema_version") == "v1"
