"""Generates the checked-in JSON Schema for v1 Duo Agent Platform (DAP) flow configs.

``FlowConfig`` (``duo_workflow_service/agent_platform/v1/flows/flow_config.py``) is the
source of truth for the structure of a v1 flow definition YAML file. This module derives
a JSON Schema from it via Pydantic's ``model_json_schema()`` and writes it to
``flow_config.schema.json`` next to this file.

Whenever ``FlowConfig`` (or any model it references, e.g. flow inputs) changes, regenerate
the checked-in schema and commit it:

    make duo-workflow-flow-schema

The ``lint:duo_workflow_flow_schema`` CI job (via ``make check-duo-workflow-flow-schema``)
fails if the checked-in file drifts from the live model, so the regeneration step cannot
be skipped silently.
"""

import json
from pathlib import Path
from typing import Any

from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    INPUT_JSONSCHEMA_VERSION,
    FlowConfig,
)

__all__ = ["SCHEMA_PATH", "generate_flow_config_schema", "write_flow_config_schema"]

SCHEMA_PATH = Path(__file__).resolve().parent / "flow_config.schema.json"


def generate_flow_config_schema() -> dict[str, Any]:
    """Return the JSON Schema dict for FlowConfig, including the ``$schema`` declaration.

    Any ``$schema`` key Pydantic's schema generator might emit is dropped in favor of
    ``INPUT_JSONSCHEMA_VERSION``, which is placed first so the serialized JSON leads with
    ``$schema``, as is conventional for JSON Schema documents.
    """
    schema = FlowConfig.model_json_schema()
    schema.pop("$schema", None)
    return {"$schema": INPUT_JSONSCHEMA_VERSION, **schema}


def write_flow_config_schema() -> None:
    """Generate the JSON Schema for FlowConfig and write it to ``SCHEMA_PATH``."""
    schema = generate_flow_config_schema()
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    write_flow_config_schema()
