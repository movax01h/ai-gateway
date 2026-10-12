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

from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    INPUT_JSONSCHEMA_VERSION,
    FlowConfig,
)

__all__ = [
    "SCHEMA_PATH",
    "FlowConfigJsonSchemaGenerator",
    "generate_flow_config_schema",
    "write_flow_config_schema",
]

SCHEMA_PATH = Path(__file__).resolve().parent / "flow_config.schema.json"


class FlowConfigJsonSchemaGenerator(GenerateJsonSchema):
    """JSON Schema generator that closes every model object and declares ``$schema``.

    Each Pydantic model in the schema (the root and every entry in ``$defs``) gets
    ``additionalProperties: false``, so editors and linters flag unknown or misspelled
    keys. The models themselves keep Pydantic's default ``extra="ignore"``: setting
    ``extra="forbid"`` would reject custom flows that carry extra keys at runtime, which
    v1 must keep loading. The schema is therefore intentionally stricter than the
    runtime validation. A model that sets its own ``extra`` keeps the value Pydantic
    emits for it.
    """

    schema_dialect = INPUT_JSONSCHEMA_VERSION

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        json_schema = super().model_schema(schema)
        json_schema.setdefault("additionalProperties", False)
        return json_schema

    def generate(
        self, schema: CoreSchema, mode: JsonSchemaMode = "validation"
    ) -> JsonSchemaValue:
        # Pydantic does not emit ``$schema``. Put it first, as is conventional for JSON
        # Schema documents.
        return {"$schema": self.schema_dialect, **super().generate(schema, mode)}


def generate_flow_config_schema() -> dict[str, Any]:
    """Return the JSON Schema dict for FlowConfig, built with ``FlowConfigJsonSchemaGenerator``."""
    return FlowConfig.model_json_schema(schema_generator=FlowConfigJsonSchemaGenerator)


def write_flow_config_schema() -> None:
    """Generate the JSON Schema for FlowConfig and write it to ``SCHEMA_PATH``."""
    schema = generate_flow_config_schema()
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    write_flow_config_schema()
