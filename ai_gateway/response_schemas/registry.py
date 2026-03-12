import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Type

import structlog.stdlib
from poetry.core.constraints.version import Version, parse_constraint

from ai_gateway.response_schemas.base import BaseAgentOutput, BaseResponseSchemaRegistry
from ai_gateway.response_schemas.converter import json_schema_to_pydantic

__all__ = ["ResponseSchemaRegistered", "ResponseSchemaRegistry"]


log = structlog.stdlib.get_logger("schema_registry")


class ResponseSchemaRegistered(NamedTuple):
    """Container for registered schema versions."""

    versions: dict[str, dict]


class ResponseSchemaRegistry(BaseResponseSchemaRegistry):
    key_schema_type_base: str = "base"

    @lru_cache(maxsize=30)
    def get(self, schema_id: str, schema_version: str) -> Type[BaseAgentOutput]:
        """Get a Pydantic model from a specific JSON schema definition."""
        try:
            schema_path = self._resolve_id(schema_id)
            registered_schema = self._load_schema_definition(schema_id, schema_path)
            schema_dict = self._get_schema_config(
                registered_schema.versions, schema_version
            )
            return json_schema_to_pydantic(schema_dict)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load schema '{schema_id}': {e}") from e

    def _resolve_id(self, schema_id: str) -> Path:
        """Find schema directory path."""
        base_path = Path(__file__).parent
        schema_path = (
            base_path / "definitions" / schema_id / self.key_schema_type_base
        ).resolve()

        # Assert containment — the resolved path must still be inside base_path
        if not schema_path.is_relative_to(base_path.resolve()):
            raise ValueError("Invalid schema_id: path traversal detected")

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")

        return schema_path

    def _load_schema_definition(
        self,
        schema_id: str,
        schema_path: Path,
    ) -> ResponseSchemaRegistered:
        """Load all JSON schema versions from directory."""
        versions = {}

        for version_file in schema_path.glob("*.json"):
            version = version_file.stem  # "1.0.0" from "1.0.0.json"
            with open(version_file, "r", encoding="utf-8") as fp:
                versions[version] = json.load(fp)

        if not versions:
            raise ValueError(f"No JSON files found for schema: {schema_id}")

        log.info("Loaded schema", schema_id=schema_id, versions=list(versions.keys()))
        return ResponseSchemaRegistered(versions=versions)

    def _get_schema_config(
        self,
        versions: dict[str, dict],
        schema_version: str,
    ) -> dict:
        """Resolves version constraint and return matching response_schemas."""

        # Parse constraint according to poetry rules. See
        # https://python-poetry.org/docs/dependency-specification/#version-constraints
        constraint = parse_constraint(schema_version)
        all_versions = [Version.parse(v) for v in versions.keys()]

        # Only stable versions for non-exact constraints (e.g. ^1.0.0, ~1.0.0)
        if not constraint.is_simple():
            all_versions = [v for v in all_versions if v.is_stable()]

        compatible = list(filter(constraint.allows, all_versions))

        if not compatible:
            log.error(
                f"No compatible versions found for schema: {schema_version}",
                versions=versions,
                schema_version=schema_version,
            )
            raise ValueError(
                f"No compatible versions found for schema: {schema_version}"
            )

        # Return highest compatible version
        compatible.sort(reverse=True)
        resolved = str(compatible[0])

        log.info("Resolved version", requested=schema_version, resolved=resolved)
        return versions[resolved]
