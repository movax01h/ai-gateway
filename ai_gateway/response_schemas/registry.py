import json
from functools import cache, lru_cache
from pathlib import Path
from typing import NamedTuple, Type

import structlog.stdlib

from ai_gateway.response_schemas.base import BaseAgentOutput, BaseResponseSchemaRegistry
from ai_gateway.response_schemas.converter import json_schema_to_pydantic
from lib.feature_roots import feature_subdirs
from lib.version import resolve_version

__all__ = [
    "ResponseSchemaRegistered",
    "ResponseSchemaRegistry",
    "schema_index",
    "schema_roots",
]


log = structlog.stdlib.get_logger("schema_registry")


def schema_roots() -> list[Path]:
    """Return every dir that holds ``<schema_id>/base/<version>.json`` files.

    The roots are ``definitions/`` next to this module, each
    ``ai/features/<domain>/<feature>/schemas/``, and each ``ai/shared/<name>/schemas/``.
    """
    roots = [(Path(__file__).parent / "definitions").resolve()]
    roots += [found.path for found in feature_subdirs("schemas")]
    return roots


@cache
def schema_index() -> dict[str, Path]:
    """Map each schema ID to its ``base/`` dir, across every schema root.

    The ID is the path from the root to the dir that holds ``base/``, so
    ``analysis/code_review/base/`` has the ID ``analysis/code_review``. A ``base/``
    dir without a ``.json`` file, such as one that a move leaves behind, is skipped.
    The index is built once per process; tests clear it with
    ``schema_index.cache_clear()``.

    Raises:
        ValueError: If two roots define the same ID. A copy left behind after a
            move would otherwise decide which version a flow gets by scan order.
    """
    index: dict[str, Path] = {}
    for root in schema_roots():
        for base_dir in sorted(root.rglob("base")):
            if not base_dir.is_dir() or not any(base_dir.glob("*.json")):
                continue
            schema_id = base_dir.parent.relative_to(root).as_posix()
            if schema_id in index:
                raise ValueError(
                    f"Duplicate schema id {schema_id!r}: defined in "
                    f"{index[schema_id]} and {base_dir}"
                )
            index[schema_id] = base_dir
    return index


class ResponseSchemaRegistered(NamedTuple):
    """Container for registered schema versions."""

    versions: dict[str, dict]


class ResponseSchemaRegistry(BaseResponseSchemaRegistry):
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
        """Return the ``base/`` dir that :func:`schema_index` lists for ``schema_id``.

        ``schema_id`` is only a key into the index, so it never builds a path.

        Raises:
            ValueError: If two roots define the same schema ID.
            FileNotFoundError: If no root defines ``schema_id``.
        """
        schema_path = schema_index().get(schema_id)
        if schema_path is None:
            raise FileNotFoundError("Schema not found")
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
        resolved = resolve_version(versions.keys(), schema_version)
        return versions[resolved]
