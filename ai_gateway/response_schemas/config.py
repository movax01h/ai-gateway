"""Configuration model for inline response schema definitions in flow YAML configs."""

import json
from typing import Any, Union

from pydantic import BaseModel, ConfigDict

__all__ = ["InlineResponseSchemaConfig"]


class InlineResponseSchemaConfig(BaseModel):
    """Inline response schema definition for use in the ``response_schemas`` flow config block.

    Mirrors the role of ``InMemoryPromptConfig`` for prompts: defined once in a
    top-level ``response_schemas`` block and referenced by components via
    ``response_schema_id`` (with no ``response_schema_version``).

    The ``definition`` field accepts either a YAML mapping (parsed as a Python dict)
    or a raw JSON string — useful when pasting a schema directly from an external
    source or JSON editor.  Both forms produce an identical dict when
    ``to_schema_dict()`` is called.

    See ``docs/flow_registry/v1.md`` for full YAML usage examples.
    """

    schema_id: str
    # Accepts either a YAML mapping (dict) or a raw JSON string.  Pydantic
    # validates the shape of this field (dict vs. str) but does NOT validate
    # that the value is a well-formed JSON Schema — that happens later, when
    # ``Flow.__init__`` calls ``register_schema()``, which runs
    # ``json_schema_to_pydantic()``.  A schema error will therefore surface as
    # a ``ValueError`` during flow construction rather than at parse time.
    definition: Union[dict[str, Any], str]

    model_config = ConfigDict(extra="forbid")

    def to_schema_dict(self) -> dict[str, Any]:
        """Return the definition as a plain Python dict.

        Parses the value with ``json.loads`` when ``definition`` is a string,
        otherwise returns the dict directly.

        Returns:
            JSON Schema as a Python dict suitable for ``json_schema_to_pydantic``.

        Raises:
            ValueError: If ``definition`` is a string that cannot be parsed as JSON.
        """
        if isinstance(self.definition, str):
            try:
                return json.loads(self.definition)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"response_schemas entry '{self.schema_id}': "
                    f"definition string is not valid JSON: {exc}"
                ) from exc
        return self.definition
