from typing import Any, Literal, Optional, Type

import structlog
from jsonschema import Draft7Validator, ValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, create_model

from ai_gateway.response_schemas.base import BaseAgentOutput

FINAL_RESPONSE_INSTRUCTIONS = """MANDATORY COMPLETION TOOL: You MUST use this tool to provide your final answer when you
    have completed the user's request. This is the ONLY way to properly end the conversation and deliver your response to
    the user.

    CRITICAL INSTRUCTIONS:
    1. You MUST call this tool when you have gathered all necessary information and completed the requested task
    2. You MUST provide ALL required fields defined in this tool's schema
    3. You MUST follow the structure and format specified by each field
    4. You MUST respect field descriptions, constraints, and data types
    5. Do NOT continue using other tools once you have the information needed to complete this response

    Carefully read each field's description below and provide accurate, complete information for each required field."""

JSON_SCHEMA_PYDANTIC_TYPE_MAP: dict = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# Recursion limit for nested JSON schema objects
RECURSION_LIMIT = 10

log = structlog.stdlib.get_logger("json_schema_to_pydantic")


def json_schema_to_pydantic(
    schema: dict[str, Any],
    _depth: int = 0,
) -> Type[BaseAgentOutput]:
    """Convert JSON schema to Pydantic model.

    Args:
        schema: JSON Schema dictionary to convert
        _depth: Internal recursion depth counter (starts at 0)

    Raises:
        ValueError: If schema is invalid or nesting exceeds RECURSION_LIMIT
    """

    # Prevent DoS via deeply nested schemas
    if _depth > RECURSION_LIMIT:
        raise ValueError(
            f"JSON schema nesting exceeds maximum depth of {RECURSION_LIMIT}."
        )

    # Validate the schema using version specified default to Draft7 if not specified
    try:
        validator_class = (
            validator_for(schema) if "$schema" in schema else Draft7Validator
        )
        validator_class.check_schema(schema)
    except ValidationError as e:
        raise ValueError(f"Invalid JSON schema: {e}")

    if "title" not in schema:
        raise ValueError("Schema must have 'title'")
    if schema.get("type") != "object":
        raise ValueError("Schema type must be 'object'")

    title: str = schema["title"]
    properties: dict[str, Any] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = get_python_type(field_schema, field_name, _depth=_depth)
        is_required = field_name in required

        # Build Field kwargs with description and constraints
        field_kwargs = {
            "description": field_schema.get("description", ""),
            **extract_field_constraints(field_schema),
        }

        # Handle default value (only for optional fields)
        if not is_required and "default" in field_schema:
            field_kwargs["default"] = field_schema["default"]
        elif not is_required:
            field_kwargs["default"] = None

        if is_required:
            fields[field_name] = (field_type, Field(**field_kwargs))
        else:
            fields[field_name] = (Optional[field_type], Field(**field_kwargs))

    model = create_model(title, __base__=BaseAgentOutput, **fields)
    model.model_config = ConfigDict(title=title, frozen=True)
    model.tool_title = title
    model.to_output = lambda self: self.model_dump()

    user_description = schema.get("description", "")

    if user_description:
        model.__doc__ = f"{FINAL_RESPONSE_INSTRUCTIONS}\n\n{user_description}"
    else:
        model.__doc__ = FINAL_RESPONSE_INSTRUCTIONS

    log.info("Created Pydantic model from schema", title=title)
    return model


def get_python_type(field_schema: dict, parent_field_name: str = "", _depth: int = 0):
    """Map JSON schema type to Python type, handling nested structures.

    Args:
        field_schema: JSON schema for the field
        parent_field_name: Name of parent field (for auto-generating nested titles)
        _depth: Internal recursion depth counter

    Returns:
        Python type corresponding to the JSON schema

    Raises:
        ValueError: If schema nesting exceeds RECURSION_LIMIT
    """

    # Handle const -> Literal[single_value]
    if "const" in field_schema:
        return Literal[field_schema["const"]]

    # Handle enum -> Literal[values...]
    if "enum" in field_schema:
        enum_values = tuple(field_schema["enum"])
        return Literal[enum_values]

    field_type = field_schema.get("type")

    # Handle arrays with items (typed arrays)
    if field_type == "array" and "items" in field_schema:
        items_schema = field_schema["items"]
        item_type = get_python_type(items_schema, parent_field_name, _depth=_depth + 1)
        return list[item_type]  # type: ignore[valid-type]

    # Handle nested objects with properties
    if field_type == "object" and "properties" in field_schema:
        # Generate title if missing
        if "title" not in field_schema:
            title = (
                f"{parent_field_name.replace('_', ' ').title().replace(' ', '')}Item"
            )
            nested_schema = {**field_schema, "title": title}
        else:
            nested_schema = field_schema
        return json_schema_to_pydantic(nested_schema, _depth=_depth + 1)

    return JSON_SCHEMA_PYDANTIC_TYPE_MAP.get(field_schema.get("type", "string"), str)


def extract_field_constraints(  # pylint: disable=too-many-branches
    field_schema: dict,
) -> dict[str, Any]:
    """Extract Field constraint parameters from JSON Schema."""
    constraints = {}
    field_type = field_schema.get("type")

    # Numeric constraints (integer/number):
    # JSON Schema ref: https://json-schema.org/understanding-json-schema/reference/numeric
    # Pydantic ref: https://docs.pydantic.dev/latest/api/standard_library_types/#integers
    if field_type in ("integer", "number"):
        if "minimum" in field_schema:
            constraints["ge"] = field_schema["minimum"]
        if "maximum" in field_schema:
            constraints["le"] = field_schema["maximum"]
        if "exclusiveMinimum" in field_schema:
            constraints["gt"] = field_schema["exclusiveMinimum"]
        if "exclusiveMaximum" in field_schema:
            constraints["lt"] = field_schema["exclusiveMaximum"]
        if "multipleOf" in field_schema:
            constraints["multiple_of"] = field_schema["multipleOf"]

    # String constraints
    # JSON Schema ref: https://json-schema.org/understanding-json-schema/reference/string
    # Pydantic ref: https://docs.pydantic.dev/latest/api/standard_library_types/#strings
    if field_type == "string":
        if "minLength" in field_schema:
            constraints["min_length"] = field_schema["minLength"]
        if "maxLength" in field_schema:
            constraints["max_length"] = field_schema["maxLength"]
        if "pattern" in field_schema:
            constraints["pattern"] = field_schema["pattern"]

    # Array constraints
    # JSON Schema ref: https://json-schema.org/understanding-json-schema/reference/array
    # Pydantic ref: https://docs.pydantic.dev/latest/api/standard_library_types/#lists
    if field_type == "array":
        if "minItems" in field_schema:
            constraints["min_length"] = field_schema["minItems"]
        if "maxItems" in field_schema:
            constraints["max_length"] = field_schema["maxItems"]

    # Examples
    # JSON Schema ref: https://json-schema.org/draft-07/json-schema-validation#rfc.section.10.4
    # Pydantic ref: https://docs.pydantic.dev/latest/api/json_schema/#pydantic.json_schema.Examples
    if "examples" in field_schema:
        constraints["examples"] = field_schema["examples"]

    return constraints
