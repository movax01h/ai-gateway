"""Shared validation utilities for DeterministicStepComponent."""

from typing import Any, TypeGuard

from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel


def extract_configured_params(inputs: list[Any]) -> set[str]:
    """Extract parameter names from input keys.

    Args:
        inputs: List of IOKey objects from either v1 or experimental state

    Returns:
        Set of parameter names extracted from the input keys
    """
    configured_params = set()
    for input_key in inputs:
        if hasattr(input_key, "alias") and input_key.alias:
            param_name = input_key.alias
        elif hasattr(input_key, "subkeys") and input_key.subkeys:
            param_name = input_key.subkeys[-1]
        else:
            param_name = str(input_key)
        configured_params.add(param_name)
    return configured_params


def select_validated_tool(
    tool: BaseTool, tool_name: str, configured_params: set[str]
) -> BaseTool:
    """Select the appropriate tool based on schema validation.

    Returns the tool from toolset if config matches current schema,
    or instantiates the superseded tool if config matches old schema.

    Args:
        tool: The tool from the toolset
        tool_name: Name of the tool (for error messages)
        configured_params: Set of parameter names from the config

    Returns:
        The validated tool instance

    Raises:
        ValueError: If validation fails or tool cannot be instantiated
    """
    # Early return: if tool has no schema or dict schema, accept as-is
    if not _has_args_schema(tool.args_schema):
        return tool

    # Try to match against current tool schema first
    current_error = None
    try:
        validate_against_schema(tool.args_schema, configured_params)
        return tool
    except ValueError as e:
        # Current tool validation failed - will try superseded tool below
        current_error = e

    # Walk the full supersession chain looking for a schema that matches
    # the configured parameters (e.g. ShellCommandWithTimeout -> ShellCommand
    # -> RunCommand).
    candidate = tool
    while hasattr(candidate, "supersedes") and candidate.supersedes:
        try:
            superseded_tool_instance = candidate.supersedes(metadata=tool.metadata)
        except Exception as e:
            raise ValueError(
                f"Tool '{tool_name}' failed to instantiate superseded tool: {e}"
            ) from e

        if _has_args_schema(superseded_tool_instance.args_schema):
            try:
                validate_against_schema(
                    superseded_tool_instance.args_schema,
                    configured_params,
                )
                # Config matches this level's schema — use it.
                return superseded_tool_instance
            except ValueError as e:
                current_error = e

        candidate = superseded_tool_instance

    # No tool in the supersession chain matched — raise the last error
    raise ValueError(f"Tool '{tool_name}' {current_error!s}") from current_error


def validate_against_schema(
    schema_class: type[BaseModel], configured_params: set[str]
) -> None:
    """Validate configured parameters against a specific schema.

    Args:
        schema_class: The Pydantic model class defining the schema
        configured_params: Set of parameter names from the config

    Returns:
        None if valid

    Raises:
        ValueError: if validation fails
    """
    schema = schema_class.model_json_schema()
    expected_params = set(schema.get("properties", {}).keys())
    required_params = set(schema.get("required", []))

    # Check for missing required parameters
    missing_required = required_params - configured_params
    if missing_required:
        raise ValueError(f"Missing required parameters: {sorted(missing_required)}")

    # Check for unknown parameters
    unknown_params = configured_params - expected_params
    if unknown_params:
        raise ValueError(
            f"Unknown parameters: {sorted(unknown_params)}. Valid parameters are: {sorted(expected_params)}"
        )


def _has_args_schema(
    args_schema: ArgsSchema | None,
) -> TypeGuard[type[BaseModel]]:
    """Type guard to check if args_schema is a valid Pydantic BaseModel class.

    Returns:
        True if schema is a BaseModel class (not dict or None)
    """
    return args_schema is not None and not isinstance(args_schema, dict)
