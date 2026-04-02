_MAX_WORKFLOW_DEFINITION_LENGTH = 256
VALID_SCHEMA_VERSIONS = frozenset({"v1", "experimental"})


def parse_deprecated_workflow_definition(value: str) -> tuple[str, str]:
    """Parse legacy '<flow_name>/<api_version>' workflow definition — strictly 2 segments.

    Returns (api_version, flow_name). Always resolves to the default flow version (1.0.0). Raises ValueError on any
    format violation or length excess. Path traversal safety is enforced downstream by _safe_resolve in
    from_yaml_config.
    """
    if len(value) > _MAX_WORKFLOW_DEFINITION_LENGTH:
        raise ValueError(
            f"workflow_definition exceeds maximum length of {_MAX_WORKFLOW_DEFINITION_LENGTH}"
        )

    raw_parts = value.split("/")

    if len(raw_parts) != 2:
        raise ValueError(
            f"Invalid workflow_definition format: '{value}'. "
            "Expected '<flow_name>/<api_version>' (e.g. 'developer/v1')."
        )

    flow_name, api_version = raw_parts
    if not flow_name:
        raise ValueError(
            f"Invalid workflow_definition format: '{value}'. "
            "Expected '<flow_name>/<api_version>' (e.g. 'developer/v1')."
        )
    if api_version not in VALID_SCHEMA_VERSIONS:
        raise ValueError(
            f"Invalid API version '{api_version}'. "
            f"Must be one of: {', '.join(sorted(VALID_SCHEMA_VERSIONS))}."
        )
    return api_version, flow_name
