from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duo_workflow_service.components.tools_registry import ToolsRegistry

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


def strip_ask_listed_pre_approvals(
    pre_approved_tools: list[str], tools_registry: ToolsRegistry
) -> list[str]:
    """Remove tools an admin `ask` rule forces to prompt from a flow config's static pre-approval list.

    The flow config cannot pre-approve past an admin `ask` rule; this list short-circuits the approval node before
    any other check.
    """
    forced = tools_registry.ask_listed_tool_names(pre_approved_tools)
    return [name for name in pre_approved_tools if name not in forced]
