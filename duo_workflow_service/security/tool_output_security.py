"""Tool output security: trust level classification and delimiter tag encoding.

This module provides security mechanisms to protect against prompt injection
attacks in tool outputs. It classifies tools by trust level and defines
delimiter tags that must be encoded in untrusted content.

Key components:
- ToolTrustLevel: Enum classifying tools by content trustworthiness
- Security delimiter tags: Constants for tags that must be encoded in untrusted content
"""

from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Security Delimiter Tags - Single Source of Truth
# ---------------------------------------------------------------------------
# These tag names are used in tool output wrappers (e.g. truncation notices)
# and MUST be encoded in untrusted content to prevent delimiter escape attacks.
# If you add a new wrapper tag, add it here so it gets encoded automatically.

TRUNCATED_TOOL_OUTPUT_TAG = "truncated_tool_output"
TRUNCATION_NOTICE_TAG = "truncation_notice"

# All delimiter tags that must be encoded in untrusted content
SECURITY_DELIMITER_TAGS = [
    TRUNCATED_TOOL_OUTPUT_TAG,
    TRUNCATION_NOTICE_TAG,
]


class ToolTrustLevel(str, Enum):
    """Trust levels for tool outputs.

    This classification determines how tool outputs should be handled
    for security purposes. The default is UNTRUSTED_USER_CONTENT for
    fail-secure behavior.

    Trust levels:
    - TRUSTED_INTERNAL: Tools that write/modify data or return metadata only
        (file paths, IDs). These operations are controlled by developers and
        don't return user-controllable content.

    - UNTRUSTED_USER_CONTENT: Tools that return user-generated content from
        GitLab (issues, MRs, comments, code). This is the default for safety.

    - UNTRUSTED_EXTERNAL: Tools that return content from external sources
        (documentation, web APIs).

    - UNTRUSTED_MCP: MCP server tools (handled separately).
    """

    TRUSTED_INTERNAL = "trusted_internal"
    UNTRUSTED_USER_CONTENT = "untrusted_user_content"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    UNTRUSTED_MCP = "untrusted_mcp"


def get_tool_trust_level(tool: Any) -> ToolTrustLevel:
    """Get the trust level for a tool.

    Args:
        tool: The tool to get the trust level for.

    Returns:
        The tool's trust level, defaulting to UNTRUSTED_USER_CONTENT
        for fail-secure behavior if not explicitly set.
    """
    return getattr(tool, "trust_level", ToolTrustLevel.UNTRUSTED_USER_CONTENT)
