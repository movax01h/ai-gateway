"""Tool trust level classification for security."""

from enum import Enum


class ToolTrustLevel(str, Enum):
    """Trust levels for tool outputs.

    This classification determines how tool outputs should be handled
    for security purposes in future implementations.

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
