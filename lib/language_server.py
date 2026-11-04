from typing import Optional

import structlog
from packaging.version import InvalidVersion, Version

__all__ = ["LanguageServerVersion"]

log = structlog.stdlib.get_logger("codesuggestions")

ADVANCED_CONTEXT_LANGUAGE_SERVER_VERSION = Version("7.17.1")
MINIMUM_WORKFLOW_CLIENT_VERSION = Version("7.43.0")
FLOW_SCHEMA_VERSION_LANGUAGE_SERVER_VERSION = Version("8.51.0")
# First release that reserves the `duo` MCP server name for the first-party server (gitlab-lsp!4448).
FIRST_PARTY_MCP_TOOLS_LANGUAGE_SERVER_VERSION = Version("9.33.0")


class LanguageServerVersion:
    @classmethod
    def from_string(cls, version: Optional[str]) -> "LanguageServerVersion":
        version = version or "0.0.0"

        try:
            return LanguageServerVersion(Version(version))
        except InvalidVersion:
            log.warning(
                "Invalid X-Gitlab-Language-Server-Version header passed.",
                version=version,
            )
            return LanguageServerVersion(Version("0.0.0"))

    def __init__(self, version: Version) -> None:
        self.version = version

    def supports_advanced_context(self) -> bool:
        return self.version >= ADVANCED_CONTEXT_LANGUAGE_SERVER_VERSION

    def supports_node_executor_tools(self) -> bool:
        return self.version >= MINIMUM_WORKFLOW_CLIENT_VERSION

    def supports_first_party_mcp_tools(self) -> bool:
        return self.version >= FIRST_PARTY_MCP_TOOLS_LANGUAGE_SERVER_VERSION

    def ignore_broken_flow_schema_version(self) -> bool:
        return self.version < FLOW_SCHEMA_VERSION_LANGUAGE_SERVER_VERSION
