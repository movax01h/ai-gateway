from contextvars import ContextVar

X_GITLAB_ENABLED_MCP_SERVER_TOOLS = "x-gitlab-enabled-mcp-server-tools"

current_mcp_server_tools_context: ContextVar[set[str]] = ContextVar(
    "current_mcp_server_tools_context", default=set()
)


def set_enabled_mcp_server_tools(tools: set[str]):
    """Set the enabled MCP server tools for the current request."""
    current_mcp_server_tools_context.set(tools)


def get_enabled_mcp_server_tools() -> set[str]:
    """Get the enabled MCP server tools for the current request."""
    return current_mcp_server_tools_context.get()
