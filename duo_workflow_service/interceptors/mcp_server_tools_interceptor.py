import grpc

from lib.mcp_server_tools.context import (
    X_GITLAB_ENABLED_MCP_SERVER_TOOLS,
    set_enabled_mcp_server_tools,
)


class McpServerToolsInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles MCP server tools propagation."""

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject MCP server tools context."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract enabled MCP server tools from metadata
        enabled_tools = metadata.get(X_GITLAB_ENABLED_MCP_SERVER_TOOLS, "").split(",")
        # Filter out empty strings from split
        enabled_tools = set(tool.strip() for tool in enabled_tools if tool.strip())

        # Set MCP server tools in context
        set_enabled_mcp_server_tools(enabled_tools)

        return await continuation(handler_call_details)
