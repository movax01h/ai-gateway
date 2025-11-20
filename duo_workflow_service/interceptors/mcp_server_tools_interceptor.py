from contextvars import ContextVar

import grpc

current_mcp_server_tools_context: ContextVar[set[str]] = ContextVar(
    "current_mcp_server_tools_context", default=set()
)


class McpServerToolsInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles MCP server tools propagation."""

    X_GITLAB_ENABLED_MCP_SERVER_TOOLS = "x-gitlab-enabled-mcp-server-tools"

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject MCP server tools context."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract enabled MCP server tools from metadata
        enabled_tools = metadata.get(self.X_GITLAB_ENABLED_MCP_SERVER_TOOLS, "").split(
            ","
        )
        # Filter out empty strings from split
        enabled_tools = set(tool.strip() for tool in enabled_tools if tool.strip())

        # Set MCP server tools in context
        current_mcp_server_tools_context.set(enabled_tools)

        return await continuation(handler_call_details)
