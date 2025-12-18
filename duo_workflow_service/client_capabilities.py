from contextvars import ContextVar


def is_client_capable(capability: str) -> bool:
    return capability in client_capabilities.get()


# client_capabilities is used to make backwards compatible changes to our
# communication protocol. This is needed usually when we're adding new
# protobuf fields and changing behaviour of the gRPC communication in
# non-backwards compatible ways.
#
# It is passed through the original client (e.g.
# `gitlab-lsp`) through workhorse and all the way to Duo Workflow Service.
# Each client in the chain intersects their capabilities such that when
# Duo Workflow Service receives this we know that all clients in the chain
# definitely support this capability.
client_capabilities: ContextVar[set[str]] = ContextVar(
    "client_capabilities", default=set()
)
