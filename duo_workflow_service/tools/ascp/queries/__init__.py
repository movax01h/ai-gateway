from importlib import resources

_GRAPHQL_DIR = (
    resources.files("duo_workflow_service") / "graphql_queries" / "tools_queries"
)

CREATE_ASCP_SCAN_MUTATION = (_GRAPHQL_DIR / "create_ascp_scan.graphql").read_text(
    encoding="utf-8",
)

LIST_ASCP_SCANS_QUERY = (_GRAPHQL_DIR / "list_ascp_scans.graphql").read_text(
    encoding="utf-8",
)

CREATE_ASCP_COMPONENT_MUTATION = (
    _GRAPHQL_DIR / "create_ascp_component.graphql"
).read_text(encoding="utf-8")

CREATE_ASCP_SECURITY_CONTEXT_MUTATION = (
    _GRAPHQL_DIR / "create_ascp_security_context.graphql"
).read_text(encoding="utf-8")

LIST_ASCP_COMPONENTS_QUERY = (_GRAPHQL_DIR / "list_ascp_components.graphql").read_text(
    encoding="utf-8"
)

__all__ = [
    "CREATE_ASCP_COMPONENT_MUTATION",
    "CREATE_ASCP_SCAN_MUTATION",
    "CREATE_ASCP_SECURITY_CONTEXT_MUTATION",
    "LIST_ASCP_COMPONENTS_QUERY",
    "LIST_ASCP_SCANS_QUERY",
]
