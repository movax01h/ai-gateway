from importlib import resources

_GRAPHQL_DIR = (
    resources.files("duo_workflow_service") / "graphql_queries" / "tools_queries"
)

GET_SECURITY_FINDING_DETAILS_QUERY = (
    _GRAPHQL_DIR / "get_security_report_finding.graphql"
).read_text(encoding="utf-8")

LIST_SECURITY_FINDINGS_QUERY = (
    _GRAPHQL_DIR / "list_pipeline_security_findings.graphql"
).read_text(encoding="utf-8")

__all__ = [
    "GET_SECURITY_FINDING_DETAILS_QUERY",
    "LIST_SECURITY_FINDINGS_QUERY",
]
