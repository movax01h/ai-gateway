from duo_workflow_service.graphql_queries import TOOLS_QUERIES_DIR, load_graphql_query

SET_REVIEWERS_MUTATION = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "set_reviewers.graphql")
)

__all__ = [
    "SET_REVIEWERS_MUTATION",
]
