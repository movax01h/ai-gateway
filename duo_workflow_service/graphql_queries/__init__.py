from importlib import resources
from pathlib import Path

TOOLS_QUERIES_DIR = (
    resources.files("duo_workflow_service") / "graphql_queries" / "tools_queries"
)


def load_graphql_query(file_path: str) -> str:
    """Load a GraphQL query from a .graphql file.

    Args:
        file_path: Path to the `.graphql` file, usually built from `TOOLS_QUERIES_DIR`.

    Returns:
        The raw contents of the file as a GraphQL query string.

    Raises:
        FileNotFoundError: If no file exists at `file_path`.
    """
    query_path = Path(file_path)
    try:
        with open(query_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"GraphQL query file not found: {query_path}") from e


__all__ = [
    "TOOLS_QUERIES_DIR",
    "load_graphql_query",
]
