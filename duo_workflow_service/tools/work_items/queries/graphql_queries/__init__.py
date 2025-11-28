from importlib import resources
from pathlib import Path


def load_graphql_query(file_path: str) -> str:
    """Load a GraphQL query from a .graphql file."""
    query_path = Path(file_path)
    try:
        with open(query_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"GraphQL query file not found: {query_path}") from e


_GRAPHQL_DIR = (
    resources.files("duo_workflow_service") / "graphql_queries" / "tools_queries"
)

GET_GROUP_WORK_ITEM_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "get_group_work_item.graphql")
)
GET_PROJECT_WORK_ITEM_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "get_project_work_item.graphql")
)
LIST_GROUP_WORK_ITEMS_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "list_group_work_items.graphql")
)
LIST_PROJECT_WORK_ITEMS_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "list_project_work_items.graphql")
)
GET_GROUP_WORK_ITEM_NOTES_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "get_group_work_item_notes.graphql")
)
GET_PROJECT_WORK_ITEM_NOTES_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "get_project_work_item_notes.graphql")
)
CREATE_WORK_ITEM_MUTATION = load_graphql_query(
    str(_GRAPHQL_DIR / "create_work_item.graphql")
)
GET_WORK_ITEM_TYPE_BY_NAME_QUERY = load_graphql_query(
    str(_GRAPHQL_DIR / "get_work_item_type_by_name.graphql")
)
CREATE_NOTE_MUTATION = load_graphql_query(str(_GRAPHQL_DIR / "create_note.graphql"))
UPDATE_WORK_ITEM_MUTATION = load_graphql_query(
    str(_GRAPHQL_DIR / "update_work_item.graphql")
)

__all__ = [
    "GET_GROUP_WORK_ITEM_QUERY",
    "GET_PROJECT_WORK_ITEM_QUERY",
    "LIST_GROUP_WORK_ITEMS_QUERY",
    "LIST_PROJECT_WORK_ITEMS_QUERY",
    "GET_GROUP_WORK_ITEM_NOTES_QUERY",
    "GET_PROJECT_WORK_ITEM_NOTES_QUERY",
    "CREATE_WORK_ITEM_MUTATION",
    "GET_WORK_ITEM_TYPE_BY_NAME_QUERY",
    "CREATE_NOTE_MUTATION",
    "UPDATE_WORK_ITEM_MUTATION",
]
