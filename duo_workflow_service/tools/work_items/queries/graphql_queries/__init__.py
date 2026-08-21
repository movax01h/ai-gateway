from duo_workflow_service.graphql_queries import TOOLS_QUERIES_DIR, load_graphql_query

GET_GROUP_WORK_ITEM_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_group_work_item.graphql")
)
GET_PROJECT_WORK_ITEM_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_project_work_item.graphql")
)
LIST_GROUP_WORK_ITEMS_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "list_group_work_items.graphql")
)
LIST_PROJECT_WORK_ITEMS_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "list_project_work_items.graphql")
)
GET_GROUP_WORK_ITEM_NOTES_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_group_work_item_notes.graphql")
)
GET_PROJECT_WORK_ITEM_NOTES_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_project_work_item_notes.graphql")
)
CREATE_WORK_ITEM_MUTATION = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "create_work_item.graphql")
)
GET_WORK_ITEM_TYPE_BY_NAME_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_work_item_type_by_name.graphql")
)
CREATE_NOTE_MUTATION = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "create_note.graphql")
)
UPDATE_WORK_ITEM_MUTATION = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "update_work_item.graphql")
)
GET_NOTE_QUERY = load_graphql_query(str(TOOLS_QUERIES_DIR / "get_note.graphql"))
GET_WORK_ITEM_STATUSES_QUERY = load_graphql_query(
    str(TOOLS_QUERIES_DIR / "get_work_item_statuses.graphql")
)

__all__ = [
    "CREATE_NOTE_MUTATION",
    "CREATE_WORK_ITEM_MUTATION",
    "GET_GROUP_WORK_ITEM_NOTES_QUERY",
    "GET_GROUP_WORK_ITEM_QUERY",
    "GET_NOTE_QUERY",
    "GET_PROJECT_WORK_ITEM_NOTES_QUERY",
    "GET_PROJECT_WORK_ITEM_QUERY",
    "GET_WORK_ITEM_STATUSES_QUERY",
    "GET_WORK_ITEM_TYPE_BY_NAME_QUERY",
    "LIST_GROUP_WORK_ITEMS_QUERY",
    "LIST_PROJECT_WORK_ITEMS_QUERY",
    "UPDATE_WORK_ITEM_MUTATION",
]
