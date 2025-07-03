GET_GROUP_WORK_ITEM_QUERY = """
query GetGroupWorkItem($fullPath: ID!, $iid: String!) {
    namespace(fullPath: $fullPath) {
        workItems(iid: $iid) {
            nodes {
                author {
                    username
                    name
                }
                closedAt
                confidential
                createdAt
                description
                id
                iid
                imported
                namespace {
                    id
                    fullPath
                }
                project {
                    id
                    fullPath
                }
                state
                title
                updatedAt
                workItemType {
                    id
                    name
                }
                archived
            }
        }
    }
}
"""

GET_PROJECT_WORK_ITEM_QUERY = """
query GetProjectWorkItem($fullPath: ID!, $iid: String!) {
    project(fullPath: $fullPath) {
        workItems(iid: $iid) {
            nodes {
                author {
                    username
                    name
                }
                closedAt
                confidential
                createdAt
                description
                id
                iid
                imported
                namespace {
                    id
                    fullPath
                }
                project {
                    id
                    fullPath
                }
                state
                title
                updatedAt
                workItemType {
                    id
                    name
                }
                archived
            }
        }
    }
}
"""
