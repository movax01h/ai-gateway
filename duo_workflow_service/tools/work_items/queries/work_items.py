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

LIST_GROUP_WORK_ITEMS_QUERY = """
query ListGroupWorkItems($fullPath: ID!, $state: IssuableState, $search: String, $authorUsername: String, $createdAfter: Time, $createdBefore: Time, $updatedAfter: Time, $updatedBefore: Time, $dueAfter: Time, $dueBefore: Time, $sort: WorkItemSort, $first: Int, $after: String, $types: [IssueType!]) {
    namespace(fullPath: $fullPath) {
        workItems(
            state: $state
            search: $search
            authorUsername: $authorUsername
            createdAfter: $createdAfter
            createdBefore: $createdBefore
            updatedAfter: $updatedAfter
            updatedBefore: $updatedBefore
            dueAfter: $dueAfter
            dueBefore: $dueBefore
            sort: $sort
            first: $first
            after: $after
            types: $types
        ) {
            pageInfo {
                hasNextPage
                endCursor
            }
            nodes {
                id
                iid
                title
                state
                createdAt
                updatedAt
                author {
                    username
                    name
                }
            }
        }
    }
}
"""

LIST_PROJECT_WORK_ITEMS_QUERY = """
query ListProjectWorkItems($fullPath: ID!, $state: IssuableState, $search: String, $authorUsername: String, $createdAfter: Time, $createdBefore: Time, $updatedAfter: Time, $updatedBefore: Time, $dueAfter: Time, $dueBefore: Time, $sort: WorkItemSort, $first: Int, $after: String, $types: [IssueType!]) {
    project(fullPath: $fullPath) {
        workItems(
            state: $state
            search: $search
            authorUsername: $authorUsername
            createdAfter: $createdAfter
            createdBefore: $createdBefore
            updatedAfter: $updatedAfter
            updatedBefore: $updatedBefore
            dueAfter: $dueAfter
            dueBefore: $dueBefore
            sort: $sort
            first: $first
            after: $after
            types: $types
        ) {
            pageInfo {
                hasNextPage
                endCursor
            }
            nodes {
                id
                iid
                title
                state
                createdAt
                updatedAt
                author {
                    username
                    name
                }
            }
        }
    }
}
"""

GET_GROUP_WORK_ITEM_NOTES_QUERY = """
query GetGroupWorkItemNotes($fullPath: ID!, $workItemIid: String!) {
    namespace(fullPath: $fullPath) {
        workItems(iid: $workItemIid) {
            nodes {
                widgets {
                    ... on WorkItemWidgetNotes {
                        notes {
                            nodes {
                                id
                                body
                                createdAt
                                author {
                                    username
                                    name
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

GET_PROJECT_WORK_ITEM_NOTES_QUERY = """
query GetProjectWorkItemNotes($fullPath: ID!, $workItemIid: String!) {
    project(fullPath: $fullPath) {
        workItems(iid: $workItemIid) {
            nodes {
                widgets {
                    ... on WorkItemWidgetNotes {
                        notes {
                            nodes {
                                id
                                body
                                createdAt
                                author {
                                    username
                                    name
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

CREATE_WORK_ITEM_MUTATION = """
mutation createWorkItem($input: WorkItemCreateInput!) {
    workItemCreate(input: $input) {
            workItem {
                id
                iid
                title
                description
                confidential
                createdAt
                updatedAt
                widgets {
                    ... on WorkItemWidgetAssignees {
                        assignees {
                        nodes {
                            id
                        username
                        }
                    }
                }
                    ... on WorkItemWidgetLabels {
                        labels {
                            nodes {
                                id
                                title
                                color
                        }
                    }
                }
                    ... on WorkItemWidgetHealthStatus {
                        healthStatus
                }
                    ... on WorkItemWidgetStartAndDueDate {
                        startDate
                        dueDate
                        isFixed
                    }
                }
            }
        errors
    }
}
"""

GET_WORK_ITEM_TYPE_BY_NAME_QUERY = """
query GetWorkItemType($fullPath: ID!) {
    namespace(fullPath: $fullPath) {
        workItemTypes {
            nodes {
                id
                name
            }
        }
    }
}
"""

CREATE_NOTE_MUTATION = """
mutation CreateNote($input: CreateNoteInput!) {
    createNote(input: $input) {
        note {
            id
            body
            createdAt
            author {
                username
                name
            }
        }
        errors
    }
}
"""

UPDATE_WORK_ITEM_MUTATION = """
mutation updateWorkItem($input: WorkItemUpdateInput!) {
    workItemUpdate(input: $input) {
        workItem {
            id
            iid
            title
            description
            confidential
            createdAt
            updatedAt
            widgets {
                ... on WorkItemWidgetAssignees {
                    assignees {
                        nodes {
                            id
                            username
                        }
                    }
                }
                ... on WorkItemWidgetLabels {
                    labels {
                        nodes {
                            id
                            title
                            color
                        }
                    }
                }
                ... on WorkItemWidgetHealthStatus {
                    healthStatus
                }
                ... on WorkItemWidgetStartAndDueDate {
                    startDate
                    dueDate
                    isFixed
                }
            }
        }
        errors
    }
}
"""
