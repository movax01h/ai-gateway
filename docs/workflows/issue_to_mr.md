# Issue to MR Flow

## What It Does

It takes a GitLab issue URL as input and automatically resolves it by creating code changes and submitting a merge request.

## When to Use

- Creating merge requests for well-scoped GitLab issues.

## How to Use

note: This workflow is intended for remote execution as it pushes the code and creates a merge request automatically.

1. Verify that all setup [prerequisites](https://docs.gitlab.com/user/duo_agent_platform/flows/issue_to_mr/#prerequisites)
   have been satisfied.
1. Open a GitLab issue in your remote project repository.
1. You should see a button `Generate MR with Duo` below issue description. Click the button, you should see a
   flash message Workflow started successfully.
1. Go to Build > Pipelines from the side menu bar. Checkout the most recent pipeline with a `workload` job for execution
   logs.
1. Once the pipeline has successfully executed, go to Merge Requests from the side menu bar. You should see a draft merge
   request titled "Draft: Resolve #{issue_iid}"
1. Review the merge request.

## Best Practices

1. Keep the issues well scoped. **Break down complex tasks** into smaller, focused and action-oriented requests
1. Specify exact file paths
1. Write specific acceptance criteria
1. Include code examples of existing patterns(when needed) to maintain consistency.

## Example

```markdown
## Description
The users endpoint currently returns all users at once, which will cause performance issues as the user base grows.
Implement cursor-based pagination for the `/api/users` endpoint to handle large datasets efficiently

## Implementation Plan
Add pagination to GET /users API endpoint.
Include pagination metadata in /users API response (per_page, page)
Add query parameters for per page size limit (default 5, max 20)

#### Files to Modify
- `src/api/users.py` - Add pagination parameters and logic
- `src/models/user.py` - Add pagination query method
- `tests/api/test_users_api.py` - Add pagination tests

## Acceptance Criteria
- Accepts page and per_page query parameters (default: page=5, per_page=10)
- Limits per_page to maximum of 20 users
- Maintains existing response format for user objects in data array
```
