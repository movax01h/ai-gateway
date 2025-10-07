# Fix Pipeline Flow

## What It Does

This flow takes a URL for GitLab CI/CD pipeline that has failing jobs as input and automatically attempts to resolve it by creating code changes and submitting a merge request.

The flow supports two scenarios:

- **Merge Request Pipelines**: When a merge request URL is provided, the flow analyzes the merge request diffs and creates a fix targeting the source branch
- **Master/Main Branch Pipelines**: When no merge request URL is provided, the flow focuses on general pipeline fixes and targets the default branch

## When to Use

Use this Flow when you have a failing pipeline that needs to be fixed, whether it's:

- A pipeline failing on a merge request
- A pipeline failing on the master/main branch

## How to Use

### For Merge Request Pipelines

1. Navigate to a merge request with a failing pipeline.
1. Click on the Pipelines tab of the Merge Request.
1. Beside the failing pipeline click the "Fix pipeline with Duo" button.
1. Click on the Agent Session link in the banner to view progress.
1. Once the flow has successfully executed, you should see a draft merge request with the proposed fixes, targeting your MR branch.
1. Review the merge request.

### For Master/Main Branch Pipelines

1. Navigate to the failing pipeline.
1. Click the "Fix pipeline with Duo" button.
1. Click on the Agent Session link in the banner to view progress.
1. Once the flow has successfully executed, you should see a draft merge request with the proposed fixes, targeting the default branch.
1. Review the merge request.
