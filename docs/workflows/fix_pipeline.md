# Fix Pipeline Flow

## What It Does

This flow takes a URL for a failing GitLab CI/CD job URL as input and automatically resolves it by creating code changes and submitting a merge request.

## When to Use

- Fixing a failing pipeline job
- Fixing a pipeline with multiple failing jobs (coming soon)

## How to Use

> **Note:** This flow is currently in early development, and can only be triggered from an API call. It is run in remote execution, and it pushes the code and creates a merge request automatically.

1. Navigate to a failing job in a merge request, and note the job URL, the branch for the MR, and the project ID
1. Execute the following `curl` command in your terminal:

   ```shell
      curl --location "http://gdk.test:3000/api/v4/ai/duo_workflows/workflows" \
          --header 'Content-Type: application/json' \
          --header "Authorization: Bearer $GDK_API_TOKEN" \
          --data '{
              "goal": "FAILING_JOB_URL",
              "workflow_definition": "fix_pipeline/experimental",
              "agent_privileges": [1,2,3,5],
              "pre_approved_agent_privileges": [1,2,3,5],
              "start_workflow": true,
              "source_branch": "MERGE_REQUEST_BRANCH"
          }'
   ```

1. Go to `Build > Pipelines` from the side menu bar. Checkout the most recent pipeline with a `workload` job for execution
   logs.
1. Once the pipeline has successfully executed, go to Merge Requests from the side menu bar. You should see a draft merge
   request titled "Fix Failing Pipeline"
1. Review the merge request.
