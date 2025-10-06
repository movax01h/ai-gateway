# Fix Pipeline Flow

## What It Does

This flow takes a URL for GitLab CI/CD pipeline that has failing jobs as input and automatically attempts to resolve it by creating code changes and submitting a merge request.

## When to Use

Use this Flow when you have a failing pipeline that needs to be fixed.

## How to Use

note: This workflow is intended for remote execution as it pushes the code and creates a merge request automatically.

1. Navigate to a merge request with a failing pipeline.
1. Click on the Pipelines tab of the Merge Request.
1. Beside the failing pipeline you should see a Button for "Fix pipeline with Duo". Click that Button.
1. Go to `Build > Pipelines` from the side menu bar. Checkout the most recent pipeline with a `workload` job for execution
   logs.
1. Once the pipeline has successfully executed, go to Merge Requests from the side menu bar. You should see a draft merge
   request titled "Fix Failing Pipeline"
1. Review the merge request.
