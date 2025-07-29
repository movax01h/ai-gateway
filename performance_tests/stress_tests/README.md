## Description

The tests in this folder are designed to be used with [GitLab Performance Tool](https://gitlab.com/gitlab-org/quality/performance), to generate Duo Agent load on a GitLab instance and identify performance bottlenecks.

They should not be added to CI pipelines at this time.

## How to run

You will need a GitLab environment, with runners deployed and configured.

1. [Set up GitLab Performance Tool on a machine with access to the target GitLab environment](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/k6.md#docker-recommended)
1. [Identify the appropriate ENVIRONMENT_FILE, or create a new one if it does not exist.](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/environment_prep.md#preparing-the-environment-file)
1. [Identify the appropriate OPTIONS_FILE, or create a new one if it does not exist.](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/k6.md#options-rps)
1. Data seeding via the GPT Data Seeder is NOT required for these tests, but you will need a project in the environment with the appropriate feature flags enabled to use Duo Agent (such as the one created by `rake gitlab:duo:setup` in GDK). This project's ID is your PROJECT_ID.
1. Generate a PAT for a user or service account. Ensure that user has all of the required feature flags + licenses enabled to run Duo Agent in CI.
1. Ensure that the user and project have all of the appropriate feature flags and licenses enabled to run Duo Agent in CI.
1. Disable the GPT pre-flight checks that ensure data seeding was done before tests run
    1. `export GPT_SKIP_VISIBILITY_CHECK=true`
    1. `export GPT_LARGE_PROJECT_CHECK_SKIP=true`
1. Run `AI_DUO_WORKFLOW_PROJECT_ID=<PROJECT_ID> ./bin/run-k6 --environment <ENVIRONMENT_FILE> --options <OPTIONS_FILE> --tests api_v4_duo_workflow_chat.js`
