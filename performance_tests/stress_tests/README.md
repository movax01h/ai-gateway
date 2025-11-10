## Description

The tests in this folder are designed to help identify performance bottlenecks by generating load for the
Duo Workflow Service via a GitLab instance.

The tests should not be added to CI pipelines at this time.

## Running with k6 directly

The `api_v4_duo_workflow_chat_graphql_api.js` test can be run directly with k6 (without requiring GitLab Performance Tool as shown below).

### Prerequisites

1. [Install k6](https://grafana.com/docs/k6/latest/set-up/install-k6/)
1. A GitLab environment with Duo Agent Platform enabled (e.g., [GDK](https://gitlab.com/gitlab-org/gitlab-development-kit/-/blob/main/doc/howto/duo_agent_platform.md))
    - For mocked responses, ensure the Duo Workflow Service is configured with:
        ```
        AIGW_MOCK_MODEL_RESPONSES=true
        AIGW_USE_AGENTIC_MOCK=true
        ```

1. A project with appropriate feature flags enabled (such as one created by `rake gitlab:duo:setup` in GDK)
1. A Personal Access Token with API scope

### Running the test

```shell
ENVIRONMENT_URL=http://gdk.test:3000 \
ACCESS_TOKEN=<gitlab personal access token with api scope> \
AI_DUO_WORKFLOW_PROJECT_ID=1000000 \
AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID=1000000 \
k6 run performance_tests/stress_tests/api_v4_duo_workflow_chat_graphql_api.js
```

### Configuration options

- `ACCESS_TOKEN`: Your GitLab Personal Access Token (required)
- `AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID`: The namespace ID with Duo Agent Platform enabled (required)
- `AI_DUO_WORKFLOW_PROJECT_ID`: The project ID in the above namespace (required)
- `SCENARIO_TYPE`: Either `mocked_llm` (default) or `real_llm`

For mocked responses, ensure the Duo Workflow Service is configured with:
```
AIGW_MOCK_MODEL_RESPONSES=true
AIGW_USE_AGENTIC_MOCK=true
```

You can adjust the load scenarios by editing the `options` object in the test file.

## Running with GitLab Performance Tool (GPT)

The `api_v4_duo_workflow_software_development_rest_api.js` and `api_v4_duo_workflow_chat_rest_api.js` tests require [GitLab Performance Tool](https://gitlab.com/gitlab-org/quality/performance).

### Prerequisites

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
1. Run `AI_DUO_WORKFLOW_PROJECT_ID=<PROJECT_ID> ./bin/run-k6 --environment <ENVIRONMENT_FILE> --options <OPTIONS_FILE> --tests api_v4_duo_workflow_chat_rest_api.js`

### Docker example

Below is an example using Docker to run a test:

```shell
docker run -it \
  --rm \
  -e ACCESS_TOKEN=$GITLAB_TOKEN \
  -e AI_DUO_WORKFLOW_PROJECT_ID=1000000 \
  -e AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID=1000000 \
  -e SCENARIO_TYPE=mocked_llm \
  -e GPT_SKIP_RETRY=true \
  -e GPT_SKIP_VISIBILITY_CHECK=true \
  -e GPT_LARGE_PROJECT_CHECK_SKIP=true \
  -v ./performance_tests/config:/config \
  -v ./performance_tests/stress_tests:/performance/k6/tests/experimental \
  -v ./results:/results \
  gitlab/gitlab-performance-tool \
  --environment /config/environments/load-test-agentic-chat.json \
  --options /config/options/5s_2rps.json \
  --tests experimental/api_v4_duo_workflow_chat_rest_api.js
```

Note the following:

- The project and namespace ids correspond to the ids of the group and project created by the Duo rake setup task.
- The GitLab instance in the environment configuration, `load-test-agentic-chat.json`, was deployed for the purpose of Duo Workflow Service load testing.
  You can replace the URL in the configuration with your GDK URL if you'd like to run the tests locally.
- `GPT_SKIP_RETRY=true` is helpful when running tests locally for development or troubleshooting purposes.
- The `tests` path is constructed so that reference to `../../lib/gpt_k6_modules.js` in the tests resolves correctly to a file in the gitlab-performance-tool codebase.
- `SCENARIO_TYPE=mocked_llm` is actually the default. Use `SCENARIO_TYPE=real_llm` for actual LLM responses.
