## Description

The tests in this folder are designed to help identify performance bottlenecks by generating load for the
Duo Workflow Service (DWS) via a GitLab instance.

The tests should not be added to CI pipelines at this time.

## Running with k6 directly

The `api_v4_duo_workflow_chat_graphql_api.js` test can be run directly with k6 (without requiring GitLab Performance Tool as shown below).

### Prerequisites

1. [Install k6](https://grafana.com/docs/k6/latest/set-up/install-k6/)
1. A GitLab environment with Duo Agent Platform enabled
   - **Preferred**: Use the [LLM caching proxy](../../docs/performance_testing/profiling_with_llm_caching_proxy.md)
     with `SCENARIO_TYPE=llm_proxy` (the default). The proxy caches actual LLM responses, reducing API costs while
     producing realistic results. Configure DWS with:
     ```shell
     DUO_WORKFLOW_USE_CACHING_PROXY=true
     ```
     See [Profiling DWS with Pyroscope and LLM caching proxy](../../docs/performance_testing/profiling_with_llm_caching_proxy.md)
     for setup instructions.
   - **Alternative**: For fully mocked responses (no LLM calls), configure DWS with:
     ```shell
     AIGW_MOCK_MODEL_RESPONSES=true
     AIGW_USE_AGENTIC_MOCK=true
     ```
     Note that agentic mocking bypasses LLM calls entirely, so results may not reflect production behaviour.
1. **For foundational agents**: Fetch agent configurations into the DWS before running tests (see below)

    When load testing foundational agents like `duo_planner` or `security_analyst_agent`, you must fetch their configurations first:

    ```shell
    poetry run fetch-foundational-agents \
      "https://gitlab.test" \
      "$GITLAB_API_TOKEN" \
      "duo_planner:348,security_analyst_agent:356" \
      --flow-registry-version v1
    ```

    **Parameters:**

    - GitLab instance URL
    - GitLab API token with read access to the agent repository
    - Comma-separated list of `agent_name:version_id` pairs
    - Flow registry version (`v1` or `experimental`)

    The fetched configurations will be used by DWS during the test. Without this step, tests using foundational agents may fail or use incorrect versions.
1. A project with appropriate feature flags enabled (such as one created by `rake gitlab:duo:setup` in GDK)
1. A Personal Access Token with API scope

### Running the test

```shell
ENVIRONMENT_URL=https://gitlab.test \
ACCESS_TOKEN=<gitlab personal access token with api scope> \
AI_DUO_WORKFLOW_PROJECT_ID=1000000 \
AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID=1000000 \
k6 run performance_tests/stress_tests/api_v4_duo_workflow_chat_graphql_api.js
```

### Configuration options

- `ACCESS_TOKEN`: Your GitLab Personal Access Token (required)
- `AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID`: The namespace ID with Duo Agent Platform enabled (required)
- `AI_DUO_WORKFLOW_PROJECT_ID`: The project ID in the above namespace (required)
- `SCENARIO_TYPE`: `llm_proxy` (default, recommended), `real_llm`, or `mocked_llm`
- `MOCKED_GOAL_FILE`: A file containing the goal(s) to pass to Agentic Chat (default: [`goals/summarize_issue_check_implementation.txt`](goals/summarize_issue_check_implementation.txt)).
  Can be a plain text file with a single goal, or a YAML file with multiple goals (distributed round-robin across test iterations).
  For example: [`goals/security_analyst_agent/example_questions.yaml`](goals/security_analyst_agent/example_questions.yaml)

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
- `SCENARIO_TYPE=llm_proxy` is the default and recommended option. Use `SCENARIO_TYPE=mocked_llm` to bypass LLM calls entirely with agentic mocking, or `SCENARIO_TYPE=real_llm` for direct LLM calls when the LLM caching proxy is not used.
