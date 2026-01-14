# Performance Tests in CI

## `tests:performance` job

The `test:performance` job runs in MR pipelines. It runs all the tests under the [`performance_test/k6_test`](../../performance_tests/k6_test/) directory against a Docker instance of AI Gateway. The Docker Compose file for the instance that the tests run against is under [`performance_test/setup`](../../performance_tests/setup/docker-compose.yml). The tests use [Grafana K6](https://grafana.com/docs/k6/latest/) to write the test.

The test uses the [Component Performance Testing](https://gitlab.com/gitlab-org/quality/component-performance-testing-aigw-poc) tool, which is maintained by [Performance Enablement Group](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/developer-experience/performance-enablement/). For any questions or feedback, reach out to the [`#g_performance_enablement`](https://gitlab.enterprise.slack.com/archives/C081476PPAM) Slack channel.

These tests are aimed to run against a mocked instance of AI Gateway. They ensure the changes submitted in the merge request cause no degradation in the TTFB values.

The `test:performance` job triggers a multi-project pipeline in [Component Performance Testing](https://gitlab.com/gitlab-org/quality/component-performance-testing-aigw-poc) project. This project spins up the AI Gateway instance as per the Docker Compose file, and runs the k6_test against it.

## Adding a new test

To add a new test:

1. Create a new `.js` file under [`performance_test/k6_test`](../../performance_tests/k6_test/).
1. Copy and paste the following boilerplate to it before beginning to write the test.
1. Update the comment sections with the respective values and code.

```javascript
export const TTFB_THRESHOLD= /* TTFB THRESHOLD VALUE EXPECTED */
export const RPS_THRESHOLD= /* RPS THRESHOLD VALUE EXPECTED */;
export const TEST_NAME=/* 'NAME OF THE TEST IN QUOTES' */
export const LOAD_TEST_VUS = 2; /* THE NUMBER OF THREADS OF ACTUAL TEST */
export const LOAD_TEST_DURATION = '50s'; /* THE DURATION FOR THE ACTUAL TEST RUN */
export const WARMUP_TEST_VUS = 1; /* THE NUMBER OF THREADS FOR WARMING UP THE SYSTEM */
export const WARMUP_TEST_DURATION = '10s'; /* THE DURATION FOR THE WARMUP RUN */
export const LOAD_TEST_START_TIME = '10s'; /* THE TIME TO WAIT AFTER WHICH THE LOAD TEST STARTS
                                              USUALLY THIS WOULD BE EQUAL TO WARMUP_TEST_DURATION */

export const options = {
scenarios:  {
    warmup: {
      executor: 'constant-vus',
      vus: WARMUP_TEST_VUS,
      duration: WARMUP_TEST_DURATION,
      gracefulStop: '0s',
      tags: { scenario: 'warmup' },
    },
    load_test: {
      executor: 'constant-vus',
      vus: LOAD_TEST_VUS,
      duration: LOAD_TEST_DURATION,
      startTime: LOAD_TEST_START_TIME,
      tags: { scenario: 'load_test' },
    },
  },
  thresholds: {
    'http_req_waiting{scenario:load_test}': [
      { threshold: `p(90)<${TTFB_THRESHOLD}`, abortOnFail: false }
    ],
    'http_reqs{scenario:load_test}': [
      { threshold: `rate>=${RPS_THRESHOLD}`, abortOnFail: false }
    ]
  },
};

export default function () {

  // WRITE THE TEST HERE


}

```

## Troubleshooting `tests:performance` job failures

The `tests:performance` job is designed to run only in merge request pipelines, and the main branch pipeline of the canonical [`ai-assist`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist) project.

### Common permission issues

If the `tests:performance` job fails due to permission issues, it might be caused by one of the following scenarios:

#### Bot-created MRs

If a bot created the merge request, verify whether the bot is managed by GitLab:

1. If you're unsure, ask in the `#it-help` channel whether the bot is managed by GitLab.
1. If confirmed, post a message in the `#performance-enablement` channel requesting someone to add the bot to the [Component Performance Testing](https://gitlab.com/gitlab-org/quality/component-performance-testing-aigw-poc) project.

#### Community contributions from forked repositories

Merge request [!2299](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/2299) implements a rule that prevents the `tests:performance` job from running in merge requests from forked repositories. This should no longer be an issue.

#### Other access-related issues

The `tests:performance` job is intended to run only on merge requests and in the `main` branch. If you observe it running in other scenarios:

1. Create a merge request to update the rules in the [CI configuration file](../../.gitlab/ci/performance.gitlab-ci.yml).
1. Ensure your changes receive the required reviews.
1. Merge the approved changes into the main branch.

### Code-related failures

If the `tests:performance` job fails due to code-related issues rather than permissions, post a message in the [`#g_performance_enablement`](https://gitlab.enterprise.slack.com/archives/C081476PPAM) Slack channel for help.

The `tests:performance` job should run only in merge request pipelines, and main branch pipeline of the canonical [`ai-assist`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist) project.
