# Performance testing

This document shows examples (based on those [provided by Stan Hu](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/77#note_1402837192)
and [Alexander Chueshev](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/126))
of using [`perf_analyzer`](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/README.md) to evaluate the performance of the Triton server.

## Single prompt

Sign in to the Triton server and save the following to [`test.json`](assets/test.json) to use as input data:

```json
{
  "data": [
    {
      "prompt": {
        "content": ["<python>def is_even(na: int) ->"],
        "shape": [1]
      },
      "request_output_len": { "content": [32], "shape": [1] },
      "temperature": [0.20000000298023224],
      "repetition_penalty": [1.0],
      "runtime_top_k": [0],
      "runtime_top_p": [0.9800000190734863],
      "start_id": [50256],
      "end_id": [50256],
      "random_seed": [1594506369],
      "is_return_log_probs": [true]
    }
  ]
}
```

Run `perf_analyzer`:

```shell
perf_analyzer -m ensemble --percentile=95 --concurrency-range 1:4 --input-data test.json
```

The output ends with a summary, like:

```plaintext
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 2.44411 infer/sec, latency 409016 usec
Concurrency: 2, throughput: 2.44411 infer/sec, latency 813556 usec
Concurrency: 3, throughput: 2.44409 infer/sec, latency 1220557 usec
Concurrency: 4, throughput: 2.44411 infer/sec, latency 1627208 usec
```

<details>
<summary>Expand for a complete example of the report</summary>

```shell
$ perf_analyzer -m ensemble --percentile=95 --concurrency-range 1:4 --input-data test

 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 4 concurrent requests
  Using synchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 44
    Throughput: 2.44411 infer/sec
    p50 latency: 408536 usec
    p90 latency: 408932 usec
    p95 latency: 409016 usec
    p99 latency: 414229 usec
    Avg HTTP time: 408686 usec (send/recv 75 usec + response wait 408611 usec)
  Server:
    Inference count: 44
    Execution count: 44
    Successful request count: 44
    Avg request latency: 408328 usec (overhead 95 usec + queue 171 usec + compute 408062 usec)

  Composing models:
  fastertransformer, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 406661 usec (overhead 78 usec + queue 66 usec + compute input 114 usec + compute infer 406144 usec + compute output 258 usec)

  postprocessing, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 803 usec (overhead 11 usec + queue 60 usec + compute input 71 usec + compute infer 478 usec + compute output 182 usec)

  preprocessing, version:
      Inference count: 45
      Execution count: 45
      Successful request count: 45
      Avg request latency: 869 usec (overhead 11 usec + queue 45 usec + compute input 21 usec + compute infer 720 usec + compute output 71 usec)

Request concurrency: 2
  Client:
    Request count: 44
    Throughput: 2.44411 infer/sec
    p50 latency: 813212 usec
    p90 latency: 813506 usec
    p95 latency: 813556 usec
    p99 latency: 813760 usec
    Avg HTTP time: 24395907 usec (send/recv 2219 usec + response wait 24393688 usec)
  Server:
    Inference count: 44
    Execution count: 44
    Successful request count: 44
    Avg request latency: 803268 usec (overhead 103 usec + queue 394618 usec + compute 408547 usec)

  Composing models:
  fastertransformer, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 801086 usec (overhead 78 usec + queue 394515 usec + compute input 139 usec + compute infer 406114 usec + compute output 239 usec)

  postprocessing, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 1311 usec (overhead 14 usec + queue 61 usec + compute input 71 usec + compute infer 519 usec + compute output 646 usec)

  preprocessing, version:
      Inference count: 45
      Execution count: 45
      Successful request count: 45
      Avg request latency: 869 usec (overhead 9 usec + queue 42 usec + compute input 21 usec + compute infer 726 usec + compute output 70 usec)

Request concurrency: 3
  Client:
    Request count: 44
    Throughput: 2.44409 infer/sec
    p50 latency: 1219880 usec
    p90 latency: 1220283 usec
    p95 latency: 1220557 usec
    p99 latency: 1220691 usec
    Avg HTTP time: 0 usec (send/recv 0 usec + response wait 0 usec)
  Server:
    Inference count: 44
    Execution count: 44
    Successful request count: 44
    Avg request latency: 1198039 usec (overhead 105 usec + queue 789386 usec + compute 408548 usec)

  Composing models:
  fastertransformer, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 1195852 usec (overhead 76 usec + queue 789289 usec + compute input 132 usec + compute infer 406113 usec + compute output 241 usec)

  postprocessing, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 1303 usec (overhead 15 usec + queue 56 usec + compute input 74 usec + compute infer 519 usec + compute output 638 usec)

  preprocessing, version:
      Inference count: 45
      Execution count: 45
      Successful request count: 45
      Avg request latency: 878 usec (overhead 8 usec + queue 41 usec + compute input 21 usec + compute infer 736 usec + compute output 71 usec)

Request concurrency: 4
  Client:
    Request count: 44
    Throughput: 2.44411 infer/sec
    p50 latency: 1626522 usec
    p90 latency: 1627196 usec
    p95 latency: 1627208 usec
    p99 latency: 1627557 usec
    Avg HTTP time: 0 usec (send/recv 0 usec + response wait 0 usec)
  Server:
    Inference count: 44
    Execution count: 44
    Successful request count: 44
    Avg request latency: 1593023 usec (overhead 103 usec + queue 1184297 usec + compute 408623 usec)

  Composing models:
  fastertransformer, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 1590807 usec (overhead 77 usec + queue 1184199 usec + compute input 148 usec + compute infer 406134 usec + compute output 248 usec)

  postprocessing, version:
      Inference count: 44
      Execution count: 44
      Successful request count: 44
      Avg request latency: 1333 usec (overhead 16 usec + queue 54 usec + compute input 74 usec + compute infer 516 usec + compute output 672 usec)

  preprocessing, version:
      Inference count: 45
      Execution count: 45
      Successful request count: 45
      Avg request latency: 882 usec (overhead 9 usec + queue 44 usec + compute input 22 usec + compute infer 734 usec + compute output 72 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 2.44411 infer/sec, latency 409016 usec
Concurrency: 2, throughput: 2.44411 infer/sec, latency 813556 usec
Concurrency: 3, throughput: 2.44409 infer/sec, latency 1220557 usec
Concurrency: 4, throughput: 2.44411 infer/sec, latency 1627208 usec
```

</details>

## Increasing concurrency

Evaluate [concurrent model execution](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#concurrent-model-execution)
by [increasing concurrency levels](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/cli.md#--concurrency-rangestartendstep). For example:

```shell
$ perf_analyzer -m ensemble --percentile=95 --concurrency-range 1:91:10 --input-data test.json

...

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 2.4441 infer/sec, latency 409490 usec
Concurrency: 11, throughput: 2.49955 infer/sec, latency 4484831 usec
Concurrency: 21, throughput: 2.49963 infer/sec, latency 8560859 usec
Concurrency: 31, throughput: 2.44411 infer/sec, latency 12635924 usec
Concurrency: 41, throughput: 2.44408 infer/sec, latency 16711496 usec
Concurrency: 51, throughput: 2.44401 infer/sec, latency 17004964 usec
Concurrency: 61, throughput: 2.44399 infer/sec, latency 20786874 usec
Concurrency: 71, throughput: 2.444 infer/sec, latency 24861428 usec
Concurrency: 81, throughput: 2.44405 infer/sec, latency 28932473 usec
Concurrency: 91, throughput: 2.44408 infer/sec, latency 33024001 usec
```

## `tests:performance` job

The `test:performance` job runs in MR pipelines. It runs all the tests under the [`performance_test/k6_test`](../performance_tests/k6_test/) directory against a Docker instance of AI Gateway. The Docker Compose file for the instance that the tests run against is under [`performance_test/setup`](../performance_tests/setup/docker-compose.yml). The tests use [Grafana K6](https://grafana.com/docs/k6/latest/) to write the test.

The test uses the [Component Performance Testing](https://gitlab.com/gitlab-org/quality/component-performance-testing-aigw-poc) tool, which is maintained by [Performance Enablement Group](https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/developer-experience/performance-enablement/). For any questions or feedback, reach out to the [`#g_performance_enablement`](https://gitlab.enterprise.slack.com/archives/C081476PPAM) Slack channel.

These tests are aimed to run against a mocked instance of AI Gateway. They ensure the changes submitted in the merge request cause no degradation in the TTFB values.

The `test:performance` job triggers a multi-project pipeline in [Component Performance Testing](https://gitlab.com/gitlab-org/quality/component-performance-testing-aigw-poc) project. This project spins up the AI Gateway instance as per the Docker Compose file, and runs the k6_test against it.

### Adding a new test

To add a new test:

1. Create a new `.js` file under [`performance_test/k6_test`](../performance_tests/k6_test/).
1. Copy and paste the following boilerplate to it before beginning to write the test.
1. Update the comment sections with the respective values and code.

```javascript
// Import necessary modules
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';
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

1. Create a merge request to update the rules in the [CI configuration file](../.gitlab/ci/performance.gitlab-ci.yml).
1. Ensure your changes receive the required reviews.
1. Merge the approved changes into the main branch.

### Code-related failures

If the `tests:performance` job fails due to code-related issues rather than permissions, post a message in the [`#g_performance_enablement`](https://gitlab.enterprise.slack.com/archives/C081476PPAM) Slack channel for help.

The `tests:performance` job should run only in merge request pipelines, and main branch pipeline of the canonical [`ai-assist`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist) project.
