import http from 'k6/http';
import { check, sleep } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';
export const TTFB_THRESHOLD= 25;
export const RPS_THRESHOLD= 2;
export const TEST_NAME='v2_code_completions'
export const LOAD_TEST_VUS = 2;
export const LOAD_TEST_DURATION = '50s';
export const WARMUP_TEST_VUS = 1;
export const WARMUP_TEST_DURATION = '10s';

export const options = {
scenarios:  {
    warmup: {
      executor: 'constant-vus',
      vus: WARMUP_TEST_VUS,
      duration: WARMUP_TEST_DURATION,
      gracefulStop: '0s',
      tags: { scenario: 'warmup' }, // Tag these requests to filter them out
    },
    load_test: {
      executor: 'constant-vus',
      vus: LOAD_TEST_VUS,
      duration: LOAD_TEST_DURATION,
      startTime: '10s', // Start after warmup completes
      tags: { scenario: 'load_test' },
    },
  },
  thresholds: {
    // Real thresholds that won't fail the test
    'http_req_waiting{scenario:load_test}': [
      { threshold: `p(90)<${TTFB_THRESHOLD}`, abortOnFail: false }
    ],
    'http_reqs{scenario:load_test}': [
      { threshold: `rate>=${RPS_THRESHOLD}`, abortOnFail: false }
    ]
  },
};

export default function () {
  const url = `http://${__ENV.AI_GATEWAY_IP}:5052/v2/code/completions`; // Replace with your API endpoint
  const payload = JSON.stringify({
    "project_path": "string",
    "project_id": 0,
    "current_file": {
      "file_name": "test",
      "language_identifier": "string",
      "content_above_cursor": "func hello_world(){\n\t",
      "content_below_cursor": "\n}"
    },
    "stream": true,
    "choices_count": 0,
    "context": [],
    "prompt_id": "code_suggestions/generations",
    "prompt_version": 2
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, params);

  console.log(`Request ${__ITER}: ${res.request.method} ${res.request.url} - Status ${res.status} - Duration ${res.timings.duration}ms`);

  sleep(1);
}

