/*global __ENV : true  */
/*
@endpoint: `POST /api/v4/ai/duo_workflows/workflows`
@description: API endpoint to begin a Duo Agent workflow for Chat
@gpt_data_version: 1
@stressed_components: Postgres, Gitaly, Rails
*/

import http from "k6/http";
import { check, group } from "k6";
import { Rate } from "k6/metrics";
import {
  logError,
  getRpsThresholds,
  getTtfbThreshold,
} from "../../lib/gpt_k6_modules.js";

// TODO: Placeholder Thresholds
export let thresholds = {
  'rps': { 'latest': 0.3 },
  'ttfb': { 'latest': 1500 },
};
export let rpsThresholds = getRpsThresholds(thresholds['rps'])
export let ttfbThreshold = getTtfbThreshold(thresholds['ttfb'])
export let successRate = new Rate("successful_requests");
export let options = {
  thresholds: {
    successful_requests: [`rate>${__ENV.SUCCESS_RATE_THRESHOLD}`],
    checks: [`rate>${__ENV.SUCCESS_RATE_THRESHOLD}`],
    http_req_waiting: [`p(90)<${ttfbThreshold}`],
    http_reqs: [`count>=${rpsThresholds["count"]}`],
  },
};

// If Service Account PAT is used for GPT, AI tests require real user PAT which can be provided via AI_ACCESS_TOKEN
export const access_token = __ENV.AI_ACCESS_TOKEN !== null && __ENV.AI_ACCESS_TOKEN !== undefined ? __ENV.AI_ACCESS_TOKEN : __ENV.ACCESS_TOKEN;

export function setup() {
  console.log("");
  console.log(`RPS Threshold: ${rpsThresholds["mean"]}/s (${rpsThresholds["count"]})`);
  console.log(`TTFB P90 Threshold: ${ttfbThreshold}ms`);
  console.log(`Success Rate Threshold: ${parseFloat(__ENV.SUCCESS_RATE_THRESHOLD) * 100}%`);
}

export default function () {
  group("API - Duo Agent - Chat", function () {
    let params = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${access_token}`,
      },
    };

    let body = {
      // This is expecting the Duo Workflow Test project to have been imported
      // to the instance, and the project ID specified in an environment file.
      project_id: __ENV.AI_DUO_WORKFLOW_PROJECT_ID,
      goal: "I am new to this project. Could you read the project structure and explain it to me?",
      start_workflow: true,
      workflow_definition: "chat",
      allow_agent_to_request_user: false,
      pre_approved_agent_privileges: [1,2,3,4,5],
      agent_privileges: [1,2,3,4,5]
    };

    let response = http.post(
      `${__ENV.ENVIRONMENT_URL}/api/v4/ai/duo_workflows/workflows`,
      JSON.stringify(body),
      params
    );

    if (!check(response, {'is status 201': (r) => r.status === 201})){
      successRate.add(false)
      logError(response)
      return
    }

    const checkOutput = check(response, {
      'verify that a workload id was provided for created job': (r) => r.json().workload.id !== undefined,
      'verify response has a created status': (r) => r.json().status == "created"
    });
    checkOutput ? successRate.add(true) : (successRate.add(false), logError(response));
  });
}
