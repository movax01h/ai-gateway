/*global __ENV : true */
/*
@endpoint: `POST /api/graphql`
@description: GraphQL endpoint to begin a Duo Agent workflow for Chat using k6 scenarios.

  WARNING: This test does NOT control how the Duo Workflow Service operates. By default, the Duo
  Workflow Service will make real LLM requests that incur costs. You must configure the Duo Workflow
  Service externally before running this test.

  To avoid real LLM costs, configure the Duo Workflow Service with one of the following:

  Option 1 - Mocked responses (no LLM calls):
  AIGW_MOCK_MODEL_RESPONSES=true
  AIGW_USE_AGENTIC_MOCK=true

  Option 2 - LLM proxy (cached/controlled LLM calls):
  Configure the Duo Workflow Service to use the LLM caching proxy.
  See docs/performance_testing/profiling_with_llm_caching_proxy.md

  This test uses WebSocket connections to receive LLM responses and has thresholds for establishing WebSocket
  connections and for WebSocket session duration.

  The test requires the following environment variables:
  - `ACCESS_TOKEN`: A personal access token with API scope
  - `AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID`: The id of a namespace with Duo Agent Platform enabled
  - `AI_DUO_WORKFLOW_PROJECT_ID`: The id of a project in the above namespace

  Optionally, set `SCENARIO_TYPE=real_llm|mocked_llm|llm_proxy` (default: llm_proxy)
  Optionally, set `MOCKED_GOAL_FILE` to specify a custom goal file within `performance_tests/stress_tests/goals` (default: security_analyst_agent/example_questions.yaml)
  Optionally, set `MODEL_IDENTIFIER` to specify a model selection (default: claude_haiku_4_5_20251001
  note this differs from the default in the GitLab UI, because the llm proxy only supports Anthropic as a provider)

  This ends up using workhorse to proxy requests to Rails. No other executor: https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/#from-the-gitlab-web-ui-without-a-separate-executor

@stressed_components: Duo Workflow Service, Postgres, Rails
*/

import http from "k6/http";
import { check, group } from "k6";
import { Rate } from "k6/metrics";
import { scenario } from "k6/execution";
import {
  connectWorkflowWebSocket,
  WORKFLOW_COMPLETE_TIMEOUT,
  loadGoalFile,
  getTestConfig,
  getScenarios,
  getAccessToken,
  logError,
  logDebug
} from "./lib/duo_workflow_chat_base.js";

const scenarioType = __ENV.SCENARIO_TYPE || 'llm_proxy';
const projectId = __ENV.AI_DUO_WORKFLOW_PROJECT_ID;
const gitlabURL = __ENV.ENVIRONMENT_URL || 'http://gdk.test:3000';
const goalFile = __ENV.MOCKED_GOAL_FILE || 'security_analyst_agent/example_questions.yaml';
const loadScenario = __ENV.LOAD_SCENARIO || 'default';
const workflowDefinition = __ENV.WORKFLOW_DEFINITION || 'chat';
const modelSelection = __ENV.MODEL_IDENTIFIER || 'claude_haiku_4_5_20251001';

const mockedGoalContent = loadGoalFile(scenarioType, goalFile, projectId, gitlabURL);
const config = getTestConfig(scenarioType, mockedGoalContent);
const successRate = new Rate("successful_requests");

// Handle both single goal and array of goals; fall back to config.goal for real_llm
const goals = Array.isArray(mockedGoalContent) ? mockedGoalContent : [mockedGoalContent ?? config.goal];

export const options = {
  scenarios: getScenarios(loadScenario),
};

export default function () {
  // Distribute goals across VUs using scenario iteration
  const goalIndex = scenario.iterationInTest % goals.length;
  const goal = goals[goalIndex];
  logDebug(`Goal: ${goal}`);
  const { testName = "API - Duo Agent - Chat (GraphQL)", scenarioType } = config;
  const access_token = getAccessToken();

  const groupName = scenarioType ? `${testName} [${scenarioType}]` : testName;

  group(groupName, function () {
    // Create workflow via GraphQL mutation
    const graphqlQuery = {
      query: `mutation createAiDuoWorkflow(
        $projectId: ProjectID
        $goal: String!
        $workflowDefinition: String!
        $agentPrivileges: [Int!]
        $preApprovedAgentPrivileges: [Int!]
      ) {
        aiDuoWorkflowCreate(
          input: {
            projectId: $projectId
            environment: WEB
            goal: $goal
            workflowDefinition: $workflowDefinition
            agentPrivileges: $agentPrivileges
            preApprovedAgentPrivileges: $preApprovedAgentPrivileges
            allowAgentToRequestUser: false
          }
        ) {
          workflow {
            id
          }
          errors
        }
      }`,
      variables: {
        projectId: `gid://gitlab/Project/${__ENV.AI_DUO_WORKFLOW_PROJECT_ID}`,
        goal: goal,
        workflowDefinition: workflowDefinition,
        agentPrivileges: [2, 3],
        preApprovedAgentPrivileges: [2, 3]
      }
    };

    let params = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${access_token}`,
        "X-GitLab-Interface": "duo_chat",
        "X-GitLab-Client-Type": "web_browser"
      },
    };

    let response = http.post(
      `${__ENV.ENVIRONMENT_URL}/api/graphql`,
      JSON.stringify(graphqlQuery),
      params
    );

    if (!check(response, {'is status 200': (r) => r.status === 200})){
      successRate.add(false);
      logError(response);
      return;
    }

    const responseData = response.json();
    const checkOutput = check(responseData, {
      'verify workflow was created': (r) => r.data?.aiDuoWorkflowCreate?.workflow?.id !== undefined,
      'no graphql errors': (r) => !r.data?.aiDuoWorkflowCreate?.errors || r.data.aiDuoWorkflowCreate.errors.length === 0
    });

    if (!checkOutput) {
      successRate.add(false);
      logError(response);
      return;
    }

    const workflowGid = responseData.data.aiDuoWorkflowCreate.workflow.id;
    const workflowId = workflowGid.split('/').pop();

    connectWorkflowWebSocket({
      workflowId: workflowId,
      goal: goal,
      accessToken: access_token,
      workflowDefinition: workflowDefinition,
      timeout: WORKFLOW_COMPLETE_TIMEOUT,
      userSelectedModelIdentifier: modelSelection,
      onComplete: (success) => {
        successRate.add(success);
      }
    });
  });
}
