/*global __ENV : true */
/*
@endpoint: `POST /api/graphql`
@description: GraphQL endpoint to begin a Duo Agent workflow for Chat using k6 scenarios.
  This test can run both real LLM responses and mocked responses based on scenario configuration.

  For mocked responses, the Duo Workflow Service should be configured with:
  AIGW_MOCK_MODEL_RESPONSES=true
  AIGW_USE_AGENTIC_MOCK=true

  It uses WebSocket connections to receive LLM responses and has thresholds for establishing WebSocket
  connections and for WebSocket session duration.

  The test requires the following environment variables:
  - `ACCESS_TOKEN`: A personal access token with API scope
  - `AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID`: The id of a namespace with Duo Agent Platform enabled
  - `AI_DUO_WORKFLOW_PROJECT_ID`: The id of a project in the above namespace

  Optionally, set `SCENARIO_TYPE=real_llm` for actual LLM responses, otherwise it uses mocked responses by default.
  Optionally, set `MOCKED_GOAL_FILE` to specify a custom goal file (default: goals/summarize_issue_check_implementation.txt)

  This ends up using workhorse to proxy requests to Rails. No other executor: https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/#from-the-gitlab-web-ui-without-a-separate-executor

@stressed_components: Duo Workflow Service, Postgres, Rails
*/

import http from "k6/http";
import { check, group } from "k6";
import { Rate } from "k6/metrics";
import {
  connectWorkflowWebSocket,
  WORKFLOW_COMPLETE_TIMEOUT,
  loadGoalFile,
  getTestConfig,
  getScenarios,
  getAccessToken,
  logError
} from "./lib/duo_workflow_chat_base.js";

const scenarioType = __ENV.SCENARIO_TYPE || 'mocked_llm';
const projectId = __ENV.AI_DUO_WORKFLOW_PROJECT_ID;
const gitlabURL = __ENV.ENVIRONMENT_URL || 'http://gdk.test:3000';
const goalFile = __ENV.MOCKED_GOAL_FILE || 'goals/summarize_issue_check_implementation.txt';
const loadScenario = __ENV.LOAD_SCENARIO || 'default';

const mockedGoalContent = loadGoalFile(scenarioType, goalFile, projectId, gitlabURL);
const config = getTestConfig(scenarioType, mockedGoalContent);
const successRate = new Rate("successful_requests");

export const options = {
  scenarios: getScenarios(loadScenario),
};

export default function () {
  const { goal, testName = "API - Duo Agent - Chat (GraphQL)", scenarioType } = config;
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
        workflowDefinition: "chat",
        agentPrivileges: [2, 3],
        preApprovedAgentPrivileges: [2]
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
      workflowDefinition: "chat",
      timeout: WORKFLOW_COMPLETE_TIMEOUT,
      onComplete: (success) => {
        successRate.add(success);
      }
    });
  });
}

