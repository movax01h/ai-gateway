import http from "k6/http";
import { check, group } from "k6";
import { Rate } from "k6/metrics";
import { WebSocket } from "k6/experimental/websockets";
import { SharedArray } from 'k6/data';

export const WORKFLOW_COMPLETE_TIMEOUT = 80000;
export const GRACEFUL_STOP_DURATION = '2m';

export function logDebug(...args) {
  if (__ENV.DEBUG === 'true') {
    console.log(...args);
  }
}

export function logError(...args) {
  console.error(...args);
}

/**
 * Loads and renders goal file content for mocked LLM scenarios
 * @param {string} scenarioType - Type of scenario ('real_llm' or 'mocked_llm')
 * @param {string} goalFile - Path to the goal file
 * @param {string} projectId - Project ID for template replacement
 * @param {string} gitlabURL - GitLab URL for template replacement
 * @returns {string|null} - Rendered goal content or null for real_llm scenarios
 */
export function loadGoalFile(scenarioType, goalFile, projectId, gitlabURL) {
  if (scenarioType !== 'mocked_llm') {
    return null;
  }

  const data = new SharedArray('goals', function () {
    const goalFilePath = import.meta.resolve(`../${goalFile}`);
    const rawContent = open(goalFilePath);
    const renderedContent = rawContent
        .replace(/\$\{projectId\}/g, projectId)
        .replace(/\$\{gitlabURL\}/g, gitlabURL);
    return [renderedContent];
  });

  return data[0];
}

/**
 * Gets test configuration based on scenario type
 * @param {string} scenarioType - Type of scenario ('real_llm' or 'mocked_llm')
 * @param {string|null} mockedGoalContent - Content for mocked scenario
 * @returns {Object} - Test configuration object
 */
export function getTestConfig(scenarioType, mockedGoalContent) {
  const testConfigs = {
    real_llm: {
      goal: "I am new to this project. Could you read the project structure and explain it to me?",
      testName: "API - Duo Agent - Chat (Real LLM)",
      scenarioType: 'real_llm'
    },
    mocked_llm: {
      goal: mockedGoalContent,
      testName: "API - Duo Agent - Chat (Mocked Responses)",
      scenarioType: 'mocked_llm'
    }
  };

  const config = testConfigs[scenarioType];
  if (!config) {
    throw new Error(`Unknown scenario type: ${scenarioType}. Valid options: real_llm, mocked_llm`);
  }

  return config;
}

/**
 * Gets k6 scenario configuration based on load scenario type
 * @param {string} loadScenario - Type of load scenario ('default', 'single', or 'spike')
 * @returns {Object} - k6 scenarios configuration
 */
export function getScenarios(loadScenario) {
  const scenarios = {
    // Default: Moderate sustained load
    default: {
      sustain_40: {
        executor: 'constant-vus',
        vus: 40,
        duration: '5m',
        gracefulStop: GRACEFUL_STOP_DURATION,
      },
    },

    // A single VU, useful for testing changes to the load test script or goals/fixtures
    single: {
      single: {
        executor: 'constant-vus',
        vus: 1,
        duration: '20s',
        gracefulStop: GRACEFUL_STOP_DURATION,
      },
    },

    // Spike test: Sudden jump to high load
    spike: {
      spike_test: {
        executor: 'ramping-vus',
        startVUs: 5,
        stages: [
          { duration: '1m', target: 5 },
          { duration: '30s', target: 70 },
          { duration: '1m', target: 5 },
        ],
        gracefulStop: GRACEFUL_STOP_DURATION,
      },
    },
  };

  return scenarios[loadScenario] || scenarios.default;
}

/**
 * Gets the appropriate access token from environment variables
 * For consistency with tests run via GitLab Performance Tool, checks AI_ACCESS_TOKEN
 * otherwise falls back to ACCESS_TOKEN (used for most other performance tests)
 * @returns {string} - Access token
 */
export function getAccessToken() {
  return __ENV.AI_ACCESS_TOKEN !== null && __ENV.AI_ACCESS_TOKEN !== undefined && __ENV.AI_ACCESS_TOKEN.trim()
    ? __ENV.AI_ACCESS_TOKEN
    : __ENV.ACCESS_TOKEN;
}

/**
 * Creates and manages a WebSocket connection for Duo Workflow chat
 * @param {Object} options - Configuration options
 * @param {string} options.workflowId - The workflow ID to connect to
 * @param {string} options.goal - The workflow goal
 * @param {string} options.accessToken - Bearer token for authentication
 * @param {string} options.workflowDefinition - Workflow definition type (default: "chat")
 * @param {number} options.timeout - Timeout in milliseconds (default: WORKFLOW_COMPLETE_TIMEOUT)
 * @param {Function} options.onComplete - Callback function called on close with success boolean
 */
export function connectWorkflowWebSocket(options) {
  const {
    workflowId,
    goal,
    accessToken,
    workflowDefinition = "chat",
    timeout = WORKFLOW_COMPLETE_TIMEOUT,
    onComplete
  } = options;

  const wsUrl = `${__ENV.ENVIRONMENT_URL.replace(/^http/, 'ws')}/api/v4/ai/duo_workflows/ws?project_id=${__ENV.AI_DUO_WORKFLOW_PROJECT_ID}&root_namespace_id=${__ENV.AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID}`;

  const ws = new WebSocket(wsUrl, null, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  ws.binaryType = 'blob';

  let receivedMessages = 0;
  let hasError = false;
  let workflowCompleted = false;

  ws.addEventListener('open', () => {
    logDebug('WebSocket connected');

    const startRequest = {
      startRequest: {
        workflowID: String(workflowId),
        clientVersion: "1.0",
        workflowDefinition: workflowDefinition,
        goal: goal
      }
    };

    ws.send(JSON.stringify(startRequest));

    ws.addEventListener('message', async (event) => {
      try {
        logDebug('Received message, data type:', typeof event.data);

        const messageText = typeof event.data === 'string' ? event.data : await event.data.text();
        logDebug('Message text:', messageText.substring(0, 100));

        const data = JSON.parse(messageText);
        logDebug('Parsed data:', JSON.stringify(data).substring(0, 200));

        // Messages are found in workflow checkpoint updates
        if (data.newCheckpoint && data.newCheckpoint.checkpoint) {
          const checkpoint = JSON.parse(data.newCheckpoint.checkpoint);

          if (checkpoint.channel_values && checkpoint.channel_values.ui_chat_log) {
            receivedMessages = checkpoint.channel_values.ui_chat_log.length;
            logDebug(`Received ${receivedMessages} chat messages`);
          }

          if (data.newCheckpoint.status) {
            const status = data.newCheckpoint.status.toLowerCase();
            logDebug(`Workflow status: ${data.newCheckpoint.status}`);

            // Mark workflow as completed for valid end states
            if (status === 'completed' || status === 'failed' || status === 'input_required') {
              workflowCompleted = true;
            }
          }
        }
      } catch (err) {
        logError('Error parsing websocket message:', err.message || err.toString());
        logError('Error stack:', err.stack);
        hasError = true;
        ws.close();
      }
    });

    ws.addEventListener('error', (err) => {
      logError('WebSocket error:', err);
      hasError = true;
    });

    const timeoutId = setTimeout(() => {
      if (!workflowCompleted) {
        logError('Timeout reached, closing connection');
        ws.close();
      }
    }, timeout);

    ws.addEventListener('close', () => {
      clearTimeout(timeoutId);
      logDebug('WebSocket closed');

      const wsCheckOutput = check(null, {
        'received workflow messages via websocket': () => receivedMessages > 0,
        'workflow completed successfully': () => workflowCompleted,
        'no websocket errors': () => !hasError
      });

      if (!wsCheckOutput) {
        logError('Workflow failed');
      }

      // Call the completion callback if provided
      if (onComplete) {
        onComplete(wsCheckOutput);
      }
    });
  });
}

export default function createDuoWorkflowChatTest(config) {
  const { goal, testName = "API - Duo Agent - Chat", scenarioType } = config;
  const successRate = new Rate("successful_requests");
  const access_token = getAccessToken();

  return function() {
    // Add scenario type to group name if provided
    const groupName = scenarioType ? `${testName} [${scenarioType}]` : testName;

    group(groupName, function () {
      // Create workflow via GraphQL mutation
      logDebug(`goal: ${goal}`);

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

      // Extract workflow ID from GraphQL response
      const workflowGid = responseData.data.aiDuoWorkflowCreate.workflow.id;
      const workflowId = workflowGid.split('/').pop();

      // Connect to websocket to receive workflow responses
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
  };
}
