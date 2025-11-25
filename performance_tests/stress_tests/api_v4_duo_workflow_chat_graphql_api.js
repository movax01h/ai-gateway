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

  This ends up using workhorse to proxy requests to Rails. No other executor: https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/#from-the-gitlab-web-ui-without-a-separate-executor

@stressed_components: Duo Workflow Service, Postgres, Rails
*/

import createDuoWorkflowChatTest from "./lib/duo_workflow_chat_base.js";

// Test configuration - use env var to determine which test to run
const scenarioType = __ENV.SCENARIO_TYPE || 'mocked_llm';

const projectId = __ENV.AI_DUO_WORKFLOW_PROJECT_ID;

const testConfigs = {
  real_llm: {
    goal: "I am new to this project. Could you read the project structure and explain it to me?",
    testName: "API - Duo Agent - Chat (Real LLM)",
    scenarioType: 'real_llm'
  },
  mocked_llm: {
    goal: `
Summarize http://gdk.test:3000/gitlab-duo/test/-/issues/1 and confirm that it has not been implemented by checking the existing files
<responses>
<response latency_ms='5000'>
I'll read the issue and then check the codebase to see if it has been implemented.
<tool_calls>[
  {"name": "get_work_item", "args": {"project_id": "gitlab-duo/test", "work_item_iid": "1"}},
  {"name": "list_repository_tree", "args": {"project_id": ${projectId}, "recursive": true}}
]</tool_calls>
</response>
<response latency_ms='5500'>
Now let me check the existing files to see what's already implemented:
<tool_calls>[
  {"name": "get_repository_file", "args": {"project_id": ${projectId}, "file_path": "README.md", "ref": "HEAD"}},
  {"name": "get_repository_file", "args": {"project_id": ${projectId}, "file_path": "python/server.py", "ref": "HEAD"}},
  {"name": "get_repository_file", "args": {"project_id": ${projectId}, "file_path": "ruby/server.rb", "ref": "HEAD"}}
]</tool_calls>
</response>
<response latency_ms='6000' stream='true' chunk_delay_ms='50'>
## Issue Summary

**Issue #1: "Add an example of GoLang HTTP server"**

- **Status**: Open
- **Assignee**: Administrator (root)
- **Description**: The issue requests adding an example of an HTTP server written in GoLang
- **Part of Epic**: "HTTP server examples for all programming languages" (Epic #1)

## Implementation Status

**NOT IMPLEMENTED** - The issue has not been implemented yet. Here's what currently exists in the repository:

### Current Structure:
\`\`\`
├── README.md (standard GitLab template)
├── python/
│   └── server.py (Python HTTP server implementation)
└── ruby/
    └── server.rb (Ruby HTTP server implementation)
\`\`\`

### What's Missing:
- No \`golang/\` or \`go/\` directory
- No Go HTTP server implementation
- No Go-related files (go.mod, main.go, etc.)

The repository currently contains HTTP server examples in Python and Ruby, but the requested GoLang implementation is completely missing. The issue remains open and unaddressed.
</response>
</responses>
`,
    testName: "API - Duo Agent - Chat (Mocked Responses)",
    scenarioType: 'mocked_llm'
  }
};

// Get configuration for the selected scenario
const config = testConfigs[scenarioType];
if (!config) {
  throw new Error(`Unknown scenario type: ${scenarioType}. Valid options: real_llm, mocked_llm`);
}

// Adjust as necessary to increase load
export const options = {
  scenarios: {
    sustain_50_requests: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
      gracefulStop: '1m',
    },
  },
};

export default createDuoWorkflowChatTest(config);
