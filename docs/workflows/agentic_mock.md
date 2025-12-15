# Using the agentic mock model

The [agentic mock model](../../ai_gateway/models/agentic_mock.py) lets you simulate multi-step workflows by specifying mock responses in your input.

Enable the agentic mock by setting both environment variables:

```shell
export AIGW_MOCK_MODEL_RESPONSES=true
export AIGW_USE_AGENTIC_MOCK=true
```

## Simple responses

Include your desired response in `<response>` tags in your input:

```xml
Summarize this issue for me.

<response>
  This issue discusses implementing a new authentication system with OAuth2 support.
</response>
```

## Tool calls

Include tool calls using `<tool_calls>` tags with JSON arrays:

```xml
Analyze the security implications of this code.

<response>
  I need to examine the code structure first.
  <tool_calls>
    [{"name": "read_file", "args": {"path": "src/auth.py"}}]
  </tool_calls>
</response>
```

Tool call requirements:

- Must be a JSON array: `[{"name": "tool1"}, {"name": "tool2"}]`
- Each tool call must have a `name` field
- Optional `args` field for parameters

## Sequential responses

Use multiple `<response>` tags to simulate multi-turn conversations:

```xml
Create a comprehensive test plan.

<response>
  I'll start by analyzing the requirements.
  <tool_calls>
    [{"name": "get_requirements", "args": {"project": "auth-system"}}]
  </tool_calls>
</response>
<response>
  Based on the requirements, here's the comprehensive test plan:

  1. Unit tests for authentication flows
  2. Integration tests for OAuth2 providers
  3. Security penetration tests
</response>
```

Note: at least one tool call must be included in a response for the agent to process the next response, as expected by [the workflow graph for Agentic Chat](../duo_workflow_service_graphs.md#graph-chat). The first `<response>` without a tool call will end the simulated workflow.

## Latency simulation

Add delays using the `latency_ms` attribute to specify the number of milliseconds the mock model should wait before returning the response:

```xml
<response latency_ms='1500'>
  This response will be delayed by 1.5 seconds
</response>
```

## Streaming simulation

Simulate streamed responses using the `stream` attribute. This will stream the response as chunks, where each chunk is a single word:

```xml
<response latency_ms='1500' stream='true' chunk_delay_ms='50'>
  After an initial delay of 1500ms this response will be streamed word by word with 50ms delay between chunks
</response>
```

## File-based responses

Agentic Chat limits the input goal to approximately 16,000 characters. That prevents us from specifying long
responses in the goal, which we might want to do to test long flows, for example.

If long mock responses are needed, you can load them from files that are included as server-side fixtures.

For example, create a file with your response content:

```plaintext
# performance_tests/stress_tests/fixtures/long_analysis.txt
This is a comprehensive analysis of the codebase...
[... many lines of detailed analysis ...]
```

In the goal, reference the file using the `file` attribute in a self-closing `<response>` tag:

```xml
Analyze this entire codebase in detail.

<response file='performance_tests/stress_tests/fixtures/long_analysis.txt' stream='true' chunk_delay_ms='50' />
```

Example with mixed inline and file-based responses:

```xml
Review this large codebase.

<response>
  I'll start by analyzing the architecture.
  <tool_calls>
    [{"name": "list_files", "args": {"path": "src/"}}]
  </tool_calls>
</response>
<response file='performance_tests/stress_tests/fixtures/long_analysis.txt' stream='true' chunk_delay_ms='100' />
<response>
  Based on my analysis, I recommend the following improvements...
</response>
```

## Template variable substitution

The agentic mock model supports Jinja2 template variable substitution, allowing you to define variables
in your input goal and substitute them into file-based responses.

For example, define variables using Jinja2 `{% set %}` statements at the beginning of your input:

```xml
{% set projectId = 1000001 %}

Explain what this project is about.

<response file='performance_tests/stress_tests/fixtures/list_repo_tree_tool_call.txt' />
```

Reference variables in the file (`performance_tests/stress_tests/fixtures/list_repo_tree_tool_call.txt`):

```xml
Let me examine the current project structure to understand the codebase.
<tool_calls>
[
  {"name": "gitlab_api_get", "args": {"endpoint": "/api/v4/projects/{{ projectId }}/repository/tree", "params": { "per_page": 100 }}}
]
</tool_calls>
```

When the file is loaded, `{{ projectId }}` will be replaced with `1000001`.

## Example workflows

Workflow with 2 tool calls:

```xml
Review this merge request for security issues.

<response>
  I'll examine the changes systematically.
  <tool_calls>
    [{"name": "get_pr_files", "args": {"pr_id": "123"}}]
  </tool_calls>
</response>
<response>
  I found several security concerns:
  1. SQL injection vulnerability in user input handling
  2. Missing authentication checks on admin endpoints
  <tool_calls>
    [{
      "name": "create_security_report",
      "args": {"pr_id": "123", "issues": ["sql_injection", "auth_bypass"]}
    }]
  </tool_calls>
</response>
<response>
  Security report created. I recommend addressing these issues before merging.
</response>
```

Workflow with simulated latency for 2 tool calls:

```xml
Help me understand this bug report.

<response latency_ms='800'>
  Let me gather information about this issue.
  <tool_calls>
    [{"name": "search_similar_issues", "args": {"keywords": ["authentication", "timeout"]}}]
  </tool_calls>
</response>
<response latency_ms='1200'>
  Based on similar issues, this appears to be related to session timeout handling.
  <tool_calls>
    [{"name": "get_code_context", "args": {"file": "session_manager.py", "line": 45}}]
  </tool_calls>
</response>
<response>
  The bug is in the session timeout logic.
  The fix is to update the session renewal mechanism in `session_manager.py`.
</response>
```

### The agentic mock model in practice: memory utilization testing

The agentic mock model was used to
[investigate and reproduce high memory usage in the Duo Workflow Service](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1648).
By simulating large streamed responses, the testing confirmed that checkpoint data accumulation was a
significant source of memory consumption.

#### Test files used

- [`performance_tests/stress_tests/goals/stream_long_response.txt`](../../performance_tests/stress_tests/goals/stream_long_response.txt) -
Tests streamed long responses with 6-second latency and 50ms chunk delays
- [`performance_tests/stress_tests/goals/no_stream_long_response.txt`](../../performance_tests/stress_tests/goals/no_stream_long_response.txt) -
Tests non-streamed long responses with 60-second latency for roughly comparable overall delays
- [`performance_tests/stress_tests/fixtures/50kb_response.txt`](../../performance_tests/stress_tests/fixtures/50kb_response.txt) -
Contains ~50KB of response content (approximately 1200 lines of code) that was used in the above goals
