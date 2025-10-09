# Flow Registry Framework experimental version documentation

This page documents capabilities of _experimental_ version of Flow Registry.
This version is not being considered as stable and can change at any moment,
**do not use it for any other purpose then internal development**

[[_TOC_]]

## YAML Configuration Structure

YAML configuration files define the structure and behavior of your flows.
Every flow YAML file must contain these top-level sections that specify components, routing logic, and execution
parameters.

```yaml
version: "experimental"
environment: remote  # or "local" for development

components:
# List of components (see Component Types section)

routers:
# Define flow between components (see examples below)

flow:
    entry_point: "component_name"  # Name of first component to run
```

### Required Fields

- **version**: Always use `"experimental"` for the current framework version
- **environment**: Set to `"remote"` for delegated tasks that agents should do in the background with little to non user
  interactions or `"local"` for pair coding experience, when user is expected to collaborate with agents in real time.
- **components**: List of components that make up your flow
- **routers**: Define how components connect to each other
- **flow**: Specify the entry point component and other options

## Component Types

### AgentComponent

The AgentComponent is the primary building block for AI-powered flows.
An AgentComponent uses a Large Language Model (LLM) to process inputs and generate responses. It provides these
capabilities:

- Execute tools to interact with the environment (read files, run commands, etc.)
- Maintain conversation history for context
- Make decisions based on the provided prompt and available tools
- Generate structured outputs that other components consume

The agent uses the prompt template from the prompt registry, processes the inputs through the LLM, and calls tools as
needed to complete its task.

#### Required Parameters

- **name**: Unique identifier for this component instance. Must not contain `:` or `.` characters.
- **type**: Must be `"AgentComponent"`
- **prompt_id**: ID of the prompt template from either the prompt registry or locally defined prompts

#### Optional Parameters

- **prompt_version**: Semantic version constraint (e.g., `"^1.0.0"`). If omitted or `null`, uses locally defined prompt
  from flow YAML.
- **inputs**: List of input data sources (default: `["context:goal"]`)
- **toolset**: List of tools available to the agent
- **ui_log_events**: UI logging configuration
- **ui_role_as**: Display role in UI (`"agent"` or `"tool"`)

#### Available Tools

Agents access the following tools in their `toolset` configuration.
Complete list of tools classes can be located at `duo_workflow_service/components/tools_registry.py`. To configure an
agent with tools, pass `name` attributes from their classes, eg: for `GetIssue` tool class pass `get_issue`
Each tool is a Python class that the agent calls to perform specific actions.

Here are some examples:

- **read_file**: Read contents of a file
- **create_file_with_contents**: Create a new file with specified content
- **edit_file**: Modify an existing file
- **list_dir**: List directory contents
- **find_files**: Search for files matching patterns

#### Prompts

Prompts define how the AI agent behaves and responds to inputs.
Every AgentComponent requires a prompt that serves as the "instructions" for the AI agent, defining its personality,
capabilities, and response patterns.
Prompts include placeholders that get replaced with actual data from the flow state.

**Important for Multi-turn Conversations:** If your AgentComponent needs access to conversation history (for context in
multi-turn interactions), include `placeholder: history` in your prompt template. This automatically provides the
component's conversation history to the AI model.

##### Prompt Sources

Flows can use prompts from two sources:

1. **File-based Prompts**: Traditional prompts stored in the AI Gateway prompt registry at
   `ai_gateway/prompts/definitions/`. Reference these using both `prompt_id` and `prompt_version` (e.g.,
   `prompt_version: "^1.0.0"`).

1. **Locally Defined Prompts**: Prompts defined directly within the flow YAML configuration. Reference these by setting
   `prompt_version: null` or omitting the `prompt_version` field entirely.

##### Locally Defined Prompts

Flows can define prompts locally within their YAML configuration, eliminating the need for separate prompt definition
files. This approach is particularly useful for flow-specific prompts or rapid prototyping.

To define prompts locally, add a `prompts` section to your flow YAML:

```yaml
version: "experimental"
environment: remote

components:
    - name: "code_analyzer"
      type: AgentComponent
      prompt_id: "my_local_prompt"
      # prompt_version omitted - uses local prompt
      ui_log_events: [ "on_agent_final_answer" ]
      inputs: ["context:goal"]
      toolset: ["read_file", "list_dir"]

prompts:
    - prompt_id: "my_local_prompt"
      model:
        params:
            model_class_provider: anthropic
            model: claude-sonnet-4-20250514
            max_tokens: 32_768
      prompt_template:
          system: "You are a helpful assistant specialized in code analysis"
          user: "Task: {{goal}}"
          placeholder: history  # Optional: include conversation history
      params:
          timeout: 180

routers:
    - from: "code_analyzer"
      to: "end"

flow:
    entry_point: "code_analyzer"

```

**Routing Logic**: The system automatically routes between local and file-based prompts:

- `prompt_version: null` or omitted → Use local prompt from flow YAML
- `prompt_version: "^1.0.0"` → Use file-based prompt from prompt registry

**Local Prompt Structure**:

Mirrors the prompt structure of prompts taken from the prompt registry. More information about the structure can be
found [here](../aigw_prompt_registry.md#ai-gateway-prompt-configuration-reference).

**Benefits of Local Prompts**:

- **Rapid Development**: No need to create separate prompt files
- **Flow Coupling**: Prompts stay co-located with their usage
- **Simplified Testing**: Easy to iterate on prompts during development
- **Self-Contained**: Flows become more portable and self-documented

For detailed information about prompt templates and placeholder configuration, see
the [AI Gateway Prompt Registry Documentation](../aigw_prompt_registry.md#message-placeholders).

#### Inputs

Agent inputs work together with prompt placeholders to provide dynamic data to the AI.
When you define inputs in your component configuration, that data becomes available as template variables in your
prompt.

For example:

```yaml
# In your flow YAML:
inputs:
    - from: "context:user_request"
      as: "task_description"
    - from: "context:analyzer.findings"
      as: "analysis_results"
```

```yaml
# In your prompt template:
user: |
    Task: {{task_description}}
    Analysis: {{analysis_results}}
    Please provide a solution.
```

#### Outputs

Each AgentComponent automatically produces:

- **context:{component_name}.final_answer**: The agent's final response
- **conversation_history:{component_name}**: Message history
- **status**: Updated workflow status

These outputs can be used as inputs by other components or referenced in routing logic to control flow execution.
For example, you might route to different components based on whether the agent's final answer equals to specific phrase

#### UI Log Events

The AgentComponent supports the following UI log events that can be specified in the `ui_log_events` configuration:

- **on_agent_final_answer**: Logged when the agent produces its final response. This event captures the agent's
  conclusion or final output and displays it in the UI.
- **on_tool_execution_success**: Logged when a tool is successfully executed by the agent. This shows users what actions
  the agent took and their successful results.
- **on_tool_execution_failed**: Logged when a tool execution fails. This provides error information and helps with
  debugging failed operations.

#### Complete AgentComponent Example

```yaml
components:
    - name: "code_assistant"
      type: AgentComponent
      prompt_id: "code_review_helper"
      prompt_version: "^1.0.0"
      inputs: [ "context:goal" ]
      toolset:
          - "read_file"
          - "list_dir"
          - "find_files"
          - "create_file_with_contents"
          - "edit_file"
      ui_log_events:
          - "on_agent_final_answer"
          - "on_tool_execution_success"
          - "on_tool_execution_failed"
      ui_role_as: "agent"
```

### HumanInputComponent

The HumanInputComponent enables human-in-the-loop interactions within flows by requesting and processing user input
during workflow execution.
This component allows workflows to pause execution, request user feedback or decisions, and then continue based on the
user's response.

The component provides these capabilities:

- **Request user input**: Display optional prompts to guide user responses
- **Interrupt workflow execution**: Cleanly pause the workflow until user input is received
- **Process different response types**: Handle text responses, approval/rejection decisions
- **Route responses**: Direct user input to specified target components in the conversation history
- **Store approval decisions**: Capture user approval/rejection decisions in the flow context

The HumanInputComponent consists of two internal nodes:

- **RequestNode**: Transitions the workflow to `INPUT_REQUIRED` status and optionally displays prompts to the user
- **FetchNode**: Waits for user input via interrupt() and processes the response based on event type

#### Required Parameters

- **name**: Unique identifier for this component instance. Must not contain `:` or `.` characters.
- **type**: Must be `"HumanInputComponent"`
- **sends_response_to**: Name of the target component that should receive the user's response in conversation history

#### Optional Parameters

- **prompt_id**: ID of the prompt template from the prompt registry to display to the user
- **prompt_version**: Semantic version constraint for the prompt (e.g., `"^1.0.0"`)
- **inputs**: List of input data sources for prompt rendering (default: empty list)
- **ui_log_events**: UI logging configuration for displaying prompts

#### Supported Event Types

The HumanInputComponent processes different types of user events:

- **RESPONSE**: Regular text input from the user that gets added to conversation history
- **APPROVE**: User approval decision that gets stored in the context as `"approve"`
- **REJECT**: User rejection decision that gets stored in the context as `"reject"`, optionally with a message added to
  conversation history

#### Outputs

Each HumanInputComponent automatically produces:

- **conversation_history:{sends_response_to}**: User messages directed to the target component
- **context:{component_name}.approval**: User approval decision (`"approve"` or `"reject"`)

#### UI Log Events

The HumanInputComponent supports the following UI log event:

- **on_user_input_prompt**: Logged when displaying a prompt to request user input. This shows the prompt content in the
  UI to guide the user's response.

#### Environment Support

The HumanInputComponent is only supported in the `"ide"` environment, as it requires the interrupt mechanism for pausing
workflow execution.

#### Complete HumanInputComponent Example

```yaml
components:
    - name: "user_approval"
      type: HumanInputComponent
      sends_response_to: "code_assistant"
      prompt_id: "approval_request"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:code_assistant.final_answer"
            as: "proposed_changes"
      ui_log_events:
          - "on_user_input_prompt"

    - name: "code_assistant"
      type: AgentComponent
      prompt_id: "code_review_helper"
      prompt_version: "^1.0.0"
      inputs:
          - "context:goal"
          - from: "context:user_approval.approval"
            as: "user_decision"
      toolset: [ "read_file", "edit_file" ]
```

#### Usage Patterns

**Approval Workflow**: Use HumanInputComponent to request user approval before proceeding with actions:

```yaml
routers:
    - from: "user_approval"
      condition:
          input: "context:user_approval.approval"
          routes:
              "approve": "execute_changes"
              "reject": "revise_proposal"
              "default_route": "manual_review"
```

**Interactive Chat**: Enable back-and-forth conversation between user and agent:

```yaml
routers:
    - from: "user_input"
      to: "chat_agent"
    - from: "chat_agent"
      to: "user_input"  # Loop back for continued interaction
```

**Conditional Input**: Request user input only when certain conditions are met:

```yaml
routers:
    - from: "analyzer"
      condition:
          input: "context:analyzer.confidence"
          routes:
              "low": "user_clarification"
              "high": "auto_processor"
```

### DeterministicStepComponent

The DeterministicStepComponent executes a **single tool** deterministically with predetermined arguments extracted from the flow state. This component provides a way to run one specific tool without AI involvement, using inputs to extract the necessary parameters and producing predictable outputs following fixed conventions.

The component provides these capabilities:

- **Execute a single tool deterministically**: Run one designated tool with parameters derived from inputs
- **Extract parameters from state**: Use component inputs to gather the tool's execution arguments
- **No AI involvement**: Direct tool execution without LLM processing
- **Integration with existing toolsets**: Compatible with any registered tool in the toolset
- **Chainable design**: Multiple DeterministicStepComponents can be chained to execute sequential tool operations

Unlike AgentComponent or OneOffComponent which use AI to determine tool usage, DeterministicStepComponent executes exactly one pre-specified tool with arguments extracted directly from the flow state, making it ideal for predictable, repeatable operations. **To execute multiple tools, chain multiple DeterministicStepComponents together in your flow.**

#### Required Parameters

- **name**: Unique identifier for this component instance. Must not contain `:` or `.` characters.
- **type**: Must be `"DeterministicStepComponent"`
- **tool_name**: Name of the single tool to execute

#### Optional Parameters

- **toolset**: Toolset containing the tool to be executed. (If no toolset is specified, a new one is created with only the `tool_name`)
- **inputs**: List of input data sources to extract tool parameters (default: empty list)
- **ui_log_events**: UI logging configuration for displaying tool execution
- **ui_role_as**: Display role in UI (default: `"tool"`)

#### Outputs

Each DeterministicStepComponent automatically produces:

- **ui_chat_log**: UI logging information for tool execution events
- **context:{component_name}.tool_responses**: Record of successful tool execution results
- **context:{component_name}.error**: Record of any errors during the tool call
- **context:{component_name}.execution_result**: Status of the execution ("success" or "failed")

#### Complete DeterministicStepComponent Example

##### Execute a single tool

```yaml
components:
   - name: "read"
     type: DeterministicStepComponent
     inputs:
       - from: "context:goal"
         as: "file_path"
     tool_name: "read_file"
     ui_log_events:
       - "on_tool_execution_success"
       - "on_tool_execution_failed"
```

##### Chain multiple tools

```yaml
components:
    - name: "read_config"
      type: DeterministicStepComponent
      inputs:
          - from: "context:goal"
            as: "config_path"
      tool_name: "read_file"
    - name: "backup_config"
      type: DeterministicStepComponent
      inputs:
          - from: "context:read_config.tool_responses"
            as: "contents"
          - from: "config_backup.txt"
            as: "file_path"
            literal: true
      tool_name: "create_file_with_contents"
```

#### Validation

The DeterministicStepComponent performs automatic validation of tool arguments:

- Validates that the specified tool exists in the provided toolset
- Checks that all required tool parameters are configured in the inputs
- Verifies that configured parameters match the tool's expected schema
- Raises clear validation errors during component initialization if configuration is invalid

This validation ensures that tool execution errors are caught at configuration time rather than runtime.

### OneOffComponent

The `OneOffComponent` functionally sits in-between the `AgentComponent` and the `DeterministicStepComponent`.
`OneOffComponent` works by taking a pre-defined toolset with an input and generating tool calls in a single round, then
finally exiting when those tool calls have been successfully executed.
The component has the ability to retry failed tool executions and iterate up to the `max_correction_attempts`.

The `OneOffComponent` is designed for scenarios where you need to execute one or more tool operations in a single round
with built-in error handling and retry logic.
Unlike the `AgentComponent` which can engage in multi-turn conversations and generate additional tool calls after seeing
results,
`OneOffComponent` is constrained to a single round of tool generation and execution.

**Note:** If your OneOffComponent prompt needs access to conversation history from previous interactions, include
`placeholder: history` in your prompt template.

#### Key Features

- **Single Round Tool Execution**: Executes one or more tool calls in a single round and exits upon successful
  completion
- **Multiple Tool Support**: Can use multiple tools from its toolset as needed to complete the task
- **Error Correction**: Automatically retries failed tool executions with error feedback
- **Configurable Retry Logic**: Set maximum correction attempts via `max_correction_attempts` parameter
- **Built-in Tool Routing**: Intelligent routing between LLM and tool nodes based on execution results
- **UI Logging**: Comprehensive logging of tool execution states and results

#### Required Parameters

- **name**: Unique identifier for this component instance
- **type**: Must be `"OneOffComponent"`
- **prompt_id**: ID of the prompt template from either the prompt registry or locally defined prompts
- **toolset**: List of tools available to the component

#### Optional Parameters

- **prompt_version**: Semantic version constraint (e.g., `"^1.0.0"`). If omitted or `null`, uses locally defined prompt
  from flow YAML.
- **inputs**: List of input data sources (default: `["context:goal"]`)
- **max_correction_attempts**: Maximum number of retry attempts for failed tool executions (default: 3)
- **ui_log_events**: UI logging configuration for displaying tool execution progress

#### Internal Architecture

The OneOffComponent consists of three internal nodes:

1. **LLM Node** (`{name}#llm`): Uses `AgentNode` to generate one or more tool calls based on the prompt and inputs
1. **Tools Node** (`{name}#tools`): Executes all generated tool calls with error correction using
   `ToolNodeWithErrorCorrection`
1. **Exit Node** (`{name}#exit`): Handles component completion and state logging

The component uses conditional routing to handle tool execution results:

- **Success**: Routes to exit node when all tool executions complete successfully
- **Retry**: Returns to LLM node when errors occur and retry attempts remain
- **Max Attempts**: Routes to exit node when maximum correction attempts are reached

#### Comparison with AgentComponent

The OneOffComponent and AgentComponent differ in their execution patterns:

**OneOffComponent**:

- **Single Round**: Generates tool calls once and executes them all in one round
- **Task-Focused**: Designed for specific, bounded tasks that can be completed in one execution cycle
- **No Iterative Reasoning**: Cannot see tool results and decide on additional actions
- **Simpler Flow**: Linear progression from tool generation → execution → completion
- **Error Handling**: Built-in retry logic for failed tool executions

**AgentComponent**:

- **Multi-Turn Conversations**: Can generate tools, see results, and decide on next actions
- **Iterative Decision Making**: Can analyze tool results and generate additional tool calls
- **Complex Reasoning**: Supports back-and-forth between LLM and tools until task completion
- **Final Output Control**: Uses `AgentFinalOutput` tool to explicitly signal completion
- **Conversation Flow**: Maintains ongoing conversation history for context

**When to Use Each**:

- Use **OneOffComponent** for: File operations, data processing, single API calls, or any task that can be completed in
  one execution round
- Use **AgentComponent** for: Interactive tasks, complex problem-solving, multi-step workflows requiring decision-making
  between steps

#### Outputs

Each OneOffComponent automatically produces:

- **ui_chat_log**: UI logging information for tool execution events
- **conversation_history:{component_name}**: Message history for the component
- **context:{component_name}.tool_calls**: Record of tool calls made by the component
- **context:{component_name}.tool_responses**: Record of tool responses received
- **context:{component_name}.execution_result**: Execution result ("success" or "failed")

#### UI Log Events

The OneOffComponent supports the following UI log events from `UILogEventsOneOff` that can be specified in the
`ui_log_events` configuration:

- **on_tool_call_input**: Logged when a tool is about to be called with its input arguments
- **on_tool_execution_success**: Logged when a tool executes successfully
- **on_tool_execution_failed**: Logged when a tool execution fails

#### Complete OneOffComponent Example

```yaml
components:
    - name: "file_reader"
      type: OneOffComponent
      prompt_id: "read_specific_file"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:goal"
            as: "target_file"
      toolset:
          - "read_file"
      max_correction_attempts: 2
      ui_log_events:
          - "on_tool_call_input"
          - "on_tool_execution_success"
          - "on_tool_execution_failed"
```

#### Usage Patterns

**Single File Operation**: Use OneOffComponent for singular file operations:

```yaml
components:
    - name: "config_updater"
      type: OneOffComponent
      prompt_id: "update_config_file"
      prompt_version: "^1.0.0"
      inputs:
          - "context:goal"
      toolset:
          - "edit_file"
      max_correction_attempts: 3
```

**Conditional Tool Execution**: Only proceeds when tool call was a success:

```yaml
routers:
    - from: "file_processor"
      condition:
          input: "context:file_processor.execution_result"
          routes:
              "success": "next_step"
              "failed": "error_handler"
```

**Error Handling Integration**: Combine with other components for robust workflows:

```yaml
components:
    - name: "backup_creator"
      type: OneOffComponent
      prompt_id: "create_backup"
      prompt_version: "^1.0.0"
      toolset: [ "create_file_with_contents" ]
      max_correction_attempts: 5

    - name: "error_reporter"
      type: AgentComponent
      prompt_id: "report_errors"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:backup_creator.tool_responses"
            as: "execution_results"
      toolset: [ "create_issue" ]

routers:
    - from: "backup_creator"
      condition:
          input: "context:backup_creator.execution_result"
          routes:
              "success": "next_step"
              "failed": "error_reporter"
    - from: "error_reporter"
      to: "end"  # Always end after error reporting
```

## Flow Examples

### Human-in-the-Loop Code Review Flow

This example demonstrates a flow that analyzes code, requests user approval, and takes action based on the user's
decision:

```yaml
version: "experimental"
environment: ide

components:
    - name: "code_analyzer"
      type: AgentComponent
      prompt_id: "code_analysis"
      prompt_version: "^1.0.0"
      inputs: [ "context:goal" ]
      toolset: [ "read_file", "list_dir", "find_files" ]
      ui_log_events: [ "on_agent_final_answer", "on_tool_execution_success" ]

    - name: "approval_request"
      type: HumanInputComponent
      sends_response_to: "code_executor"
      prompt_id: "approval_prompt"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:code_analyzer.final_answer"
            as: "analysis_results"
      ui_log_events: [ "on_user_input_prompt" ]

    - name: "code_executor"
      type: AgentComponent
      prompt_id: "code_execution"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:code_analyzer.final_answer"
            as: "analysis"
          - from: "context:approval_request.approval"
            as: "user_decision"
      toolset: [ "edit_file", "create_file_with_contents" ]
      ui_log_events: [ "on_agent_final_answer", "on_tool_execution_success" ]

routers:
    - from: "code_analyzer"
      to: "approval_request"
    - from: "approval_request"
      condition:
          input: "context:approval_request.approval"
          routes:
              "approve": "code_executor"
              "reject": "end"
              "default_route": "code_executor"
    - from: "code_executor"
      to: "end"

flow:
    entry_point: "code_analyzer"
```

### Interactive Chat Flow

This example shows a continuous conversation loop between user and agent:

```yaml
version: "experimental"
environment: ide

components:
    - name: "chat_agent"
      type: AgentComponent
      prompt_id: "chat_assistant"
      prompt_version: "^1.0.0"
      inputs: [ "context:goal" ]
      toolset: [ "read_file", "list_dir", "create_file_with_contents" ]
      ui_log_events: [ "on_agent_final_answer", "on_tool_execution_success" ]

    - name: "user_input"
      type: HumanInputComponent
      sends_response_to: "chat_agent"
      prompt_id: "continue_conversation"
      prompt_version: "^1.0.0"
      ui_log_events: [ "on_user_input_prompt" ]

routers:
    - from: "chat_agent"
      to: "user_input"
    - from: "user_input"
      to: "chat_agent"

flow:
    entry_point: "chat_agent"
```

### Conditional User Input Flow

This example demonstrates requesting user input only when the agent's confidence is low:

```yaml
version: "experimental"
environment: ide

components:
    - name: "decision_maker"
      type: AgentComponent
      prompt_id: "decision_analysis"
      prompt_version: "^1.0.0"
      inputs: [ "context:goal" ]
      toolset: [ "read_file", "find_files" ]
      ui_log_events: [ "on_agent_final_answer" ]

    - name: "user_clarification"
      type: HumanInputComponent
      sends_response_to: "final_processor"
      prompt_id: "clarification_request"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:decision_maker.final_answer"
            as: "initial_analysis"
      ui_log_events: [ "on_user_input_prompt" ]

    - name: "final_processor"
      type: AgentComponent
      prompt_id: "final_processing"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:decision_maker.final_answer"
            as: "analysis"
      toolset: [ "edit_file", "create_file_with_contents" ]
      ui_log_events: [ "on_agent_final_answer", "on_tool_execution_success" ]

routers:
    - from: "decision_maker"
      condition:
          input: "context:decision_maker.final_answer"
          routes:
              "needs_clarification": "user_clarification"
              "default_route": "final_processor"
    - from: "user_clarification"
      to: "final_processor"
    - from: "final_processor"
      to: "end"

flow:
    entry_point: "decision_maker"
```

### Data Analysis Flow

```yaml
version: experimental
environment: remote

components:
    # Component 1: Read and explore the data file
    - name: "file_reader"
      type: OneOffComponent
      prompt_id: "analysis/file_reader"
      prompt_version: "^1.0.0"
      max_correction_attempts: 3
      ui_log_events:
          - "on_tool_execution_success"
          - "on_tool_execution_failed"
      inputs:
          - from: "context:goal"
            as: "goal"
      toolset:
          - "read_file"
          - "list_dir"
          - "grep"

    # Component 2: Analyse the file content
    - name: "data_analyzer"
      type: OneOffComponent
      prompt_id: "analysis/data_processor"
      prompt_version: "^1.0.0"
      max_correction_attempts: 3
      ui_log_events:
          - "on_tool_execution_success"
          - "on_tool_execution_failed"
      inputs:
          - from: "context:file_reader.tool_responses"
            as: "data"
      toolset:
          - "grep"
          - "find_files"

    # Component 3: Generate analysis report
    - name: "report_generator"
      type: OneOffComponent
      prompt_id: "analysis/report_generator"
      prompt_version: "^1.0.0"
      max_correction_attempts: 3
      ui_log_events:
          - "on_tool_execution_success"
          - "on_tool_execution_failed"
      inputs:
          - from: "context:file_reader.tool_responses"
            as: "file_reader_results"
          - from: "context:data_analyzer.tool_responses"
            as: "data_analysis_results"
      toolset:
          - "create_file_with_contents"

routers:
    # Simple linear routing through the pipeline
    - from: "file_reader"
      to: "data_analyzer"
    - from: "data_analyzer"
      to: "report_generator"
    - from: "report_generator"
      to: "end"

flow:
    entry_point: "file_reader"
```

More examples will be added as the framework matures and additional use cases are identified.
