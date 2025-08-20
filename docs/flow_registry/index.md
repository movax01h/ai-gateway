# Flow Registry Framework Developer Guide

The Flow Registry is a component-based framework that enables developers to build and execute AI-powered flows using YAML configurations.
This guide provides the practical knowledge developers need to build flows effectively.

[[_TOC_]]

## Quick Start

This section provides a step-by-step approach to creating your first flow.
Follow these steps to build a basic AI agent flow that can interact with the repository files and respond to user requests.

1. Create a YAML file in `duo_workflow_service/agent_platform/experimental/flows/configs/` which will configure your flow. A name of the file will become flow identifier used to trigger it later on. The file should has this basic structure:

   ```yaml
   version: "experimental"
   environment: remote
   components:
      - name: "my_agent"
        type: AgentComponent
        prompt_id: "your_prompt_id"
        prompt_version: "^1.0.0"
        inputs: ["context:goal"]
        toolset: ["read_file", "create_file_with_contents"]
   routers:
      - from: "my_agent"
        to: "end"
   flow:
      entry_point: "my_agent"
   ```

1. Create a prompt template in the AI Gateway prompt registry at `ai_gateway/prompts/definitions/your_prompt_id/base/1.0.0.yml`:

   ```yaml
   name: Your prompt name
   model:
      config_file: claude_4_0
      params:
         max_tokens: 8_192
      unit_primitives: []
   prompt_template:
      system: |
        You are GitLab Duo Chat, an agentic AI Coding assistant built by GitLab.
        Your role is to help the user complete their request by using the available tools.
        Your response style is concise and actionable.
      user: |
        Here is my task:
        {{goal}}
      placeholder: history
   params:
      timeout: 30
   ```

1. Accessing the new Flow

   Currently, accessing flows via gRPC is not fully implemented.
   To use your new flow in the Duo Chat interface within VSCode, follow this workaround:

   1. In the workflow registry file `duo_workflow_service/server.py`, uncomment the line:

      ```python
      workflow_class: FlowFactory = resolve_workflow_class("prototype/experimental")
      ```

   1. After making this change, your new flow will be available in the Duo Chat interface for interaction in VSCode

## Key Framework Concepts

A **Component** is a reusable building block that performs a specific task in your flow.
Each component declares its outputs, which will be available for subsequent components to read from. In addition each component may request inputs, that should be sourced from outputs of proceeding components, or initial flow trigger request data. Finally some components accept additional configuration parameters like  _tools_ or _prompt_id_ and _prompt_version_. Consult component documentation to understand how it can be used, and what configuration it requires.
Components are stateless and compose together to create complex flows.

A **Router** determines the flow control between components.
Routers are either simple (always go to the next component) or conditional (route based on some condition).
They define the execution path through your flow.

Components and routers connect through the **Input/Output System**.
Components produce outputs that are automatically stored in the flow state, and these outputs are consumed by other components through their input configuration.
Routers also use component outputs to make routing decisions, creating dynamic flow paths based on the results of component execution.

## YAML Configuration Structure

YAML configuration files define the structure and behavior of your flows.
Every flow YAML file must contain these top-level sections that specify components, routing logic, and execution parameters.

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
- **environment**: Set to `"remote"` for delegated tasks that agents should do in the background with little to non user interactions or `"local"` for pair coding experience, when user is expected to collaborate with agents in real time.
- **components**: List of components that make up your flow
- **routers**: Define how components connect to each other
- **flow**: Specify the entry point component and other options

## Input/Output System

The Input/Output System is the core mechanism that enables data flow between components in your flow.
This system manages all data passing between components and routers through a structured shared state, ensuring seamless communication throughout the workflow execution.

### Understanding Flow State

Components and routers communicate through a shared state with these fields:

- **context**: Nested dictionary for data exchange
- **conversation_history**: Component-specific message history
- **status**: Current workflow status

### Input Syntax

Inputs use the format `target:path.to.data`:

```yaml
inputs:
  - "context:goal"                        # Access goal from context
  - "context:previous_agent.final_answer" # Access result from previous component
  - "status"                              # Access workflow status
```

Input paths must reference existing data in the flow state.
If a component tries to access non-existent data, the flow will fail.

### Input Aliases

Input aliases provide cleaner data mapping and improved readability:

```yaml
inputs:
  - from: "context:source_data"
    as: "input_data"
  - from: "context:analyzer.final_answer"
    as: "analysis_results"
```

The `as` keyword provides these benefits:

- **Simplifying prompt templates**: Instead of referencing `{{context:analyzer.final_answer}}` in your prompt, you use `{{analysis_results}}`
- **Making flows more readable**: Clear, descriptive names improve flow understanding
- **Reducing coupling**: Components don't need to know internal structure of other components' outputs

### Output

Output management handles the automatic production and storage of component results.
Each component automatically produces outputs that may be consumed by other components.
The outputs are automatically available to subsequent components when they are referenced in their `inputs` configuration or used in router conditions.
Refer to the documentation of every component to check what outputs it produces.

## UI Logs

UI logs provide visibility into component execution progress and results for end users.
This system captures and displays component activities in the user interface, enabling real-time feedback about workflow execution.

Each component defines its own set of available events to be logged and visualized.
For example, the AgentComponent provides events for tool execution and final responses, allowing users to see exactly what the agent is doing and what results it produces.

You need to specify what events to log for every component that provides UI feedback.
The events you choose determine what information users see about the component's execution in the Duo interface.

## Routers

Routers control the flow execution path between components.
These components determine how workflow execution moves from one step to the next, enabling both linear and conditional flow patterns.

### Simple Router

A simple router always routes from one component to another without any conditions:

```yaml
routers:
  - from: "analyzer"
    to: "processor"
```

This router will always send the flow from the "analyzer" component to the "processor" component.

### Conditional Router

A conditional router makes routing decisions based on component outputs or flow state:

```yaml
routers:
  - from: "decision_maker"
    condition:
      input: "context:decision_maker.final_answer"
      routes:
        "approve": "approval_handler"
        "reject": "rejection_handler"
        "default_route": "manual_review"
```

This router examines the `final_answer` output from the "decision_maker" component and routes to different components based on the content:

- If the answer equals to "approve", route to the "approval_handler" component
- If the answer equals to "reject", route to the "rejection_handler" component
- For any other content, route to the "manual_review" component (default_route)

### Complex Multi-Path Router

```yaml
routers:
  - from: "file_analyzer"
    condition:
      input: "status"
      routes:
        "completed": "report_generator"
        "error": "error_handler"
        "partial": "manual_processor"
        "default_route": "end"
```

This router uses the workflow status to determine the next step in processing, allowing for sophisticated error handling and partial result processing.

## Component Types

### AgentComponent

The AgentComponent is the primary building block for AI-powered flows.
An AgentComponent uses a Large Language Model (LLM) to process inputs and generate responses. It provides these capabilities:

- Execute tools to interact with the environment (read files, run commands, etc.)
- Maintain conversation history for context
- Make decisions based on the provided prompt and available tools
- Generate structured outputs that other components consume

The agent uses the prompt template from the prompt registry, processes the inputs through the LLM, and calls tools as needed to complete its task.

### Required Parameters

- **name**: Unique identifier for this component instance. Must not contain `:` or `.` characters.
- **type**: Must be `"AgentComponent"`
- **prompt_id**: ID of the prompt template from the prompt registry
- **prompt_version**: Semantic version constraint (e.g., `"^1.0.0"`)

### Optional Parameters

- **inputs**: List of input data sources (default: `["context:goal"]`)
- **toolset**: List of tools available to the agent
- **ui_log_events**: UI logging configuration
- **ui_role_as**: Display role in UI (`"agent"` or `"tool"`)

### Available Tools

Agents access the following tools in their `toolset` configuration.
Complete list of tools classes can be located at `duo_workflow_service/components/tools_registry.py`. To configure an agent with tools, pass `name` attributes from their classes, eg: for `GetIssue` tool class pass `get_issue`
Each tool is a Python class that the agent calls to perform specific actions.

Here are some examples:

- **read_file**: Read contents of a file
- **create_file_with_contents**: Create a new file with specified content
- **edit_file**: Modify an existing file
- **list_dir**: List directory contents
- **find_files**: Search for files matching patterns

### Prompts

Prompts define how the AI agent behaves and responds to inputs.
Every AgentComponent requires a prompt that serves as the "instructions" for the AI agent, defining its personality, capabilities, and response patterns.
Prompts include placeholders that get replaced with actual data from the flow state.

### Inputs

Agent inputs work together with prompt placeholders to provide dynamic data to the AI.
When you define inputs in your component configuration, that data becomes available as template variables in your prompt.

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

### Outputs

Each AgentComponent automatically produces:

- **context:{component_name}.final_answer**: The agent's final response
- **conversation_history:{component_name}**: Message history
- **status**: Updated workflow status

These outputs can be used as inputs by other components or referenced in routing logic to control flow execution.
For example, you might route to different components based on whether the agent's final answer equals to specific phrase

### UI Log Events

The AgentComponent supports the following UI log events that can be specified in the `ui_log_events` configuration:

- **on_agent_final_answer**: Logged when the agent produces its final response. This event captures the agent's conclusion or final output and displays it in the UI.
- **on_tool_execution_success**: Logged when a tool is successfully executed by the agent. This shows users what actions the agent took and their successful results.
- **on_tool_execution_failed**: Logged when a tool execution fails. This provides error information and helps with debugging failed operations.

#### Complete AgentComponent Example

```yaml
components:
  - name: "code_assistant"
    type: AgentComponent
    prompt_id: "code_review_helper"
    prompt_version: "^1.0.0"
    inputs: ["context:goal"]
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

The HumanInputComponent enables human-in-the-loop interactions within flows by requesting and processing user input during workflow execution.
This component allows workflows to pause execution, request user feedback or decisions, and then continue based on the user's response.

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
- **REJECT**: User rejection decision that gets stored in the context as `"reject"`, optionally with a message added to conversation history

#### Outputs

Each HumanInputComponent automatically produces:

- **conversation_history:{sends_response_to}**: User messages directed to the target component
- **context:{component_name}.approval**: User approval decision (`"approve"` or `"reject"`)

#### UI Log Events

The HumanInputComponent supports the following UI log event:

- **on_user_input_prompt**: Logged when displaying a prompt to request user input. This shows the prompt content in the UI to guide the user's response.

#### Environment Support

The HumanInputComponent is only supported in the `"ide"` environment, as it requires the interrupt mechanism for pausing workflow execution.

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
    toolset: ["read_file", "edit_file"]
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

## Flow Examples

### Human-in-the-Loop Code Review Flow

This example demonstrates a flow that analyzes code, requests user approval, and takes action based on the user's decision:

```yaml
version: "experimental"
environment: ide

components:
  - name: "code_analyzer"
    type: AgentComponent
    prompt_id: "code_analysis"
    prompt_version: "^1.0.0"
    inputs: ["context:goal"]
    toolset: ["read_file", "list_dir", "find_files"]
    ui_log_events: ["on_agent_final_answer", "on_tool_execution_success"]

  - name: "approval_request"
    type: HumanInputComponent
    sends_response_to: "code_executor"
    prompt_id: "approval_prompt"
    prompt_version: "^1.0.0"
    inputs:
      - from: "context:code_analyzer.final_answer"
        as: "analysis_results"
    ui_log_events: ["on_user_input_prompt"]

  - name: "code_executor"
    type: AgentComponent
    prompt_id: "code_execution"
    prompt_version: "^1.0.0"
    inputs:
      - from: "context:code_analyzer.final_answer"
        as: "analysis"
      - from: "context:approval_request.approval"
        as: "user_decision"
    toolset: ["edit_file", "create_file_with_contents"]
    ui_log_events: ["on_agent_final_answer", "on_tool_execution_success"]

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
    inputs: ["context:goal"]
    toolset: ["read_file", "list_dir", "create_file_with_contents"]
    ui_log_events: ["on_agent_final_answer", "on_tool_execution_success"]

  - name: "user_input"
    type: HumanInputComponent
    sends_response_to: "chat_agent"
    prompt_id: "continue_conversation"
    prompt_version: "^1.0.0"
    ui_log_events: ["on_user_input_prompt"]

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
    inputs: ["context:goal"]
    toolset: ["read_file", "find_files"]
    ui_log_events: ["on_agent_final_answer"]

  - name: "user_clarification"
    type: HumanInputComponent
    sends_response_to: "final_processor"
    prompt_id: "clarification_request"
    prompt_version: "^1.0.0"
    inputs:
      - from: "context:decision_maker.final_answer"
        as: "initial_analysis"
    ui_log_events: ["on_user_input_prompt"]

  - name: "final_processor"
    type: AgentComponent
    prompt_id: "final_processing"
    prompt_version: "^1.0.0"
    inputs:
      - from: "context:decision_maker.final_answer"
        as: "analysis"
    toolset: ["edit_file", "create_file_with_contents"]
    ui_log_events: ["on_agent_final_answer", "on_tool_execution_success"]

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

More examples will be added as the framework matures and additional use cases are identified.
