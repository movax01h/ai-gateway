# Flow Registry Framework Developer Guide

The Flow Registry is a component-based framework that enables developers to build and execute AI-powered flows using
YAML configurations.
This guide provides the practical knowledge developers need to build flows effectively.

[[_TOC_]]

## Versions

Flow Registry is being versioned in order to provide stable experience, and assure backwards compatibility.
At the current moment a stable version of Flow Registry is `v1`, it should be used for all other purposes
than experimental feature development.

|version|description|documentation| flow config location |
|-------|-----------|-------------|---------|
| `v1`  | The current stable version of Flow Registry | [documentation](v1.md) | [v1 flow configs](/duo_workflow_service/agent_platform/v1/flows/configs/) |
| `experimental` | Development version of Flow Registry, does not offer any backwards compatibility guarantees | [documentation](experimental.md) | [experimental flow configs](/duo_workflow_service/agent_platform/experimental/flows/configs/) |

Flow Registry configs declares used version via `version` attribute.

Each version is being implemented by an independent Python package in AI Gateway repository, flow configs
location must adhere to a declared version.

## Development Plan

Currently, we are in an ideation phase for Agent Flows. Please
see [this epic](https://gitlab.com/groups/gitlab-org/-/epics/19001) for instructions on how to submit your ideas for new
flows.

The Flow Registry is currently in active development, but the high-level plan for rollout is:

1. Add components required to replicate [existing flows](../workflows) using the Flow Registry. More information
   in [this epic](https://gitlab.com/groups/gitlab-org/duo-workflow/-/epics/1)
1. Create an AI Catalog UI to allow internal teams to build and test their own Agent Flows

In the meantime, it is possible to set up the Flow Registry locally, and create agent flows via YAML files. This guide
will explain the details of how to do that. But note that it can be quite time-intensive to do this setup and once the
Flow Registry is ready this setup will no longer be required for those building flows.

## Quick Start

For quick start guide please visit to _v1_ [documentation](v1.md#quick-start) page.

## Contribution Guidelines

If you which to contribute to Flow Registry framework please familiarize yourself with [Contribution Guidelines](./contribution_guidelines.md).

## Key Framework Concepts

A **Component** is a reusable building block that performs a specific task in your flow.
Each component declares its outputs, which will be available for subsequent components to read from. In addition each
component may request inputs, that should be sourced from outputs of proceeding components, or initial flow trigger
request data. Finally some components accept additional configuration parameters like  _tools_ or _prompt_id_ and
_prompt_version_. Consult component documentation to understand how it can be used, and what configuration it requires.
Components are stateless and compose together to create complex flows.

A **Router** determines the flow control between components.
Routers are either simple (always go to the next component) or conditional (route based on some condition).
They define the execution path through your flow.

Components and routers connect through the **Input/Output System**.
Components produce outputs that are automatically stored in the flow state, and these outputs are consumed by other
components through their input configuration.
Routers also use component outputs to make routing decisions, creating dynamic flow paths based on the results of
component execution.

## YAML Configuration Structure

YAML configuration files define the structure and behavior of your flows.
Every flow YAML file must contain these top-level sections that specify components, routing logic, and execution
parameters.

```yaml
version: "v1"
environment: ambient

components:
# List of components (see Component Types section)

routers:
# Define flow between components (see examples below)

flow:
    entry_point: "component_name"  # Name of first component to run
```

### Required Fields

- **version**: Declares Flow Registry version used by a config
- **environment**: Controls level of interaction between a human and AI agents
- **components**: List of components that make up your flow
- **routers**: Define how components connect to each other
- **flow**: Specify the entry point component and other options
- **prompts**: List of inline prompt templates for flow components to use

### Optional fields

- **name**: User-readable name for the flow
- **description**: Description of the flow
- **product_group**: Attributes team ownership of flow (e.g. `agent_foundations`)

## Input/Output System

The Input/Output System is the core mechanism that enables data flow between components in your flow.
This system manages all data passing between components and routers through a structured shared state, ensuring seamless
communication throughout the workflow execution.

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

- **Simplifying prompt templates**: Instead of referencing `{{context:analyzer.final_answer}}` in your prompt, you use
  `{{analysis_results}}`
- **Making flows more readable**: Clear, descriptive names improve flow understanding
- **Reducing coupling**: Components don't need to know internal structure of other components' outputs

### Input Literals

Input literals can be used to explicitly state values for inputs, by adding `literal: true`. When using input literals,
the `as` keyword is required:

```yaml
inputs:
    - from: "file.txt"
      as: "file_path"
      literal: true
```

This will set the value of the input variable `file_path` to be `file.txt`, rather than interpreting the input source as
a path.

### Optional Inputs and Default Values

By default, all inputs are required and the flow will fail if the specified path doesn't exist in the state. You can
make inputs optional:

**Optional inputs** allow components to gracefully handle missing data:

```yaml
inputs:
    - from: "context:inputs.user_preferences.theme"
      as: "theme"
      optional: true
```

When `optional: true` is set:

- If the path exists in the state, its value is used
- If the path doesn't exist, the value will be `None`
- No error is raised for missing data

**Example with multiple optional inputs**:

```yaml
components:
    - name: "report_generator"
      type: AgentComponent
      prompt_id: "generate_report"
      inputs:
          - from: "context:goal"
            as: "task"
          - from: "context:analysis.findings"
            as: "findings"
            optional: true
```

In this example:

- `goal` is required and will fail if missing
- `findings` will be set to None if the analysis component didn't run

### Additional Context

Additional Context can be passed to the Flow (in additional to the `goal`), but the schema for these fields must be defined in the `flow` section of the YAML file:

```yaml
flow:
  entry_point: "first_component_name"
  inputs:
    - category: merge_request_info
      input_schema:
        url:
          type: string
          format: uri
          description: GitLab Merge Request URL
        source_branch:
          type: string
          description: Merge Request Source Branch
    - category: pipeline_info
      input_schema:
        url:
          type: string
          format: uri
          description: GitLab Pipeline URL
```

The `format` and `description` fields are optional in the definitions. Allowed `type` and `format` fields can be found in the [jsonschema](https://json-schema.org/docs) docs.

When making the call to the Service API, these additional context parameters are passed in as serialized JSON in the `Content` fields:

```json
"additional_context": [
    {"Category": "merge_request_info", "Content": "{\"url\": \"www.example.com\", \"source_branch\": \"testbranch\"}"},
    {"Category": "pipeline_info", "Content": "{\"url\": \"www.example.com\"}"}
]
```

### Output

Output management handles the automatic production and storage of component results.
Each component automatically produces outputs that may be consumed by other components.
The outputs are automatically available to subsequent components when they are referenced in their `inputs`
configuration or used in router conditions.
Refer to the documentation of every component to check what outputs it produces.

> **Note:** Tool responses have automatic truncation applied. See [Tool Development Guide](../adding_new_tool.md#2-️-tool-response-truncation) for details.

## UI Logs

UI logs provide visibility into component execution progress and results for end users.
This system captures and displays component activities in the user interface, enabling real-time feedback about workflow
execution.

Each component defines its own set of available events to be logged and visualized.
For example, the AgentComponent provides events for tool execution and final responses, allowing users to see exactly
what the agent is doing and what results it produces.

You need to specify what events to log for every component that provides UI feedback.
The events you choose determine what information users see about the component's execution in the Duo interface.

## Routers

Routers control the flow execution path between components.
These components determine how workflow execution moves from one step to the next, enabling both linear and conditional
flow patterns.

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

This router examines the `final_answer` output from the "decision_maker" component and routes to different components
based on the content:

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

This router uses the workflow status to determine the next step in processing, allowing for sophisticated error handling
and partial result processing.

## Component Types

To see complete list of available components see the latest Flow Registry version documentation [page](v1.md#component-types)

## Debugging Flows

To run and debug your flow within your local GDK:

- Set up a
  local [Agent Platform](https://gitlab.com/gitlab-org/gitlab-development-kit/-/blob/main/doc/howto/duo_agent_platform.md)
  to work with Remote Flows
- Create flow, note the name and version (in the form `<flow_config_file_name_without_extension>/<version>` eg: `prototype/experimental`)
- Run this `curl` command to start your flow:

```shell
export DEFINITION="prototype/experimental"
export GOAL="create test.sh script that outputs done to stdout"
export PROJECT_ID="19"

curl -X POST \
    -H "Authorization: Bearer $GDK_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
        \"project_id\": \"$PROJECT_ID\",
        \"agent_privileges\": [1,2,3,4,5],
        \"goal\": \"$GOAL\",
        \"start_workflow\": true,
        \"workflow_definition\": \"$DEFINITION\",
        \"environment\": \"web\",
        \"source_branch\": \"branch-name-to-run-in\"
    }" \
    http://gdk.test:3000/api/v4/ai/duo_workflows/workflows
```

- You can check your flow execution in your rails instance, under Pipelines
- You can confirm the flow instance in your Rails DB: `Ai::DuoWorkflows::Workflow.last`
- Examine Agent Platform logs with `gdk tail duo-workflow-service`
- Trace your flow's execution
  in [Langsmith](https://docs.gitlab.com/development/ai_features/duo_chat/#use-tracing-with-langsmith)
- Alternatively, if you are running Duo Workflow Service individually using PyCharm run configuration,
  you can use [debugging flow execution using AI Agents debugger plugin](#debugging-flow-execution-using-ai-agents-debugger-plugin)
- Or view the latest flow session state snapshot via API `curl -X GET -H "X-Per-Page: 1" -H "X-Page: 1" -H "Authorization: Bearer $GDK_API_PRIVATE_TOKEN" 'https://gdk.test:3000/api/v4/ai/duo_workflows/workflows/1139603/checkpoints?page=1&per_page=1' > .debug_checkpoint`

### Debugging flow execution using AI Agents Debugger plugin

- Install [AI Agents Debugger](https://plugins.jetbrains.com/plugin/26921-ai-agents-debugger) plugin in PyCharm.
- Setup PyCharm run configuration to run Duo Workflow server.
- Run any flow that invokes Duo Workflow service server.
- When the request reaches the server, the plugin analyzes the code execution and automatically attaches itself to the
  LangGraph workflow and provides detailed insights into the LangGraph agent nodes, metadata, inputs, and outputs of
  each node.
- You can analyze time spent in each node's execution, system prompts, human input and AI response, tool calls and
  output within the IDE.

### Running Flows with Duo CLI Headless Mode

For a simpler alternative to the full GDK + Agent Platform setup, you can use the Duo CLI in headless mode to run and test flows directly from the command line.

#### Prerequisites

- Install the [Duo CLI](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/tree/main/packages/cli?ref_type=heads)

#### Running a Flow

Use the `duo run` command with the `--flow-config` flag pointing to your flow YAML file:

```shell
duo run \
  --flow-config duo_workflow_service/agent_platform/v1/flows/configs/your_flow.yml \
  --flow-config-schema-version v1 \
  -g "Your goal description here"
```

#### Passing Additional Context

If your flow defines `inputs` in the `flow` section, pass them via the `DUO_WORKFLOW_ADDITIONAL_CONTEXT_CONTENT` environment variable as a JSON array:

```shell
export DUO_WORKFLOW_ADDITIONAL_CONTEXT_CONTENT='[{"Category":"your_category","Content":"{\"key\":\"value\",\"another_key\":\"another_value\"}"}]'
```

```shell
duo run \
  --flow-config duo_workflow_service/agent_platform/v1/flows/configs/your_flow.yml \
  --flow-config-schema-version v1 \
  -g "Your goal description here"
```

The `Category` must match a category defined in your flow's `flow.inputs` section, and the `Content` must be a JSON-serialized string matching the `input_schema`.

#### When to Use Headless Mode

This method is particularly useful for:

- Quick iteration on flow configurations
- Testing flow logic without full infrastructure
- Debugging specific flow components in isolation
- Rapid prototyping of new flows

For more advanced options, including a Docker wrapper for reproducible testing, see the [Duo CLI headless mode documentation](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/tree/main/packages/cli?ref_type=heads#headless-mode).
