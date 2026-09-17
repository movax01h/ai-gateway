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
- **prompts**: List of inline prompt templates for flow components to use

### Optional fields

- **name**: User-readable name for the flow
- **description**: Description of the flow
- **product_group**: Attributes team ownership of flow (e.g. `agent_foundations`)

## IOKey Abstractions

The experimental version of Flow Registry extends the IOKey hierarchy with a third resolution tier.
For the foundational `IOKey` and `IOKeyTemplate` abstractions, see the
[Contribution Guidelines](contribution_guidelines.md#iokey-and-iokeytemplate-abstraction).

| Abstraction     | Resolved at                      | How                            |
|-----------------|----------------------------------|--------------------------------|
| `IOKey`         | Graph-build time (static)        | Direct field values            |
| `IOKeyTemplate` | Graph-build time (parameterised) | `to_iokey(replacements: dict)` |
| `RuntimeIOKey`  | Graph-execution time (runtime)   | `to_iokey(state: FlowState)`   |

### RuntimeIOKey

`RuntimeIOKey` is an `IOKey` subclass whose concrete identity is resolved at graph-execution time.
It is used when the concrete state path (target and subkeys) depends on runtime state — for example, a
subsession-scoped output key whose subsession ID is only known once the graph is running.

#### Key Properties

- **`alias`** (required): The statically-declared Jinja2 template variable name. This allows prompt-input validators
  to check that every template variable has a corresponding input key without executing the graph.
- **`factory`**: A callable `(state: FlowState) -> IOKey` that resolves the concrete key at runtime.
- **`target`**: Raises `RuntimeError` when accessed directly. The concrete target is only known at runtime;
  use `to_iokey(state).target` to obtain it after resolution.

#### Methods

- **`to_iokey(state: FlowState) -> IOKey`**: Resolves the concrete `IOKey` by calling `factory(state)`.
- **`to_nested_dict(value, state: FlowState) -> dict`**: Resolves the concrete `IOKey` at runtime and delegates
  to `IOKey.to_nested_dict(value)`. Requires `state` (unlike the parent's `to_nested_dict(value)`) because the
  target and subkeys are only known after resolution.
- **`value_from_state(state)`**: Resolves the key and reads its value from state.
- **`template_variable_from_state(state)`**: Resolves the key and returns its template variable dict.

#### Construction

```python
from duo_workflow_service.agent_platform.experimental.state.base import (
    IOKey,
    IOKeyTemplate,
    RuntimeIOKey,
)

# Wrap a static IOKey for uniform handling in nodes that accept RuntimeIOKey
static_key = RuntimeIOKey(
    alias="final_answer",
    factory=lambda _: IOKey(target="context", subkeys=["my_agent", "final_answer"]),
)

# Resolve a dynamic key based on runtime state (e.g. active subsession ID)
template = IOKeyTemplate(
    target="context",
    subkeys=[IOKeyTemplate.SUBAGENT_NAME_TEMPLATE, IOKeyTemplate.SUBSESSION_ID_TEMPLATE, "final_answer"],
)
dynamic_key = RuntimeIOKey(
    alias="final_answer",
    factory=lambda state: template.to_iokey({
        IOKeyTemplate.SUBAGENT_NAME_TEMPLATE: state["context"]["active_subagent"],
        IOKeyTemplate.SUBSESSION_ID_TEMPLATE: str(state["context"]["active_subsession"]),
    }),
)
```

#### Usage in Nodes

Nodes that write to state accept `RuntimeIOKey` for their output and conversation-history parameters.
Call `to_iokey(state)` to obtain the resolved `IOKey`, then use its methods:

```python
async def run(self, state: FlowState) -> dict:
    output_iokey = self._output_key.to_iokey(state)
    return output_iokey.to_nested_dict(result_value)
```

Or use `to_nested_dict` directly on the `RuntimeIOKey` when you have the state available:

```python
async def run(self, state: FlowState) -> dict:
    return self._output_key.to_nested_dict(result_value, state)
```

#### Design Rationale

`RuntimeIOKey` prevents accidental misuse of build-time IOKey APIs on keys whose target is not yet known:

- Accessing `.target` directly raises `RuntimeError` to make the mistake immediately visible.
- `to_nested_dict` requires `state` as an explicit parameter, making the runtime dependency clear in the call site.
- The `alias` field provides a stable, statically-known name for prompt-input validation without graph execution.

## Component Types

### AgentComponent

The `AgentComponent` (including supervisor mode) has been promoted to the stable v1 version.
See the [AgentComponent documentation in v1.md](v1.md#agentcomponent) for full details.

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

- **message_template**: Jinja2 template to be used to render a message when requesting user input
- **inputs**: List of input data sources for template rendering (default: empty list)
- **ui_log_events**: UI logging configuration for displaying messages

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
      message_template: "Confirm if you want to proceed with {{ proposed_changes }}"
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

The DeterministicStepComponent executes a **single tool** deterministically with predetermined arguments extracted from
the flow state. This component provides a way to run one specific tool without AI involvement, using inputs to extract
the necessary parameters and producing predictable outputs following fixed conventions.

The component provides these capabilities:

- **Execute a single tool deterministically**: Run one designated tool with parameters derived from inputs
- **Extract parameters from state**: Use component inputs to gather the tool's execution arguments
- **No AI involvement**: Direct tool execution without LLM processing
- **Integration with existing toolsets**: Compatible with any registered tool in the toolset
- **Chainable design**: Multiple DeterministicStepComponents can be chained to execute sequential tool operations

Unlike AgentComponent or OneOffComponent which use AI to determine tool usage, DeterministicStepComponent executes
exactly one pre-specified tool with arguments extracted directly from the flow state, making it ideal for predictable,
repeatable operations. **To execute multiple tools, chain multiple DeterministicStepComponents together in your flow.**

#### Required Parameters

- **name**: Unique identifier for this component instance. Must not contain `:` or `.` characters.
- **type**: Must be `"DeterministicStepComponent"`
- **tool_name**: Name of the single tool to execute

#### Optional Parameters

- **toolset**: Toolset containing the tool to be executed. (If no toolset is specified, a new one is created with only
  the `tool_name`)
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
- **compaction**: Configuration for conversation compaction. Useful when OneOffComponent is used in flows with prior
  conversation history. See [Conversation Compaction](../context_management/compaction.md) for details.
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

---

## Component Attributes

A component attribute is not a component type of its own: it is a field an existing component declares to change
how its own body runs. Attributes are documented here rather than under [Component Types](#component-types)
because any component type accepts them.

### for_each

`for_each` runs a component's body once per item of a list and collects what each run produced back into the flow.
The list decides how many runs happen, so the fan-out is fixed by data rather than chosen by a model.

Reach for it when an earlier component has already produced a list of independent work items — files to review,
findings to triage, directories to scan — and each item should be handled by its own fresh run of the same
component.

`for_each` wraps a component instead of changing it, so adding it does not change which component runs, only how
many times. Any component type can be wrapped, and the component it wraps is configured exactly as it would be on
its own:

```yaml
- name: "review_one"
  type: AgentComponent
  # ... the component's own configuration is unchanged ...
  for_each:
      items: "context:discover.final_answer.files"
      as: "context:item"
```

#### Feature flag

The fan-out is gated on the `dap_for_each` feature flag. While the flag is off a `for_each` block is ignored and a
warning is logged: the flow still loads, and the component runs once over the flow's own state. Enable the flag
for the instance or group a flow runs in before that flow depends on fanning out.

#### Configuration

| Field | Required | Default | Description |
|---|---|---|---|
| `items` | yes | — | Path to the list to iterate over, written like any other input path, for example `context:discover.final_answer.files`. It must resolve to a list when the component runs; anything else fails the flow with a `TypeError` naming the component, this field and the value it found. An empty list is not an error: nothing runs, `results` is published empty, and the flow carries on. An agent's `final_answer` is a list only when the agent declares a response schema with an array field, so point `items` at that field. |
| `as` | yes | — | Path under which one item is published to the body for one run, for example `context:item`. Unlike an input path this one is written to, so it has to be a `context:` path with at least one subkey; while the flag is on, the flow is rejected at load time otherwise. It also may not write into the namespace `items` reads from, because each item would overwrite the list being iterated. Give the item a namespace of its own. |
| `max_items` | no | `1000` | How many items are run at all. Must be between `1` and `1000`, and it defaults to that ceiling, so nothing is dropped unless a flow asks for it. **A longer list is not an error: the items past `max_items` are never run, and the flow still succeeds.** The list's own order decides which survive, so order the list upstream to put what matters first. The run publishes `total_items` and `truncated` either way, so a later component can tell what was left out; the service also logs a warning naming the component and both counts. |
| `max_concurrency` | no | `10` | How many of this component's branches run at once. Must be between `1` and `100`. The cap belongs to this one fan-out: each wrapped component has a gate of its own, and it bounds neither the rest of the flow nor the model concurrency of the fleet. Raising it trades a larger burst of concurrent model calls for a shorter wall-clock run. An approval inside a capped fan-out is reached after about `total items / max_concurrency` rounds of the body rather than one round, which is true of any cap. |

These are all the fields there are: while the flag is on, a key that is not one of them is rejected when the flow
is loaded, and the message names it. A mistyped `max_items` therefore fails rather than running on the default.
The block is only validated once the flag has been checked, so while the flag is off it is ignored whatever it
contains.

#### How a fan-out runs

A fanned-out component is three nodes rather than one, named after the component the same way every other node is:

| Node | What it does |
|---|---|
| `<name>#foreach_dispatch` | Reads `items`, applies `max_items`, and starts one branch per item |
| `<name>#foreach_unit` | Runs the whole wrapped body over one item |
| `<name>#foreach_collect` | Fans the branches back in, and is where the flow's own router takes over |

Branches are started as LangGraph `Send` tasks from the dispatch node's own state update, so the branches and the
slot they write into are published together. The edge from the unit node to the collect node is static, which is
what lets LangGraph collapse however many branches ran into a single collect step.

Routing is unaffected: a router that names the component enters the dispatch node and leaves from the collect
node, exactly as it would enter and leave the component's own body.

#### Where each item lands

Each branch is invoked with a `FlowState` of its own, built from scratch rather than copied from the flow:

- `conversation_history`, `ui_chat_log` and `agent_context_limits` start empty, so a branch neither reads nor
  re-emits what the flow accumulated before it.
- `context` is a one-level copy of the flow's own, with this item written at the path named in `as`. So with
  `as: "context:item"`, the body reads `context:item`, and a prompt template refers to the item by the last part
  of that path. `context:goal` still holds the flow's own user prompt, unchanged, in every branch.
- The component's own `context:<name>` namespace is left out. That is where the fan-out publishes its results,
  and every branch payload is checkpointed, so carrying it would make checkpoints grow with the square of the
  item count.
- `context:<name>.for_each_index` carries the item's position in the list. It is scoped to the
  component, so a fan-out nested inside another one cannot overwrite the outer one's index.

A branch therefore sees what the flow had produced before the fan-out began, and nothing of what its siblings are
doing.

#### Collected outputs

| Key | Description |
|---|---|
| `context:<name>.results` | A list with one entry per item that ran, **in the order of the input list**. An entry holds the wrapped component's own `context:<name>` namespace as that branch left it — for a `DeterministicStepComponent` named `read_one` that is `{tool_responses, error, execution_result}`, and for an `AgentComponent` `{final_answer}` plus any response-schema fields. An item that failed holds an error record instead, in the same position: see [When an item fails](#when-an-item-fails). |
| `context:<name>.errors` | A list of `{index, type, message}`, in input order, one per failed item. The aggregate view of the same failures the `results` entries carry, so a router condition can test `errors` without walking `results`. |
| `context:<name>.total_items` | How long the `items` list was, before `max_items` was applied. |
| `context:<name>.processed_items` | How many items actually ran, which is `len(results)`. |
| `context:<name>.succeeded` | How many of those produced a result. |
| `context:<name>.failed` | How many recorded an error, which is `len(errors)`. |
| `context:<name>.truncated` | `true` when `max_items` cut the list short, so `processed_items` is less than `total_items`. |
| `ui_chat_log` | What each branch showed the user, appended to the flow's own log. A failed item contributes nothing here. |

Branches finish in whatever order they finish, and each writes an index-keyed scratch slot of its own so that
concurrent writes never collide. The collect node is the barrier where every slot exists, so that is where the
scratch is turned into the list above and then cleared. A template iterating `results` therefore walks the items
in the order they were given, and `results[2]` is the third item's outcome.

The counts are what let a reader judge the list it got. `total_items` and `truncated` are published by the
dispatch node, before any branch runs, so a truncated fan-out says so in the state rather than only in a service
log.

`ui_chat_log` is the exception, and it is worth knowing before fanning out anything chatty: it is an append-only
channel fed by concurrent branches, so entries arrive in **completion order, not item order**. Two branches'
entries can also interleave. Do not read the log to tell what a particular item did — read that item's own entry
in `results`.

Nothing else a branch produced is published. In particular each branch's `conversation_history` stays inside the
branch, so a fanned-out agent's per-item transcript is not available to the rest of the flow.

#### What a branch may write

The flow's state has no automatic isolation for what a branch returns, so what siblings may safely write at the
same time is limited by the reducer behind each channel:

- Append-reduced channels, `ui_chat_log` above all, accept concurrent writes and keep every one of them.
- Distinct leaf keys of `context` are safe, because `context` is merged key by key. This is why the branches
  write an index-keyed scratch slot: each branch writes a leaf nothing else writes.
- Anything else two branches write at the same time is last-write-wins, with no error and no warning. That
  includes `status`, `conversation_history`, and any single `context` leaf two branches both write.

Design the wrapped component's body within that: give each branch its own leaf, and let the fan-out assemble the
whole.

#### When an item fails

A branch that raises does not take the fan-out down with it. The failure is caught, recorded in that item's own
`results` slot, and every other branch runs to completion:

```yaml
"context:review_one.results":
    - final_answer: "looks fine"
    - for_each_error:
          type: "ToolException"
          message: "read_file: no such path 'src/gone.py'"
    - final_answer: "needs a test"
"context:review_one.errors":
    - index: 1
      type: "ToolException"
      message: "read_file: no such path 'src/gone.py'"
```

Containment is unconditional, and there is nothing to configure. `results` therefore always holds one entry per
item that ran, success or error, so `len(results)` is still the number of branches. An entry is a failure exactly
when it carries a `for_each_error` key, and `errors` is the same set of failures collected in one place with the
item's position attached.

That key is namespaced on purpose. A component may publish an `error` key of its own — `DeterministicStepComponent`
does — so a bare `error` could not distinguish a branch that failed from a branch that ran fine and recorded a
tool error.

**A stage that reads `results` has to handle error entries.** Nothing strips them out before the next component
sees them, so a reader that assumes every entry holds the wrapped component's own keys raises a `KeyError` on the
first failed item. Test for `for_each_error` first, and read the rest only when it is absent.

A failed item also contributes no `ui_chat_log` entries. The branch's state never came back, so whatever its body
showed the user before raising is gone: the user sees the failure, but not the steps leading to it.

##### Failures that stay terminal

Some failures are not the item's to answer for. They report that something is wrong with the flow, and every
sibling is about to hit the same thing, so they propagate and fail the flow with no error record written:

| Exception | Why it is not one item's problem |
|---|---|
| `GraphBubbleUp` | Not a failure. It carries `interrupt()` and a parent-targeted `Command`, so it is LangGraph's own control flow and has to reach LangGraph. |
| `GraphRecursionError` | The run reached the step limit, which the flow reports to the user as such. |
| `InvalidRequestException` | The request itself is rejected, and the flow deliberately stays out of `FAILED` so the caller can correct it. |
| `ModelError` | The provider failed after its retries ran out, or failed non-retryably: authentication, permission, invalid request, context too large. |
| `NotifiableAgentException`, `NotifiableException` | Both carry a message meant for the user, which an error record buries. |
| `SecurityException` | Prompt injection and its relatives. The result must never reach a model again. |
| `InsufficientCredits`, `UsageQuotaError` | Out of credits or entitlements, or the quota check is unavailable. Every sibling fails the same way. |

##### Every item failing is a bug, not a result

When *every* item fails, the fan-out raises `AllItemsFailedError` instead of publishing results. A fault that
reaches all of them — an unregistered `prompt_id`, a missing tool, an input path that resolves to nothing — is a
misconfigured fan-out, and reporting it as N error records lets the flow succeed with nothing usable in it. The
message names the component, how many items failed, how many distinct error types they raised, and quotes the
lowest-numbered item's error, so there is a cause to act on rather than a count.

Partial failure stays data: one surviving item is enough for the fan-out to publish and the flow to carry on. An
empty `items` list is not this case either — nothing ran, nothing failed, `results` is published empty, and the
flow continues.

#### Limitations

- **The concurrency cap is per fan-out, not fleet-wide.** `max_concurrency` bounds the branches of one
  component. It says nothing about what the rest of the flow, other flows, or other instances are asking of the
  same model at the same time, so it is a blast-radius control rather than a rate limit.
- **There is no per-item fail-fast.** A flow cannot ask the fan-out to abort on the first item that fails. Every
  item runs, and failures come back as error records (see [When an item fails](#when-an-item-fails)); there is no
  config field to change that. Adding one is a small additive change, so raise it if a flow needs the whole
  fan-out to stop early.
- **A pause belongs to the whole component, not to one item.** An `interrupt()` from inside a branch — a tool
  call awaiting the user's approval, for example — reaches the flow and pauses it. While the flow is paused the
  fan-out has published nothing, so a component reading `results` cannot run yet.
- **More than one pending approval inside one fan-out cannot be resumed.** If two branches interrupt, the resume
  paths each carry a single value, so LangGraph asks for the interrupt ID and the resume fails. The limit is
  latent rather than live: the other path that dispatches concurrently is gated on the `dap_parallel_subagents`
  feature flag, which is not yet defined in the GitLab feature-flag config. A failed resume does not destroy the
  run — the checkpoint survives, and a resume keyed by interrupt ID completes it. Keep at most one
  approval-gated tool call inside a fanned-out body.
- **Routing happens once, after the fan-out.** A router attached to the component runs a single time, on the
  flow's state, once every branch has finished. There is no way to send one item down a different path from
  another. To route on what the items produced, put a component after the fan-out that reads `results` and
  publishes its own verdict for the router to branch on.
- **An item is matched to its result by position only.** A `results` entry does not repeat the item it came from.
  Keep the list passed to `items` if a later component needs to name the item again, and remember that positions
  line up against the first `max_items` entries when `truncated` is `true`.

#### Complete for_each Example

This flow discovers a list of files, reviews each one in its own agent run, and then summarizes the reviews.

```yaml
version: "experimental"
environment: remote

components:
    - name: "discover"
      type: AgentComponent
      prompt_id: "discover_changed_files"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:goal"
      toolset:
          - "find_files"
      # The response schema is what makes the answer a list. Without one, an
      # agent's `final_answer` is the text it wrote, and `for_each` iterates a
      # list only -- so `items` points at the array field inside the schema.
      response_schema_id: "discovered_files"

    - name: "review_one"
      type: AgentComponent
      prompt_id: "review_one_file"
      prompt_version: "^1.0.0"
      inputs:
          - from: "context:project_id"
          - from: "context:item"
      toolset:
          - "read_file"
      for_each:
          # Shown with a non-default `max_items` on purpose: the default is 1000.
          items: "context:discover.final_answer.files"
          as: "context:item"
          max_items: 200

    - name: "summarize"
      type: AgentComponent
      prompt_id: "summarize_reviews"
      prompt_version: "^1.0.0"
      inputs:
          # `review_one` publishes the collected results, not one final_answer.
          - from: "context:review_one.results"

routers:
    - from: "discover"
      to: "review_one"
    - from: "review_one"
      to: "summarize"
    - from: "summarize"
      to: "end"

response_schemas:
    - schema_id: "discovered_files"
      definition:
          "$schema": "http://json-schema.org/draft-07/schema#"
          title: "discovered_files_response"
          type: object
          properties:
              files:
                  type: array
                  items:
                      type: string
                  description: "Paths of the files to review, one per item"
          required: [files]

flow:
    entry_point: "discover"
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
