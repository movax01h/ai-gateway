# Conversation Compaction

This document explains conversation compaction - what it is, why it's needed, and how to configure it across different workflow types.

[[_TOC_]]

## What is Conversation Compaction?

Conversation compaction is a feature that automatically summarizes older parts of conversation history when it grows too large. This helps maintain context within LLM token limits while preserving important information from earlier in the conversation.

When enabled, compaction:

1. Monitors the conversation history token count
1. When the threshold is exceeded, summarizes older messages into a concise summary
1. Keeps recent messages intact for immediate context
1. Replaces the conversation history with: summary + recent messages

## Why Use Compaction?

**Problem**: LLMs have finite context windows. Long-running agent conversations can exceed these limits, causing:

- Truncated context (losing important early information)
- API errors from exceeding token limits
- Degraded agent performance due to missing context

**Solution**: Compaction intelligently summarizes older conversation turns, preserving key information while staying within token budgets.

**When to Enable Compaction**:

- Multi-turn agent conversations that may run for many iterations
- Agents that accumulate significant tool call history
- Flows where early context (user requirements, decisions made) must persist

**When NOT to Enable Compaction**:

- Single-turn or few-turn interactions
- Simple tool executions with minimal history
- When you need exact conversation history preserved

## Supported Workflow Types

Compaction is being integrated incrementally across workflow implementations:

| Workflow Type | Support | Configuration Method |
|---------------|---------|---------------------|
| Legacy Chat Workflow | Yes | Enabled by default via `create_agent()` factory |
| Legacy Software Development Workflow | Yes | Enabled via `CompactionConfig` in workflow setup |
| Flow Registry Experimental (AgentComponent) | Yes | YAML configuration |
| Flow Registry Experimental (OneOffComponent) | Yes | YAML configuration |
| Flow Registry v1 (AgentComponent) | No | Not yet integrated |
| Flow Registry v1 (OneOffComponent) | No | Not yet integrated |
| Flow Registry (DeterministicStepComponent) | No | No conversation history |

## Configuration

### Legacy Chat Workflow

Compaction is **enabled by default** for the legacy chat workflow. The `create_agent()` factory function in `chat_agent_factory.py` creates a `ConversationCompactor` when `compaction=True` (the default in the chat workflow).

To customize compaction behavior, pass a `CompactionConfig` to the `compaction` parameter:

```python
# In workflow setup
from duo_workflow_service.agents.chat_agent_factory import create_agent
from duo_workflow_service.conversation.compaction import CompactionConfig

agent = create_agent(
    user=user,
    tools_registry=tools_registry,
    # ... other params ...
    compaction=CompactionConfig(
        max_recent_messages=20,
        trim_threshold=0.8,
    ),
)
```

To disable compaction, pass `False`:

```python
agent = create_agent(
    # ... other params ...
    compaction=False,  # Disabled
)
```

### Legacy Software Development Workflow

Compaction is **enabled by default** for all agents in the software development workflow (context_builder, planner, executor). The `build_agent()` factory function in `agent.py` creates a `ConversationCompactor` when a `CompactionConfig` is provided.

To customize compaction behavior, pass a `CompactionConfig` when setting up workflow components:

```python
# In workflow setup
from duo_workflow_service.conversation.compaction import CompactionConfig

# For PlannerComponent
planner_component = PlannerComponent(
    # ... other params ...
    compaction=CompactionConfig(
        max_recent_messages=20,
        trim_threshold=0.8,
    ),
)

# For ExecutorComponent
executor_component = ExecutorComponent(
    # ... other params ...
    compaction=CompactionConfig(
        max_recent_messages=20,
        trim_threshold=0.8,
    ),
)
```

To disable compaction, omit the `compaction` parameter or pass `None`:

```python
planner_component = PlannerComponent(
    # ... other params ...
    # compaction not specified = disabled
)
```

### Flow Registry Experimental

For flows built with Flow Registry Experimental, configure compaction in the YAML flow configuration.

The `compaction` field accepts either a boolean (`true`/`false`) or a `CompactionConfig` object.

#### Basic Configuration (AgentComponent)

Enable compaction with default settings:

```yaml
components:
  - name: "my_agent"
    type: AgentComponent
    prompt_id: "my_prompt"
    toolset: ["read_file", "edit_file"]
    compaction: true
```

#### Custom Configuration (AgentComponent)

Fine-tune compaction behavior by providing config parameters directly:

```yaml
components:
  - name: "my_agent"
    type: AgentComponent
    prompt_id: "my_prompt"
    toolset: ["read_file", "edit_file"]
    compaction:
      max_recent_messages: 20
      recent_messages_token_budget: 20000
      trim_threshold: 0.8
```

#### OneOffComponent Configuration

```yaml
components:
  - name: "file_processor"
    type: OneOffComponent
    prompt_id: "process_files"
    toolset: ["read_file", "edit_file"]
    compaction: true
    max_correction_attempts: 3
```

#### Disabling Compaction

Either omit the `compaction` field entirely, or explicitly set to `false`:

```yaml
components:
  - name: "simple_agent"
    type: AgentComponent
    prompt_id: "simple_prompt"
    toolset: ["read_file"]
    # No compaction field = disabled (default is false)

  - name: "another_agent"
    type: AgentComponent
    prompt_id: "another_prompt"
    toolset: ["read_file"]
    compaction: false  # Explicitly disabled
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_recent_messages` | int | 10 | Maximum number of recent messages to keep uncompacted |
| `recent_messages_token_budget` | int | 40000 | Token budget for the recent messages section when determining which messages to keep |
| `trim_threshold` | float | 0.7 | Ratio of model's max context limit that triggers compaction (0.0-1.0) |

In YAML, set `compaction: true` to use defaults, or provide a config object with custom parameters. In Python code, pass `CompactionConfig()` to enable or `False` to disable.

## How It Works

1. **Token Estimation**: Before each LLM call, the system estimates the total token count of the conversation history
1. **Threshold Check**: Compaction triggers when both conditions are met:
   - Message count exceeds `max_recent_messages`, AND
   - Total history tokens exceed `trim_threshold * model_max_context_limit`
1. **Message Slicing**: Messages are split into three parts:
   - **Leading context**: First consecutive HumanMessages (initial user context/instructions)
   - **Messages to summarize**: Older conversation turns between leading context and recent messages
   - **Recent messages to keep**: Complete turns from the end, respecting both `max_recent_messages` count and `recent_messages_token_budget` token limits
1. **Summarization**: An LLM call generates a summary of the messages to be compacted
1. **Replacement**: The conversation history is replaced with: leading context + summary message + recent messages

Note: Compaction is not currently surfaced in the user-facing chat UI. The process happens transparently without user notification.

### State Management

Compaction integrates with state management differently depending on the workflow type:

**Legacy Chat Workflow**: The compacted history is written directly back to the state's `conversation_history` before the LLM call. The state is mutated in place within the `ChatAgent.run()` method.

**Legacy Software Development Workflow**: The compacted history is written directly back to the state's `conversation_history` before the LLM call in the `Agent.run()` method, similar to the Chat workflow.

**Flow Registry Experimental**: Compaction happens before the LLM invocation in `AgentNode.run()`. The compacted history is used for the prompt, and the complete history (compacted + new completion) is returned via the component's output. The reducer handles updating the conversation history in the flow state.

## Best Practices

1. **Start with defaults**: The default configuration works well for most use cases
1. **Increase `max_recent_messages` for complex tasks**: If recent context is critical, keep more messages
1. **Lower `trim_threshold` for aggressive compaction**: If you're hitting token limits, trigger compaction earlier
1. **Monitor compaction frequency**: If compaction happens every turn, consider increasing the token budget

## Feature Flag

Compaction requires the `AI_CONTEXT_COMPACTION` feature flag (runtime key: `ai_context_compaction`) to be enabled. The flag is checked in the `maybe_compact_history()` function. Even with compaction configured, it will fall back to token-based trimming if the feature flag is disabled.
