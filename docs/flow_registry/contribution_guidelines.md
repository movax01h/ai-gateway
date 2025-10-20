# Flow Registry Contribution Guidelines

This document provides guidelines for contributing to the Flow Registry framework. It is intended for developers who want to extend the framework by adding new components, routers, or improving core functionality.

For information on how to use the Flow Registry to build flows, see the [Flow Registry Framework Developer Guide](index.md).

[[_TOC_]]

## Overview

The Flow Registry is a component-based framework that forms a public API within AI Gateway. Contributions should maintain the framework's stability, extensibility, and backwards compatibility while enabling developers to build powerful AI-powered flows.

### Who Should Read This

- Developers adding new components or routers to the framework
- Contributors improving core Flow Registry functionality
- Team members reviewing Flow Registry changes

## Versioning and Breaking Changes

Flow Registry forms a public API of AI Gateway and follows similar principles to REST API [versioning](https://docs.gitlab.com/development/api_styleguide/#breaking-changes). Interfaces of framework entities (components, routers, flows, prompts, etc.) must not introduce breaking changes within a stable version.

### Current Versions

| Version | Status | Purpose | Backwards Compatibility |
|---------|--------|---------|------------------------|
| `v1` | **Stable** | Production use and feature development | Guaranteed - all changes must be backwards compatible |
| `experimental` | **Unstable** | Prototyping and de-risking high-impact changes | Not guaranteed - breaking changes allowed |

For detailed version information, see the [versions documentation](index.md#versions).

### Version Guidelines

**For `v1` (Stable Version):**

- All changes MUST be backwards compatible
- Existing component interfaces cannot be modified in breaking ways:
  - Renaming of the component.
  - Removing or renaming a parameter.
  - Non backwards compatible change of the type of the parameter value.
  - Adding a new parameter if it is required.
- New optional parameters can be added
- Deprecation warnings should precede removal of functionality (follow GitLab deprecation process)
- Documentation must be updated with all changes

**For `experimental` Version:**

- Breaking changes are allowed
- Suitable for prototyping and experimentation
- NOT suitable for external feature development
- Can be used to validate designs before implementing in stable version

## Core Architecture

### State Management

Flow Registry uses a fixed LangGraph state structure to ensure consistency and compatibility across all components and routers. The state structure is defined in [`duo_workflow_service/agent_platform/v1/state/base.py`](/duo_workflow_service/agent_platform/v1/state/base.py).

#### State Structure

The state includes the following top-level attributes:

- **`context`**: Nested dictionary for data exchange between components
- **`conversation_history`**: Component-specific message history for AI interactions
- **`status`**: Current workflow execution status
- **`ui_chat_log`**: List of data envelopes for client UI updates

#### State Design Principles

1. **Fixed Top-Level Structure**: The top-level attributes (`context`, `conversation_history`, `status`, `ui_chat_log`) must remain fixed to ensure cross-component and router compatibility.

1. **Dynamic Content Placement**: All dynamic content should be placed within the `context` attribute. This includes:
   - Data generated during flow execution
   - Input provided via [additional context](index.md#additional-context) at session start
   - Component outputs and intermediate results

1. **Avoid Direct State Access**: Components and routers should not access state directly. Instead, use the `IOKey` and `IOKeyTemplate` abstractions (see next section).

### IOKey and IOKeyTemplate Abstraction

To abstract state interactions and prevent tight coupling across the codebase, Flow Registry uses `IOKey` and `IOKeyTemplate` classes.

#### IOKey

An `IOKey` object wraps a single leaf in the flow session state, providing a clean interface for reading from and writing to specific state locations.

**Example:**

```python
# Project ID stored at context.project_id
project_id_key = IOKey(
    target='context',
    subkeys=['project_id'],
    alias=None,
    literal=False
)
```

#### IOKeyTemplate

When a component generates data dynamically and cannot declaratively specify the complete state path at design time, use `IOKeyTemplate`. This abstraction defines `IOKey` instances with dynamic parts.

**Why Use IOKeyTemplate?**

- Prevents naming collisions when multiple instances of the same component exist in a flow
- Enables component-scoped data storage
- Maintains clean separation between component instances

**Example:**

To ensure `AwesomeComponent` writes results without risking collisions:

```python
# Define template with component name placeholder
template = IOKeyTemplate(
    target='context',
    subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, 'cool_data']
)

# Convert to IOKey at runtime using component's actual name
iokey = template.to_iokey({
    IOKeyTemplate.COMPONENT_NAME_TEMPLATE: self._name
})
# Result: IOKey(target='context', subkeys=['my_cool_stuff', 'cool_data'], ...)
```

#### Design Principle

**All Flow Registry elements (components, routers, etc.) MUST use `IOKey` and `IOKeyTemplate` for state interactions.** Never access the state dictionary directly.

## Contributing Components

Components are the fundamental building blocks of Flow Registry. They represent specific structures within LangGraph graphs, implemented as sets of nodes and edges that attach to the main graph via the `attach` method.

### Component Architecture

#### Base Class

All components MUST inherit from the `BaseComponent` class defined in [`duo_workflow_service/agent_platform/v1/components/base.py`](duo_workflow_service/agent_platform/v1/components/base.py). This base class defines the required interface and attributes that all components must implement.

#### Required Interface

Every component must implement the following methods:

1. **`attach(self, graph: StateGraph, router: RouterProtocol) -> None`**
   - Builds and attaches the component's LangGraph structure to the main graph
   - Parameters:
     - `graph`: The LangGraph StateGraph instance to attach to
     - `router`: Router instance controlling graph exit from the component
   - Should add nodes, edges, and conditional edges as needed

1. **`__entry_hook__(self) -> str`**
   - Property that returns the name of the LangGraph node serving as the entry point
   - Used by the framework to connect components in the flow

#### Required Attributes

Components must override these class attributes:

1. **`_outputs: List[IOKeyTemplate]`**
   - Declares all state attributes the component mutates during execution
   - Enables framework validation and documentation generation
   - Example: `[IOKeyTemplate(target='context', subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, 'result'])]`

1. **`_allowed_input_targets: List[str]`**
   - Lists top-level state keys the component can read from
   - Typically includes `['context', 'status', 'conversation_history']`
   - Used for input validation

1. **`supported_environments: List[str]`**
   - Specifies environment types compatible with this component
   - Options: `['ambient', 'chat', 'chat-partial']` (or subset), see the current version environment [documentation](v1.md#environment)
   - Prevents component usage in incompatible environments

### Component State Interactions

Components MUST NOT interact directly with the state dictionary. Instead, use `IOKey` instances to read from and write to the state.

#### Input Handling

Components receive inputs through the `inputs` attribute, which is configured by flow authors in the YAML configuration:

```yaml
components:
  - name: "my_component"
    type: "AgentComponent"
    inputs:
      - "context:goal"
      - from: "context:previous.result"
        as: "previous_result"
```

The component processes these inputs using the IOKey system, ensuring type safety and validation. The `BaseComponent` class implements `build_base_component` as a Pydantic validator that handles conversion of string representations into `IOKey` instances.

#### Output Generation

Components write outputs to state locations declared in their `_outputs` attribute. Use `IOKeyTemplate` to generate component-scoped output paths:

```python
class MyComponent(BaseComponent):
    _outputs = [
        IOKeyTemplate(
            target='context',
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, 'result']
        ),
        IOKeyTemplate(
            target='context',
            subkeys=[IOKeyTemplate.COMPONENT_NAME_TEMPLATE, 'metadata']
        ),
    ]
```

### Routing Out of Components

The router instance passed to the `attach` method controls how execution exits the component. Use LangGraph's conditional edges to connect the component to the router.

#### Basic Routing

For components with a single exit point, connect directly to the router:

```python
def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
    # ... component setup ...

    # Connect final node to router
    graph.add_conditional_edges(
        node_final_response.name,
        router.route,
    )
```

#### Conditional Routing

For components requiring internal routing logic before delegating to the external router, wrap the router with a custom routing method:

```python
def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
    # ... component setup ...

    # Add conditional edge with wrapped router
    graph.add_conditional_edges(
        f"{self.name}#tools",
        partial(self._tools_router, router)
    )

def _tools_router(
    self,
    outgoing_router: RouterProtocol,
    state: FlowState
) -> str:
    """Route based on tool execution results and correction attempts."""
    conversation = state.get(FlowStateKeys.CONVERSATION_HISTORY, {}).get(
        self.name, []
    )

    if not conversation:
        raise RoutingError(
            f"No conversation history found for component {self.name}. "
            f"Tool node should have added messages."
        )

    last_message = conversation[-1]

    # Delegate to outgoing router when conditions are met
    if self._should_exit(last_message):
        return outgoing_router.route(state)

    # Continue internal component processing
    return self._get_internal_node_name()
```

**Key Principles:**

- Always delegate to the injected router for final exit routing
- Use internal routing logic only for component-specific flow control
- Raise `RoutingError` for exceptional conditions

### Node Classes

Components can extract node execution logic into separate Node classes for better code organization and reusability.

#### Node Architecture

- **Internal to Components**: Node classes are implementation details, not part of the public Flow Registry interface
- **Separation of Concerns**: Nodes handle specific execution tasks while components manage graph structure
- **Reusability**: There is an open [issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1374) to research and outline guidelines how to reuse nodes across components, for the time being Flow Registry advises to not share nodes, in order to avoid cross component coupling

#### Example Structure

```python
class MyComponentNode:
    """Internal node implementation for MyComponent."""

    def __init__(self, component_name: str, config: dict):
        self.component_name = component_name
        self.config = config

    def execute(self, state: FlowState) -> FlowState:
        """Execute node logic and return updated state."""
        # Node implementation
        return state

class MyComponent(BaseComponent):
    def attach(self, graph: StateGraph, router: RouterProtocol) -> None:
        node = MyComponentNode(self.name, self.config)
        graph.add_node(f"{self.name}#main", node.execute)
```

## Component Development Workflow

### 1. Design Phase

Before implementing a new component:

- **Define the component's purpose**: What specific task does it perform?
- **Identify inputs and outputs**: What data does it need and produce?
- **Determine environment compatibility**: Which environments support this component?
- **Check for existing components**: Can existing components be extended instead?

### 1. Implementation Phase

1. **Create the component class** in `duo_workflow_service/agent_platform/experimental/components/`. Implementing **new components within experimental version** grants required time to verify stability and quality of it, before it gets released into stable version
1. **Inherit from `BaseComponent`**
1. **Implement required methods and attributes**
1. **Add comprehensive docstrings** explaining usage and configuration
1. **Implement node logic** (extract to Node classes if complex)

### 1. Testing Phase

Comprehensive testing ensures Flow Registry components are reliable and maintainable. Follow these testing principles when contributing components or improving framework functionality.

#### Core Testing Principles

1. **Test the Public Interface**: Test components through their public methods (`__init__`, `attach`, `__entry_hook__`) rather than private methods (prefixed with `_`). Assert on return values and calls to other classes.

1. **Integration-Style Testing**: Test components by attaching them to real `StateGraph` instances and executing the compiled graph. This validates the complete component behavior including graph structure, node execution, and state management.

1. **Extract Complex Node Logic**: If component nodes contain complex logic requiring in-depth testing, extract the node into a separate class and test it independently.

#### Component Testing Pattern

The recommended approach for testing components is demonstrated in `TestOneOffComponentToolsRouter` from [`tests/duo_workflow_service/agent_platform/v1/components/one_off/test_v1_one_off_component.py`](../../tests/duo_workflow_service/agent_platform/v1/components/one_off/test_v1_one_off_component.py).

**Example Test Structure:**

```python
def test_successful_execution_flow(
    self,
    component_name,
    flow_id,
    flow_type,
    mock_dependencies,
    mock_router,
    mock_node_classes,
    base_flow_state,
):
    """Test component execution flow with real graph."""
    # 1. Create real StateGraph instance
    graph = StateGraph(FlowState)

    # 2. Configure mock node to return expected state updates
    mock_node = mock_node_class.return_value
    mock_node.run.return_value = {
        **base_flow_state,
        FlowStateKeys.CONVERSATION_HISTORY: {
            component_name: [AIMessage(content="Expected result")]
        },
    }

    # 3. Configure router behavior
    mock_router.route.return_value = END

    # 4. Instantiate and attach component
    component = MyComponent(
        name=component_name,
        flow_id=flow_id,
        flow_type=flow_type,
        # ... other parameters
    )
    component.attach(graph, mock_router)

    # 5. Set entry point, compile, and execute
    graph.set_entry_point(component.__entry_hook__())
    compiled_graph = graph.compile()
    result = compiled_graph.invoke(base_flow_state)

    # 6. Assert on node calls
    mock_node.run.assert_called_once()

    # 7. Assert on router calls
    mock_router.route.assert_called_once()

    # 8. Assert on final state
    assert "expected_data" in result["context"][component_name]
```

**Key Points:**

- **Use Real Graphs**: Always create actual `StateGraph` instances, don't mock the graph itself
- **Mock Node Classes**: Mock the Node class constructors but use their instances in real graph execution
- **Execute End-to-End**: Compile and invoke the graph to test complete execution flow
- **Assert on Behavior**: Verify node calls, router calls, and final state, not internal implementation details

#### Testing Complex Node Logic

When nodes contain complex logic (error handling, retry logic, multiple operations), extract the node class and test it independently.

**Example:** See [`tests/duo_workflow_service/agent_platform/v1/components/one_off/nodes/test_v1_tool_node_with_error_correction.py`](../../tests/duo_workflow_service/agent_platform/v1/components/one_off/nodes/test_v1_tool_node_with_error_correction.py)

```python
class TestMyNodeClass:
    """Test suite for MyNode class."""

    @pytest.mark.asyncio
    async def test_successful_execution(
        self,
        node_instance,
        flow_state,
        component_name,
    ):
        """Test node execution with valid inputs."""
        result = await node_instance.run(flow_state)

        # Assert on state updates
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Verify expected data in state
        assert "expected_output" in result["context"][component_name]
```

#### What to Test

Every component contribution must include tests for:

1. **Initialization and Configuration**
   - Input parameter handling and validation
   - IOKeyTemplate replacement with component names
   - Default values for optional parameters

1. **Graph Structure Creation**
   - Correct nodes added to graph via `attach` method
   - Edges between nodes
   - Entry point hook returns correct node name

1. **Execution Flow and Routing**
   - Successful execution paths
   - Error handling and retry logic
   - Routing decisions (internal and external)
   - State management throughout execution

1. **Node-Specific Logic** (when extracted)
   - Complex error handling
   - Retry mechanisms
   - Security and validation
   - Monitoring and metrics

#### Test Organization

Structure tests using descriptive test classes:

```python
class TestMyComponentInitialization:
    """Test suite for component initialization and configuration."""

class TestMyComponentAttachNodes:
    """Test suite for node creation via attach method."""

class TestMyComponentAttachEdges:
    """Test suite for edge creation and graph structure."""

class TestMyComponentExecutionFlow:
    """Test suite for component execution and routing behavior."""
```

#### Mocking Strategy

**Mock External Dependencies, Not the Graph:**

```python
@pytest.fixture(name="mock_node_class")
def mock_node_class_fixture(component_name):
    """Fixture for mocked Node class."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.my_component.component.MyNode"
    ) as mock_cls:
        mock_node = Mock()
        mock_node.name = f"{component_name}#node"
        mock_cls.return_value = mock_node
        yield mock_cls

@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    """Fixture for mocked Toolset."""
    toolset = Mock()
    toolset.bindable = [Mock(name="tool1")]
    return toolset
```

### 1. Documentation Phase

1. **Update component documentation** within appropriate version page
1. **Add example configurations** showing common use cases
1. **Document all configuration parameters**

### 1. Review and Merge

1. **Create merge request** following GitLab contribution guidelines
1. **Request review** from Flow Registry experts (agent foundations group)

## Code Style and Conventions

### Naming Conventions

- **Component classes**: PascalCase with "Component" suffix (e.g., `AgentComponent`)
- **Node classes**: PascalCase with "Node" suffix (e.g., `AgentNode`)
- **Methods**: snake_case (e.g., `attach`, `_internal_method`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`)

### Documentation

- **Docstrings**: Use Google-style docstrings for all public methods and classes
- **Type hints**: Include type hints for all method parameters and return values
- **Comments**: Explain complex logic and design decisions

### Code Organization

```plaintext
duo_workflow_service/agent_platform/v1
├── __init__.py
├── components
│   ├── __init__.py
│   ├── agent  # Component implementation
│   │   ├── __init__.py
│   │   ├── component.py
│   │   ├── nodes
│   │   │   ├── __init__.py
│   │   │   ├── agent_node.py
│   │   │   ├── final_response_node.py
│   │   │   └── tool_node.py
│   │   └── ui_log.py
│   ├── base.py  # BaseComponent implementation
│   ├── one_off  # Component implementation
│   │   ├── __init__.py
│   │   ├── component.py
│   │   ├── nodes # Node implementations (if needed)
│   │   │   └── tool_node_with_error_correction.py
│   │   └── ui_log.py
│   └── registry.py
├── flows  # Flow implementations
│   ├── __init__.py
│   ├── base.py
│   ├── configs  # Flows YAML config files implementing AI powered features
│   │   ├── __init__.py
│   │   └── sast_fp_detection.yml
│   └── flow_config.py  # Pydantic model class which models Flow Registry syntax
├── routers  # Router implementations
│   ├── __init__.py
│   ├── base.py
│   └── router.py
├── state
│   ├── __init__.py
│   └── base.py  # State definitions
└── ui_log
    ├── __init__.py
    ├── base.py
    └── factory.py
```

## Documentation Requirements

When contributing components, you must:

1. **Update the components section** within appropriate version page (use `experimental` for new components) with:
   - Component name and purpose
   - Configuration parameters
   - Input/output specifications
   - Example YAML configuration
   - Supported environments

1. **Provide usage examples** demonstrating:
   - Basic configuration
   - Advanced features
   - Common patterns
   - Integration with other components

## Review Process

### What Reviewers Look For

- **Backwards compatibility**: No breaking changes in stable versions
- **Code quality**: Clean, maintainable, well-documented code
- **Test coverage**: Comprehensive tests for all functionality
- **Documentation**: Clear, complete documentation with examples
- **Performance**: Efficient state management and execution
- **Error handling**: Graceful failure and clear error messages

### Merge Request Checklist

Before submitting, ensure:

- [ ] All tests pass locally and in CI/CD
- [ ] Documentation is complete and accurate
- [ ] Code follows style conventions
- [ ] Backwards compatibility is maintained (for v1)
- [ ] Example YAML configurations are provided
- [ ] Merge request description explains changes clearly

## Common Pitfalls to Avoid

1. **Direct state access**: Always use IOKey/IOKeyTemplate
1. **Missing output declarations**: Declare all state mutations in `_outputs`
1. **Hardcoded values**: Use configuration parameters for flexibility
1. **Poor error messages**: Provide actionable error information
1. **Incomplete documentation**: Document all parameters and behaviors
1. **Breaking changes in v1**: Never break backwards compatibility
1. **Missing environment checks**: Validate environment compatibility

## Getting Help

- **Questions**: Ask in the `#g_agent_foundations` Slack channel
- **Issues**: Create issues in the AI Gateway project
- **Design discussions**: Start discussion in merge requests or issues

## Additional Resources

- [Flow Registry Developer Guide](index.md)
- [v1 Component Reference](v1.md)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [GitLab Contributing Guidelines](https://docs.gitlab.com/ee/development/contributing/)
