# Adding a New Tool to Duo Workflow Service

For a quick reference you can
use [this merge request](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/2630)
to see all the code changes needed to add a new simple tool. You can continue reading to understand more complex cases.

This guide provides step-by-step instructions for implementing and integrating a new tool into the GitLab Duo Workflow
Service.

## Introduction

Tools enable AI agents in the GitLab Duo Workflow Service to interact with GitLab resources, manipulate files, execute
commands, and perform various actions. Each tool has a specific purpose with defined inputs and outputs.

This guide covers how to create and integrate new tools into the Duo Workflow Service, including design considerations,
implementation details, and best practices.

## Implementation Steps

### 2. Create the Tool Class

1. **Choose the Right Location**:
   - Create a new file in `duo_workflow_service/tools/` for a new category of tools
   - Or add to an existing file for related functionality

1. **Define Input Schema**:
   Create a Pydantic model for your tool's input parameters:

   ```python
   from pydantic import BaseModel, Field

   class YourToolInput(BaseModel):
       param1: str = Field(description="Description of the first parameter")
       param2: int = Field(description="Description of the second parameter")
       optional_param: str = Field(None, description="Description of an optional parameter")
   ```

1. **Implement the Tool Class**:
   Extend the `DuoBaseTool` class:

   ```python
   from typing import Type

   from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
   from contract import contract_pb2
   from duo_workflow_service.executor.action import _execute_action

   class YourTool(DuoBaseTool):
       name: str = "your_tool_name"
       description: str = """
       Detailed description of what your tool does.
       Include usage examples and any important notes.
       """
       args_schema: Type[BaseModel] = YourToolInput  # type: ignore

       async def _arun(self, param1: str, param2: int, optional_param: str = None) -> str:
           # Implement the tool logic here

           # If interacting with the executor:
           return await _execute_action(
               self.metadata,  # type: ignore
               contract_pb2.Action(
                   yourToolAction=contract_pb2.YourToolAction(
                       param1=param1,
                       param2=param2,
                       optionalParam=optional_param or "",
                   )
               ),
           )

           # If interacting with GitLab API:
           # result = await self.gitlab_client.make_request(...)
           # return result

       def format_display_message(self, args: YourToolInput) -> str:
           # Format a user-friendly message for the UI
           return f"Performing action with {args.param1}"
   ```

### 3. Update Protocol Buffers (if needed)

Some new tools will require new or updated behavior from the Duo Workflow Executor. If this is the case you'll need to
roll out these changes over multiple steps. This is usually means:

1. Update the protocol buffer definition
1. Update the protocol buffer library in the 2 executors
1. Release the new executors
1. Update Duo Workflow Service to use the new protocol buffers (including adding the tool)

To update the protocol buffer definitions:

1. **Edit Contract Definition**:
   Modify `contract/contract.proto` to add your new action:

   ```protobuf
   message YourToolAction {
     string param1 = 1;
     int32 param2 = 2;
     string optional_param = 3;
   }

   message Action {
     // Existing actions...
     oneof action {
       // Other actions...
       YourToolAction yourToolAction = 123; // Use next available number
     }
   }
   ```

1. **Generate Protobuf Files**:

   ```shell
   make gen-proto
   ```

### 4. Register the Tool

It is critical for security that tools are categorized by their capabilities. These categories are not for code
organization but for security and safety purposes. Our `read_write_files` and `read_only_gitlab` are the only tools that
don't usually require approval and as such these lists should not contain any risky behavior. Any additions to those
lists requires an appsec approval and you should explain clearly what the new tool can do and the risks if this tool is
called without user approval by an agent misled by prompt injection.

Add your tool to the appropriate list in `duo_workflow_service/components/tools_registry.py`:

```python
from duo_workflow_service.tools.your_tool_file import YourTool

# For read-only tools
_READ_ONLY_GITLAB_TOOLS: list[Type[BaseTool]] = [
    # Existing tools...
    YourTool,
]

# For read-write tools
_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_write_files": [
        # Existing tools...
    ],
    "use_git": [
        # Existing tools...
    ],
    "read_write_gitlab": [
        # Existing tools...
        YourTool,  # Add here if it modifies GitLab resources
    ],
    "read_only_gitlab": [
        # Existing tools...
        YourTool,  # Add here if it only reads GitLab resources
    ],
    "run_commands": [
        # Existing tools...
    ],
}
```

### 5. Add the Tool to Workflows

Add your tool to the appropriate workflow tool lists in:

- `duo_workflow_service/workflows/software_development/workflow.py` (for executor tools)
- `duo_workflow_service/workflows/chat/workflow.py` (for chat tools)

```python
EXECUTOR_TOOLS = [
    # Existing tools...
    "your_tool_name",
]

# Or for context builder tools
CONTEXT_BUILDER_TOOLS = [
    # Existing tools...
    "your_tool_name",
]
```

## Best Practices

1. **Follow Naming Conventions**:
   - Use descriptive names for your tool and input classes
   - Use snake_case for tool names (e.g., `get_file_content`, `update_issue`)
   - Use PascalCase for classes (e.g., `GetFileContent`, `UpdateIssue`)

1. **Write Clear Documentation**:
   - Documentation is read by the LLM to determine when and how to call a tool
   - Documentation that is too verbose may waste time and input tokens
   - Documentation that is not clear enough may confuse an LLM
   - Provide detailed descriptions for your tool
   - Document parameters thoroughly with examples
   - Explain any side effects

1. **Security Considerations**:
   - Add the tool to the appropriate privilege group based on what it does
   - Consider whether the tool should
     require [human approval](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/#how-agent-tool-set-is-being-defined-for-each-workflow-run)
     before execution

1. **Performance**:
   - Keep tools focused on a single responsibility
   - Optimize for minimal API calls when possible
   - Use async properly to avoid blocking operations

## Troubleshooting

### Common Issues

1. **Tool Not Appearing in Agent Interface**:
   - Verify the tool is registered in `tools_registry.py`
   - Check that the tool name is included in the appropriate workflow tool list
   - Ensure the workflow is using the tool created

1. **Tool Failing to Execute**:

1. **Protocol Buffer Errors**:
   - Make sure you've regenerated the protocol buffers with `make gen-proto`
   - Check that your action is properly defined in the proto file
   - Verify field numbers don't conflict with existing ones

1. **GitLab API permissions**:
   - When making calls to GitLab API, a 403 error indicates insufficient permissions.
   - To resolve this, ensure the endpoint allows the `ai_workflows` scope.
     See [MR](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/193297) for more details.

### Tool Implementation Examples

- [Epic API Tool](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/178085/diffs#2128623ff30bc6500f22d7daf419c3c604327984).

- [File System Tool refactoring](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/2555).
