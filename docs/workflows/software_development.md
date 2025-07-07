# Software Development Flow

## What It Does

Helps you build features, fix bugs, refactor code, and create tests through AI-powered development assistance. Since
it's connected to your GitLab project, it can directly solve issues from your GitLab instance.

## When to Use

- **Feature Development**: Build new functionality from requirements
- **Bug Fixing**: Debug and fix issues across multiple files
- **Code Refactoring**: Improve code structure and quality
- **Test Creation**: Generate comprehensive test suites
- **Documentation**: Create or update code documentation
- **GitLab Issue Resolution**: Directly implement fixes for issues in your GitLab project

## How to Use

To run it locally in VS Code IDE, follow the steps outlined
in [Use the agent in VS code](https://docs.gitlab.com/user/duo_agent_platform/#use-the-agent-platform-in-vs-code)

1. **Describe your task clearly**

   ```plaintext
   Implement user authentication with JWT tokens, including login/logout endpoints,
   middleware for protected routes, and proper error handling
   ```

1. **Provide context**

- Tech stack (for example, Node.js, Express, PostgreSQL)
- Coding standards or patterns to follow
- Existing code examples if applicable

1. **Review the plan** before execution

1. **Test thoroughly** after completion

## Examples

### Good Request

```plaintext
Add input validation to the user registration endpoint:
- Validate email format
- Ensure password meets security requirements (8+ chars, uppercase, number)
- Check username uniqueness
- Return specific error messages
- Follow our existing validation pattern from auth.validator.js
```

### GitLab Issue Request (Good example)

```plaintext
Implement the fix for issue #342 - users can't upload files larger than 5MB
```

### Too Vague

```plaintext
Fix the login bug
Make the code better
```

## Tools Permissions and Approval System

Duo Agent Platform system includes a tool approval mechanism to manage risk when agents use different tools. Tools are
organized into privilege buckets, allowing users to control which tools are available and require approval before
execution. This system helps prevent unintended access to confidential data or execution of potentially harmful
commands.
For detailed information about the tools permissions and approval system, see
the [GitLab Architecture Design Document](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/#tools-permissions-and-approval-system).

## Capabilities

### Can Do

- Create new files and modules
- Modify existing code
- Implement design patterns
- Write unit/integration tests
- Add error handling and logging
- Performance optimizations
- Read and implement GitLab issues directly
- Access project context and existing code
- Run or debug code directly (with command approval)

### Cannot Do

- Access external APIs during execution

## Best Practices

1. **Break down complex tasks** into smaller, focused requests
1. **Include examples** of existing patterns you want followed
