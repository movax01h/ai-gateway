# Tool Security Override Guide

## Overview

By default, all tools apply a standard set of security functions to their responses. However, some tools may require custom security configurations based on their risk profile and data sources.

**⚠️ Important: All security function overrides require AppSec team approval before merging.**

## When to Use Overrides

Use `TOOL_SECURITY_OVERRIDES` when:
- Your tool processes trusted, controlled data sources
- Default security is too restrictive for your use case
- You need a specific combination of security functions

**Do NOT use overrides for:**
- Tools handling user-generated content (issues, comments, MRs)
- Tools processing untrusted external data
- When in doubt - default security is appropriate

## How to Configure

### ⚠️ Single Source of Truth (SSoT)

**IMPORTANT:** All tool security overrides MUST be defined in the `TOOL_SECURITY_OVERRIDES` dictionary in `prompt_security.py`. This is the Single Source of Truth.

**DO NOT** set overrides dynamically at runtime like this:
```python
# ❌ WRONG - Do not do this!
PromptSecurity.TOOL_SECURITY_OVERRIDES['my_tool'] = [encode_dangerous_tags]
```

This ensures all security configurations are:
- Centralized and easy to audit
- Subject to AppSec review via CODEOWNERS
- Version controlled with proper change history

### Step 1: Edit the TOOL_SECURITY_OVERRIDES Dictionary

Open `duo_workflow_service/security/prompt_security.py` and add your override to the `TOOL_SECURITY_OVERRIDES` dictionary:

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # read_file accesses only our controlled repository content.
    # Standard security is too restrictive and modifies legitimate file content.
    # We apply minimal security to preserve file integrity.
    'read_file': [encode_dangerous_tags],

    # Code analysis tool needs exact code content without modifications
    # Risk: LOW - only processes repository code, not user-generated content
    'lint_code': [],
}
```

### Step 2: Document Your Decision

Always add a comment explaining:
1. **Why** the override is needed
2. **Risk assessment** (LOW/MEDIUM/HIGH)
3. **Data source** (where does the tool get its data from)

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # Tool: read_file
    # Why: Accesses only controlled repository content
    # Risk: LOW - no user-generated content
    # Security: Minimal encoding to preserve file integrity
    'read_file': [encode_dangerous_tags],
}
```

### Step 3: Get AppSec Approval

1. Create your MR with the changes to `prompt_security.py`
2. Request review from AppSec team (automatically required by CODEOWNERS)
3. Address any security concerns raised during review

## Available Security Functions

You can import and use these functions in your override:

```python
from duo_workflow_service.security.prompt_security import (
    encode_dangerous_tags,
    strip_hidden_unicode_tags,
)
from duo_workflow_service.security.markdown_content_security import (
    strip_hidden_html_comments,
    strip_mermaid_comments,
)
```

## Configuration Patterns

### Pattern 1: Reduced Security (Subset of Functions)

For tools with controlled data sources, add to the dictionary in `prompt_security.py`:

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # my_tool: Processes only internal API responses
    # Risk: LOW - controlled data source
    'my_tool': [
        encode_dangerous_tags,
        strip_hidden_unicode_tags,
    ],
}
```

### Pattern 2: No Security (Empty List)

For fully trusted internal tools, add to the dictionary in `prompt_security.py`:

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # internal_tool: Static code analysis on repository code only
    # Risk: LOW - no external input, reads source files directly
    'internal_tool': [],
}
```

### Pattern 3: Default Security (No Override)

For tools with user content, simply don't add an entry to the dictionary:

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # No entry for 'get_issue' - it automatically gets default security
    # This is the safest option when unsure
}
```

## Testing Your Configuration

After adding an override to the `TOOL_SECURITY_OVERRIDES` dictionary in `prompt_security.py`, test it:

### Manual Testing

```python
from duo_workflow_service.security.prompt_security import PromptSecurity

# Test your tool's security configuration
result = PromptSecurity.apply_security_to_tool_response(
    "test input with <system>tags</system>",
    "my_tool"
)
print(result)
```

### Run Test Suite

Ensure no regressions:

```bash
poetry run pytest tests/duo_workflow_service/security/ -v
```

## Risk Assessment Checklist

Before adding an override, verify:

- [ ] Tool does NOT process user-generated content
- [ ] Data source is controlled/trusted
- [ ] Override is necessary (default security causes issues)
- [ ] You've documented the reason for the override
- [ ] You've tested with sample data
- [ ] **AppSec approval obtained** (required for all security overrides)

## Examples

All examples below show entries in the `TOOL_SECURITY_OVERRIDES` dictionary in `prompt_security.py`:

### Example 1: File Reading Tool

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # read_file: Reads from controlled git repository only
    # Risk: LOW - no user-generated content
    # Security: Minimal encoding to preserve file integrity
    'read_file': [encode_dangerous_tags],
}
```

### Example 2: Code Analysis Tool

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # analyze_code: Static analysis tool needs exact code content
    # Risk: LOW - processes repository source code only
    # Security: None - needs unmodified code for accurate analysis
    'analyze_code': [],
}
```

### Example 3: High-Risk Tool (No Override)

```python
TOOL_SECURITY_OVERRIDES: Dict[...] = {
    # get_issue fetches user-generated content - use default security
    # NO entry here - defaults apply automatically ✓
}
```

## Important Notes

1. **Single Source of Truth**: ALL overrides must be in the `TOOL_SECURITY_OVERRIDES` dictionary in `prompt_security.py`
2. **Tool Name Match**: Use exact tool name string (case-sensitive)
3. **Empty List Valid**: `[]` means no security functions will be applied
4. **Order Matters**: Security functions execute in the order listed in the dictionary
5. **No Runtime Changes**: Never modify `TOOL_SECURITY_OVERRIDES` dynamically at runtime

## Need Help?

- Review existing overrides in `duo_workflow_service/security/prompt_security.py`
- Check test cases: `tests/duo_workflow_service/tools/test_prompt_security.py`
- Contact the AppSec team for approval and guidance on security overrides
- All overrides require AppSec review - see `.gitlab/CODEOWNERS`

## Future: Phase 2 - Flow-Level Security

This guide covers Phase 1 (tool-level overrides). Phase 2 will introduce flow-level security policies where the same tool can have different security based on the flow context. Your tool-level overrides will integrate seamlessly with future flow-level policies.
