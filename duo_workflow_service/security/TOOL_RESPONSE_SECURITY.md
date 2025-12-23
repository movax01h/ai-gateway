# Tool Response Security

Protects against prompt injection and malicious content in tool responses.

## How It Works

```text
Tool Response
     │
     ▼
┌─────────────────────────────────────────────────┐
│  PromptSecurity (sanitization)                  │
│  - Encodes dangerous tags (<system>, <goal>)    │
│  - Strips hidden unicode, HTML comments         │
│  - Controlled by: TOOL_SECURITY_OVERRIDES       │
│  - Runs: synchronously, always                  │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  HiddenLayer (detection)                        │
│  - Detects prompt injection patterns            │
│  - Controlled by: tool trust level + FF         │
│  - Runs: async (evaluation mode)                │
│  - Skips: trusted tools                         │
└─────────────────────────────────────────────────┘
     │
     ▼
  Sanitized Response
```

**HiddenLayer modes:**

- Current: evaluation only (async, logs but doesn't block)
- Future: blocking/interrupting (sync), skip entirely

## Configuration

| Variable | Description |
|----------|-------------|
| `PROMPT_SCANNER` | `hidden_layer` or `default` |
| `HL_CLIENT_ID` | HiddenLayer client ID |
| `HL_CLIENT_SECRET` | HiddenLayer client secret |
| `HIDDENLAYER_ENVIRONMENT` | `prod-us` or `prod-eu` |
| `HIDDENLAYER_BASE_URL` | Custom Base URL for HiddenLayer |
| `HL_PROJECT_ID` | Project ID or alias for request governance |

**Feature flag:** `AI_PROMPT_SCANNING` enables HiddenLayer for untrusted tools.

## Tool Trust Levels

| Level | HiddenLayer |
|-------|-------------|
| `TRUSTED_INTERNAL` | skipped |
| `UNTRUSTED_USER_CONTENT` | runs |
| `UNTRUSTED_EXTERNAL` | runs |
| `UNTRUSTED_MCP` | runs |

## Customizing Security

**Skip PromptSecurity for a tool:**

```python
# In prompt_security.py
TOOL_SECURITY_OVERRIDES = {
    "my_tool": [],  # empty = no sanitization
}
```

**Custom sanitization for a tool:**

```python
TOOL_SECURITY_OVERRIDES = {
    "my_tool": [encode_dangerous_tags, strip_hidden_unicode_tags],
}
```

## Adding New Scanners

1. Implement `PromptScanner` interface from `prompt_scanner.py`
1. Add creation logic in `scanner_factory.py`
