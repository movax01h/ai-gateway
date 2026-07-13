# Usage Quota

The usage quota system checks whether a consumer has sufficient GitLab credits before
executing a feature. It is implemented as a decorator (`has_sufficient_usage_quota`) applied at the
route or gRPC method level in two separate services:

- **AI Gateway** (`ai_gateway/api/middleware/route/usage_quota.py`): FastAPI HTTP middleware
- **Duo Workflow Service** (`duo_workflow_service/interceptors/route/usage_quota.py`): gRPC interceptor

Both decorators share the same underlying `UsageQuotaService` and `UsageQuotaClient`
(`lib/usage_quota/`), which communicate with the CustomersDot API to determine whether a
consumer has sufficient credits.

## Architecture

```plaintext
Request
  │
  ▼
has_sufficient_usage_quota (decorator)
  │
  ├─► should_skip_usage_quota_for_user()   ──► skip_usage_cutoff JWT claim present?
  │                                              Yes → bypass check, proceed
  │
  ├─► UsageQuotaService.execute()
  │     │
  │     └─► UsageQuotaClient.check_quota_available()
  │               │
  │               ├─► Client disabled (no credentials)? → allow request
  │               │
  │               ├─► In-memory cache hit? → return cached result
  │               │
  │               ├─► In-flight request for same key? → coalesce, await shared result
  │               │
  │               └─► HEAD /api/v1/consumers/resolve (CustomersDot)
  │                     ├─ 200 → allow, cache with Cache-Control max-age (or 1h fallback)
  │                     ├─ 402 → deny  (InsufficientCredits raised)
  │                     ├─ 403 → deny  (InsufficientEntitlements raised)
  │                     ├─ other HTTP error (4xx/5xx) → deny  (UsageQuotaCheckUnavailable raised)
  │                     └─ timeout, connection failure, or unexpected error → deny  (UsageQuotaCheckUnavailable raised)
  │
  ├─► InsufficientCredits raised
  │     ├─ HTTP: 402 JSON response
  │     └─ gRPC: RESOURCE_EXHAUSTED status
  │
  └─► Proceed to route handler
```

## Caching

Results are cached in-memory using the `Cache-Control: max-age` from the CustomersDot
response, with a 1-hour fallback. When multiple requests arrive simultaneously for the
same consumer before the cache is populated, only one HTTP call is made to CustomersDot
and the rest wait for its result.

## Events

`UsageQuotaEvent` (defined in `lib/usage_quota/service.py`) provides the `event_type`
value sent to CustomersDot as part of the quota check context. It identifies the type of
operation being checked, alongside `feature_qualified_name`:

| Enum value | `event_type` sent to CustomersDot | `feature_qualified_name` | Used by |
|---|---|---|---|
| `DAP_FLOW_ON_EXECUTE` | `duo_agent_platform_workflow_on_execute` | Dynamic — resolved from workflow definition | Duo Workflow `ExecuteWorkflow`, `TrackSelfHostedExecuteWorkflow` |
| `DAP_FLOW_ON_GENERATE_TOKEN` | `duo_agent_platform_workflow_on_generate_token` | Dynamic — resolved from workflow definition (falls back to `dap_feature_legacy`) | Duo Workflow `GenerateToken` |
| `CODE_SUGGESTIONS_CODE_COMPLETIONS` | `code_completions` | `code_suggestions` | Code Suggestions completion endpoints |
| `CODE_SUGGESTIONS_CODE_GENERATIONS` | `code_generations` | `code_suggestions` | Code Suggestions generation endpoints |
| `AMAZON_Q_INTEGRATION` | `amazon_q_integration` | `amazon_q_integration` | Amazon Q endpoints |
| `AIGW_PROXY_USE` | `ai_gateway_proxy_use` | `ai_gateway_proxy_use` | Anthropic, OpenAI, Vertex AI proxy endpoints |

## AI Gateway HTTP decorator

**File**: `ai_gateway/api/middleware/route/usage_quota.py`

### Usage example

Static event (single event type for the route):

```python
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from lib.events import FeatureQualifiedNameStatic
from lib.usage_quota import UsageQuotaEvent

@router.post("/completions")
@has_sufficient_usage_quota(
    feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
    event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
)
async def completions(request: Request, payload: CompletionRequest):
    ...
```

Dynamic event (event type resolved from the request payload):

```python
async def get_event_type(payload: CompletionRequest) -> UsageQuotaEvent:
    if payload.is_generation:
        return UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS
    return UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS

@router.post("/completions")
@has_sufficient_usage_quota(
    feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
    event=get_event_type,
)
async def completions(request: Request, payload: CompletionRequest):
    ...
```

## Duo Workflow Service gRPC decorator

**File**: `duo_workflow_service/interceptors/route/usage_quota.py`

### Usage example

```python
from duo_workflow_service.interceptors.route import has_sufficient_usage_quota
from lib.usage_quota import UsageQuotaEvent

class DuoWorkflowServicer(contract_pb2_grpc.DuoWorkflowServicer):

    @has_sufficient_usage_quota(event=UsageQuotaEvent.DAP_FLOW_ON_EXECUTE)
    async def ExecuteWorkflow(self, request, context):
        ...

    @has_sufficient_usage_quota(event=UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN)
    async def GenerateToken(self, request, context):
        ...
```
