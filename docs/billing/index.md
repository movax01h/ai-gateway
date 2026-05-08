# Billing

AI Gateway tracks billable consumption of AI features (tokens, requests, executions) across
all GitLab deployment types and reports it to the Data Insights Platform for invoicing and
analytics. Billing events are emitted exclusively by the **Cloud AI Gateway** — the instance
owned and operated by GitLab, as opposed to self-hosted instances running on customer
infrastructure.

## End-to-end flow

Every billable request passes through two stages:

```plaintext
Incoming request
  │
  ▼
1. Pre-request: has_sufficient_usage_quota (decorator)
     │
     ├─► CustomersDot quota check (HEAD /api/v1/consumers/resolve)
     │     ├─ 200 → allow, cache result
     │     ├─ 402 → deny (HTTP 402 / gRPC RESOURCE_EXHAUSTED)
     │     └─ error → fail-open, allow request
     │
     ▼
   Route handler executes (LLM call, workflow, etc.)
     │
     ▼
2. Post-request: BillingEventsClient / BillingEventService
     │
     ├─ Collect LLM operations (token counts, model metadata)
     ├─ Build BillingEventContext from current_event_context
     └─ Emit StructuredEvent to Snowplow → Data Insights Platform
          └─ Also fires internal `usage_billing_event` for correlation
```

## Key components

**Billing events** (`lib/billing_events/`):

- **`BillingEventsClient`** (`client.py`) — low-level Snowplow emitter; builds and sends `BillingEventContext`
- **`BillingEventService`** (`service.py`) — higher-level service; resolves LLM operations and wraps the client
- **`BillingEvent`** (`client.py`) — enum of all billable event types
- **`BillingEventContext`** (`context.py`) — Snowplow context schema ([iglu schemas](https://gitlab.com/gitlab-org/iglu/-/tree/master/public/schemas/com.gitlab/billable_usage/jsonschema))
- **`ContainerBillingEvent`** (`container.py`) — DI container wiring client and service

**Usage quota** (`lib/usage_quota/`):

- **`has_sufficient_usage_quota`** — pre-request quota decorator; HTTP in `ai_gateway/api/middleware/route/`, gRPC in `duo_workflow_service/interceptors/route/`
- **`UsageQuotaService` / `UsageQuotaClient`** — shared logic that calls CustomersDot to check available credits

## Deployment types

Billing behavior differs by deployment type:

| Deployment | LLM operation source | Notes |
|---|---|---|
| SaaS | `get_llm_operations()` context variable | Real token counts and model metadata from LLM response |
| Self-managed | `get_llm_operations()` context variable | Same as SaaS |
| Self-hosted Duo | `SelfHostedLLMOperations.get_operations()` | LLM runs on customer infrastructure; flat-rate pricing — actual token counts are unavailable, so standardized placeholder values are used |

See [Further reading](#further-reading) for links to each deployment type's documentation.

## Configuration at a glance

**Billing events** (`AIGW_BILLING_EVENT__*` in `.env`):

- `AIGW_BILLING_EVENT__ENABLED` — enable or disable billing event emission
- `AIGW_BILLING_EVENT__ENDPOINT` — Snowplow collector endpoint URL
- `AIGW_BILLING_EVENT__BATCH_SIZE` — events per batch (set to `1` for local testing)
- `AIGW_BILLING_EVENT__THREAD_COUNT` — number of async emitter worker threads

**Usage quota** (set in `.env`, no `AIGW_` prefix):

- `CUSTOMER_PORTAL_USAGE_QUOTA_API_USER` — CustomersDot API username
- `CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN` — CustomersDot API token

See [Application Settings](../application_settings.md) for the full configuration reference.

## Further reading

- [Technical Overview](technical_overview.md): `BillingEventsClient` and `BillingEventService` API reference, `BillingEvent` enum, event types table, local testing setup, and best practices.
- [Self-Hosted DAP Usage Billing](self-hosted-dap.md): Architecture for billing in self-hosted Duo deployments.
- [Usage Quota](../usage_quota.md): Per-consumer credit checks before billable operations execute, `UsageQuotaEvent` enum, and decorator usage for HTTP and gRPC.
- [Internal Events](../internal_events.md): `InternalEventsClient` used for `usage_billing_event` correlation tracking.
