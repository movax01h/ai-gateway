# Billable Usage Technical Overview

AI Gateway emits a structured event to the Data Insights Platform after each billable AI request. Each event captures the feature that ran (Code Suggestions, Duo Chat, Duo Workflow), the user and instance it ran for, and how much was consumed (tokens or requests). CustomersDot processes these events to calculate charges and produce usage analytics.

## `BillingEvent` types

`BillingEvent` is a `StrEnum` defined in `lib/billing_events/client.py`. The event type determines what billing action CustomersDot applies to the target feature. Always pick the value that matches the feature boundary where the LLM call originates.

## `BillingEventContext`

Every billing event carries a `BillingEventContext` — a Pydantic model (`lib/billing_events/context.py`) that is serialized as the Snowplow context payload under the schema `iglu:com.gitlab/billable_usage/jsonschema`. You never construct it directly; `BillingEventsClient.track_billing_event` builds it automatically from two sources:

1. **`current_event_context`** — a `ContextVar[EventContext]` (`lib/internal_events/context.py`) that is populated per-request by the authentication and middleware layers. It carries the ambient request state: realm, instance identifiers, user ID, user type, namespace and project IDs, correlation ID, deployment type, and feature enablement type. Because it is a context variable, it is isolated per async task and requires no explicit passing through call chains.

1. **`CloudConnectorUser`** — passed explicitly to `track_billing_event`. The `unique_instance_id` field on `BillingEventContext` is sourced from `user.claims.gitlab_instance_uid` rather than from `current_event_context`, because the Cloud Connector JWT is the authoritative source for the instance UID.

Key fields to provide when calling `BillingEventsClient` or `BillingEventService`:

- **`unit_of_measure`** — the base unit for the consumption being reported. Use `"request"` for discrete API or workflow calls, and `"tokens"` when reporting actual LLM token counts.
- **`quantity`** — the amount consumed in this unit. Typically `1` for per-request events, or the total token count for token-based events. Must be greater than zero.

## `BillingEventsClient`

`BillingEventsClient` (`lib/billing_events/client.py`) is the low-level Snowplow emitter used in AI Gateway. It takes a `BillingEvent`, assembles a `BillingEventContext` from `current_event_context` and the provided `CloudConnectorUser`, and sends it as a `StructuredEvent` to the Snowplow collector. It also fires an internal `usage_billing_event` for correlation.

Inject it via `ContainerApplication.billing_event.client`. In Duo Workflow Service, use `BillingEventService` instead — it wraps this client and handles LLM operation resolution and self-hosted deployments automatically.

```python
from lib.billing_events import BillingEventsClient, BillingEvent
from dependency_injector.wiring import Provide, inject
from ai_gateway.container import ContainerApplication

@inject
async def my_handler(
    billing_client: BillingEventsClient = Provide[
        ContainerApplication.billing_event.client
    ],
):
    billing_metadata = {
        "execution_environment": "code_generations",
        "llm_operations": get_llm_operations(),
        "feature_qualified_name": gl_event_context.feature_qualified_name,
        "feature_ai_catalog_item": gl_event_context.feature_ai_catalog_item,
    }
    billing_client.track_billing_event(
        user=user,
        event=BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
        category=__name__,
        unit_of_measure="tokens",
        quantity=total_token_count,
        metadata=billing_metadata,
    )
```

> **Note:** `gl_event_context` is a `GLReportingEventContext` (`lib/events/base.py`) built using `GLReportingEventContext.from_static_name(FeatureQualifiedNameStatic.CODE_SUGGESTIONS)`. It exposes `feature_qualified_name`, which CustomersDot uses alongside the event type to identify the target feature and apply the correct billing action.

## `BillingEventService`

`BillingEventService` (`lib/billing_events/service.py`) is the preferred billing entry point in Duo Workflow Service. It wraps `BillingEventsClient` and resolves LLM operation metadata automatically from the request context via `get_llm_operations()` — prefer this path for new implementations. The `llm_ops` parameter is available for cases where automatic resolution is not possible, such as self-hosted deployments. If no operations can be resolved from any source, it raises a `ValueError`.

Inject it via `ContainerApplication.billing_event.service`.

```python
from lib.billing_events import BillingEvent, BillingEventService, ExecutionEnvironment
from dependency_injector.wiring import Provide, inject
from ai_gateway.container import ContainerApplication

@inject
async def my_workflow_handler(
    billing_service: BillingEventService = Provide[
        ContainerApplication.billing_event.service
    ],
):
    billing_service.track_billing(
        workflow_id="wf_123456",
        user=user,
        gl_context=gl_event_context,
        event=BillingEvent.DAP_FLOW_ON_COMPLETION,
        execution_env=ExecutionEnvironment.DAP,
        category=__name__,
        unit_of_measure="request",
        quantity=1,
    )
```

> **Note:** `gl_context` is a `GLReportingEventContext` (`lib/events/base.py`) built from the workflow definition via `GLReportingEventContext.from_workflow_definition(workflow_definition)`. It exposes `feature_qualified_name`, which CustomersDot uses alongside the event type to identify the target feature and apply the correct billing action.

## `BillingEvent` types reference

> **Note:** All new implementations should use `BillingEventService`. Migration from `BillingEventsClient` to `BillingEventService` is in progress.

| `BillingEvent` | String value | What it means | `feature_qualified_name` | Emitter | `unit_of_measure` |
|---|---|---|---|---|---|
| `CODE_SUGGESTIONS_CODE_COMPLETIONS` | `code_completions` | Inline code completion (cursor-triggered). | `code_suggestions` (static) | `BillingEventsClient` | `request` |
| `CODE_SUGGESTIONS_CODE_GENERATIONS` | `code_generations` | Code generation (intent-driven, explicit trigger). Event type resolved dynamically from request payload. | `code_suggestions` (static) | `BillingEventsClient` | `request` |
| `AIGW_PROXY_USE` | `ai_gateway_proxy_use` | Proxied LLM request to an upstream provider (Anthropic, OpenAI, Vertex AI). Covers Duo Chat and other proxy-routed features. | `ai_gateway_proxy_use` (static) | `BillingEventsClient` | `request` |
| `DAP_FLOW_ON_COMPLETION` | `duo_agent_platform_workflow_completion` | DAP workflow reaches a terminal billable state. One event per full workflow execution. | Dynamic — resolved from the workflow definition via `GLReportingEventContext.from_workflow_definition()` (for example, `software_development`, `my_flow/v1`) | `BillingEventService` | `request` |

## Test locally

This setup targets the **Cloud AI Gateway** and covers SaaS and self-managed billing flows.
For self-hosted Duo deployments, see the [Self-Hosted Usage Billing](self-hosted.md) architecture instead.

1. Enable snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Update [the application settings](../application_settings.md#how-to-update-application-settings):

   ```shell
   AIGW_BILLING_EVENT__ENABLED=true
   AIGW_BILLING_EVENT__ENDPOINT=http://127.0.0.1:9091
   AIGW_BILLING_EVENT__BATCH_SIZE=1
   AIGW_BILLING_EVENT__THREAD_COUNT=1
   ```

1. Run snowplow micro with `gdk start snowplow-micro`.
1. Run AI Gateway with `poetry run ai_gateway` or `gdk start gitlab-ai-gateway`, OR
   run Duo Workflow Service with `poetry run duo-workflow-service` or `gdk start duo-workflow-service`.

Visit [the UI dashboard](http://localhost:9091/micro/ui) to see the billing events received by snowplow micro.

## Best Practices

1. **Use the correct BillingEvent enum**: Select the appropriate `BillingEvent` enum value that matches your feature (e.g., `BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS` for code completions)
1. **Include relevant metadata**: Add context that helps with billing analysis (model, feature, language, etc.)
1. **Track accurately**: Ensure quantity reflects actual usage (tokens consumed, requests made, etc.)
1. **Validate inputs**: The client automatically rejects negative or zero quantities and handles missing metadata
1. **Use appropriate categories**: Pass the class name (e.g., `__name__`) where the event is triggered for better debugging
1. **Choose the right unit of measure**: Match the unit to what you're tracking ("tokens" for LLM usage, "request" for API calls, etc.)
