# Billing Event Tracking

To collect billable usage events, use [`BillingEventsClient`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/lib/billing_events/client.py) in AI Gateway and Duo Workflow Service.
This is a Python client for tracking GitLab billable usage events that are sent to the Data Insights Platform for billing and usage analytics.

The billing events system tracks consumption of AI features (tokens, requests, executions) to provide accurate billing data for GitLab AI-powered features like Code Suggestions, Duo Chat, and Duo Workflow.

## Trigger events

To trigger a billing event, call the `track_billing_event` method of the `BillingEventsClient` object with the desired arguments:

```python
from lib.billing_events import BillingEventsClient
from dependency_injector.wiring import Provide, inject
from ai_gateway.container import ContainerApplication

@inject
async def ai_completion_feature(
    billing_event_client: BillingEventsClient = Provide[
        ContainerApplication.billing_event.client
    ],
):

   billable_client.track_billing_event(
        user=user,
        event_type="billable_event_type",  # → action + event_type
        category=__name__,                 # → where event happened
        metadata={                         # → billable context metadata
            "workflow_id": "wf_123456",
            "execution_environment": "ci_pipeline",
            "resource_consumption": {
                "cpu_seconds": 324.5,
                "memory_mb_seconds": 1024.7,
                "storage_operations": 55
            },
            "llm_operations": {
                "token_count": 4328,
                "model_id": "claude-3-sonnet-20240229",
                "prompt_tokens": 3150,
                "completion_tokens": 2178
            },
            "commands_executed": 17
        },
        unit_of_measure="request",
        quantity=2300,
    )

```

## Event Parameters

The `track_billing_event` method accepts the following parameters:

- **`event_type`** (required): The type of billable event (e.g., "ai_completion", "code_suggestions", "duo_chat")
- **`unit_of_measure`** (optional, default: "request"): The unit used for measurement ("token", "request", "second", "message", "execution") Used for accurate unit conversion and billing calculations.
- **`quantity`** (required): The quantity of usage for this billing record
- **`metadata`** (optional): Dictionary containing additional billing-related data
- **`category`** (optional, default: "default_category"): The location/class where the billing event occurred

## Billing Event Context

Each billing event automatically includes context from the current request through the `current_event_context`. The following fields are automatically populated:

- **Event identification**: `event_id` (unique UUID), `timestamp`
- **Instance information**: `instance_id`, `host_name`, `realm`
- **User context**: `subject` (user ID), `seat_ids`
- **Project context**: `project_id`, `namespace_id`, `root_namespace_id`
- **Request correlation**: `correlation_id`

## Data Schema

Billing events follow the GitLab billable usage schema (`iglu:com.gitlab/billable_usage/jsonschema/1-0-1`):

```python
class BillingEventContext:
    event_id: str                    # Unique event identifier
    event_type: str                  # Type of billable event
    unit_of_measure: str             # Measurement unit
    quantity: float                  # Usage quantity
    timestamp: str                   # ISO timestamp
    realm: str
    instance_id: Optional[str]       # GitLab instance ID
    unique_instance_id: Optional[str] # Unique instance identifier
    host_name: Optional[str]         # Host name
    project_id: Optional[int]        # Project ID
    namespace_id: Optional[int]      # Namespace ID
    subject: Optional[str]           # User subject
    root_namespace_id: Optional[int] # Root namespace ID
    correlation_id: Optional[str]    # Request correlation ID
    seat_ids: Optional[List[str]]    # Associated seat IDs
    metadata: Optional[Dict[str, Any]] # Additional metadata
```

## Test locally

1. Enable snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Update [the application settings](application_settings.md#how-to-update-application-settings):

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

## Configuration

Various configuration options are available for Billing Event Tracking.
See `AIGW_BILLING_EVENT` prefixed variables in the [application settings](application_settings.md#how-to-update-application-settings).

Key configuration options:

- `AIGW_BILLING_EVENT__ENABLED`: Enable/disable billing event tracking
- `AIGW_BILLING_EVENT__ENDPOINT`: Snowplow collector endpoint
- `AIGW_BILLING_EVENT__BATCH_SIZE`: Number of events per batch (set to 1 for local testing)
- `AIGW_BILLING_EVENT__THREAD_COUNT`: Number of worker threads

When testing locally, make sure that `AIGW_BILLING_EVENT__BATCH_SIZE` is `1`.
Otherwise, no events are visible in the UI until the batch size has been reached.

## Best Practices

1. **Use descriptive event types**: Choose clear, consistent names like "ai_completion", "code_suggestions", "duo_chat"
1. **Include relevant metadata**: Add context that helps with billing analysis (model, feature, language, etc.)
1. **Track accurately**: Ensure quantity reflects actual usage (tokens consumed, requests made, etc.)
1. **Validate inputs**: The client automatically rejects negative quantities and handles missing metadata
1. **Use appropriate categories**: Pass the class name where the event is triggered for better debugging
