# Internal Event Tracking

To collect product usage metrics, use [`InternalEventsClient`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/lib/internal_events/client.py) in AI Gateway and Duo Workflow Service.
This is a Python client for the [GitLab Internal Event Tracking](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html) system.

Previously, we were using [`SnowplowInstrumentator`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/ai_gateway/tracking/snowplow.py) for tracking Code Suggestion events, however, this instrumentator is deprecated since it's hard to extend for various events.
See [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561) for migrating from `SnowplowInstrumentator` to `InternalEventsClient`.

## Trigger events

To trigger an event, call the `track_event` method of the `InternalEventsClient` object with the desired arguments:

```python
from lib.internal_events import InternalEventsClient
from dependency_injector.wiring import Provide, inject
from ai_gateway.container import ContainerApplication

@inject
async def awesome_feature(
    internal_event_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
):
    # Send "request_awesome_feature" event to Snowplow.
    internal_event_client.track_event("request_awesome_feature")
```

Additional properties can be passed when tracking events. They can be used to save additional data related to a given event.

Snowplow has built-in properties with keys `label` (string), `property` (string), and `value` (numeric). It's recommended to use these properties first. If you need to pass more properties, you can send custom key-value pairs. For example:

```python
from lib.internal_events.context import InternalEventAdditionalProperties
...
# Send "request_awesome_feature" event to Snowplow with additional properties.
additional_properties = InternalEventAdditionalProperties(
    label="completion_event", property="property_value", value=1, key="value"
)
internal_event_client.track_event(
    event_name="request_awesome_feature",
    additional_properties=additional_properties,
)
```

There is another parameter called `category` where we capture where the event happened. We should pass the name of the class where the event happened.

```python
internal_event_client.track_event(
    f"request_{path_unit_primitive_map[chat_invokable]}",
    category=__name__,
)
```

### AI context

When tracking AI-related events, you can provide AI-specific metadata using the `ai_context` parameter. This is the **preferred approach** for new code as it provides type safety and clarity.

#### Using explicit AIContext (recommended)

Pass an `AIContext` object to explicitly set workflow, agent, and token metadata:

```python
from lib.internal_events import AIContext, InternalEventsClient

internal_event_client.track_event(
    event_name="workflow_route_decision",
    ai_context=AIContext(
        workflow_id="wf-123",
        flow_type="fix_pipeline",
        agent_name="supervisor",
    ),
    category="Router",
)
```

For events with token usage:

```python
ai_context = AIContext(
    workflow_id="wf-456",
    flow_type="code_review",
    input_tokens=1500,
    output_tokens=800,
    total_tokens=2300,
    cache_read=200,
    cache_creation=100,
)

internal_event_client.track_event(
    event_name="token_usage_completion",
    ai_context=ai_context,
)
```

#### Implicit extraction (legacy)

For backwards compatibility, `track_event` still extracts AI context from `additional_properties.extra` and `**kwargs` when `ai_context` is not provided:

```python
# Legacy approach - still works but not recommended for new code
additional_properties = InternalEventAdditionalProperties(
    label="completion_event",
    workflow_id="wf-789",
    flow_type="chat",
    agent_name="assistant",
)

internal_event_client.track_event(
    event_name="request_completion",
    additional_properties=additional_properties,
    input_tokens=500,
    output_tokens=300,
)
```

#### Precedence rules

When both explicit `ai_context` and implicit values are present, the explicit `ai_context` takes precedence for all its fields:

- `workflow_id`, `flow_type`, `agent_name` from `AIContext` override values in `additional_properties.extra`
- Token fields (`input_tokens`, `output_tokens`, `total_tokens`, `cache_read`, `cache_creation`, `ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`) from `AIContext` override values from `**kwargs`
- `session_id` is always derived from `additional_properties.value`, regardless of `ai_context`

Various arguments can be set aside from the event name.
See [this section](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#trigger-events) for more information.

## Unit tests

Whenever you add a new internal event, make sure to include a corresponding test in the MR. The tracking can be tested as follows:

```python
from unittest.mock import Mock
from lib.internal_events import InternalEventAdditionalProperties


def test_track_internal_event(internal_event_client: Mock):
    additional_properties = InternalEventAdditionalProperties(label="event_label")

    instance = YourClass()
    instance._internal_event_client = internal_event_client

    instance.trigger_action()

    internal_event_client.track_event.assert_called_once_with(
        event_name="trigger_action",
        additional_properties=additional_properties,
        category="Instance",
    )
```

## Test locally

1. Enable snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Update [the application settings](application_settings.md#how-to-update-application-settings):

   ```shell
   AIGW_INTERNAL_EVENT__ENABLED=true
   AIGW_INTERNAL_EVENT__ENDPOINT=http://127.0.0.1:9091
   AIGW_INTERNAL_EVENT__BATCH_SIZE=1
   AIGW_INTERNAL_EVENT__THREAD_COUNT=5
   ```

1. Run snowplow micro with `gdk start snowplow-micro`.
1. Run AI Gateway with `poetry run ai_gateway` or `gdk start gitlab-ai-gateway`, OR
   run Duo Workflow Service with `poetry run duo-workflow-service` or `gdk start duo-workflow-service`.

Visit [the UI dashboard](http://localhost:9091/micro/ui) to see the events received by snowplow micro.

## Configuration

Various configuration options are available for the Internal Event Tracking.
See `AIGW_INTERNAL_EVENT` prefixed variables in the [application settings](application_settings.md#how-to-update-application-settings).

When testing locally, make sure that `AIGW_INTERNAL_EVENT__BATCH_SIZE` is `1`.
Otherwise, no events are visible in the UI until the batch size has been reached.

## Internal Event Middleware

Some of the fundamental event arguments are collected at `InternalEventMiddleware` and set to all events automatically.

## Adding New Events

If you are creating any new events, please create an event definition in the `config/events` folder. This will help to discover which events are being tracked in [Metric dictionary](https://metrics.gitlab.com/events).

If we are updating any existing events with new parameters like adding `label`,`property` or `value` we should document it in the existing event definition file as well. Follow event definition [guide](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/event_definition_guide.html) to structure the event definition file.
