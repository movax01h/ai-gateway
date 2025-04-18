# Internal Event Tracking

To collect product usage metrics, use [`InternalEventsClient`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/ai_gateway/internal_events/client.py) in AI Gateway.
This is a Python client for the [GitLab Internal Event Tracking](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html) system.

Previously, we were using [`SnowplowInstrumentator`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/ai_gateway/tracking/snowplow.py) for tracking Code Suggestion events, however, this instrumentator is deprecated since it's hard to extend for various events.
See [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561) for migrating from `SnowplowInstrumentator` to `InternalEventsClient`.

## Trigger events

To trigger an event, call the `track_event` method of the `InternalEventsClient` object with the desired arguments:

```python
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.async_dependency_resolver get_internal_event_client

@router.post("/awesome_feature")
async def awesome_feature(
    request: Request,
    internal_event_client: Annotated[InternalEventsClient, Depends(get_internal_event_client)],
):
    # Send "request_awesome_feature" event to Snowplow.
    internal_event_client.track_event("request_awesome_feature")
```

Additional properties can be passed when tracking events. They can be used to save additional data related to a given event.

Snowplow has built-in properties with keys `label` (string), `property` (string), and `value` (numeric). It's recommended to use these properties first. If you need to pass more properties, you can send custom key-value pairs. For example:

```python
from ai_gateway.internal_events.context import InternalEventAdditionalProperties
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

There are various arguments you can set aside from the event name.
See [this section](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#trigger-events) for more information.

## Test locally

1. Enable snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Update [the application settings](application_settings.md#how-to-update-application-settings):

   ```shell
   AIGW_INTERNAL_EVENT__ENABLED=true
   AIGW_INTERNAL_EVENT__ENDPOINT=http://127.0.0.1:9091
   AIGW_INTERNAL_EVENT__BATCH_SIZE=1
   AIGW_INTERNAL_EVENT__THREAD_COUNT=1
   ```

1. Run snowplow micro with `gdk start snowplow-micro`.
1. Run AI Gateway with `poetry run ai_gateway`.

Visit [the UI dashboard](http://127.0.0.1:9091) to see the events received by snowplow micro.

## Configuration

There are various configuration options for the Internal Event Tracking.
See `AIGW_INTERNAL_EVENT` prefixed variables in the [application settings](application_settings.md#how-to-update-application-settings).

## Internal Event Middleware

Some of the fundamental event arguments are collected at `InternalEventMiddleware` and set to all events automatically.

## Adding New Events

If you are creating any new events, please create an event definition in the  `config/events` folder. This will help to discover which events are being tracked in [Metric dictionary](https://metrics.gitlab.com/events)

If we are updating any existing events with new parameters like adding `label`,`property` or `value` we should document it in the existing event definition file as well. Follow event definition [guide](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/event_definition_guide.html) to structure the event definition file.

## Internal Event Tracking for Duo Workflow Service

To collect product usage metrics, use [`InternalEventsClient`](https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-service/-/blob/main/duo_workflow_service/internal_events/client.py) in Duo Workflow Service.
This is a Python client for the [GitLab Internal Event Tracking](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html) system.

### Trigger events

To trigger an event, call the `track_event` method of the `InternalEventsClient` object with the desired arguments:

```python
from duo_workflow_service.internal_events import DuoWorkflowInternalEvent

# Send "request_duo_workflow" event to Snowplow.
DuoWorkflowInternalEvent.track_event(event_name="request_duo_workflow")
```

Additional properties can be passed when tracking events. They can be used to save additional data related to a given event.

Snowplow has built-in properties with keys `label` (string), `property` (string), and `value` (numeric). It's recommended to use these properties first. If you need to pass more properties, you can send custom key-value pairs. For example:

```python
from duo_workflow_service.internal_events.context import InternalEventAdditionalProperties
from duo_workflow_service.internal_events import DuoWorkflowInternalEvent

# Send "request_duo_workflow" event to Snowplow with additional properties.
additional_properties = InternalEventAdditionalProperties(
    label="completion_event", property="workflow_id", value=1, total_tokens=20
)
DuoWorkflowInternalEvent.track_event(
    event_name="request_duo_workflow",
    additional_properties=additional_properties,
)
```

There is another parameter called `category` where we capture where the event happened. We should pass the name of the class where the event happened.

```python
from duo_workflow_service.internal_events import DuoWorkflowInternalEvent

DuoWorkflowInternalEvent.track_event(
    "request_duo_workflow",
    category=__name__,
)
```

There are various arguments you can set aside from the event name.
See [this section](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#trigger-events) for more information.

### Test locally

1. Enable Snowplow micro in GDK with [these instructions](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/local_setup_and_debugging.html#snowplow-micro).
1. Add below env variables to `.env`:

   ```shell
   DW_INTERNAL_EVENT__ENABLED=true
   DW_INTERNAL_EVENT__ENDPOINT=http://gdk.test:9091
   DW_INTERNAL_EVENT__APP_ID=gitlab_duo_workflow
   DW_INTERNAL_EVENT__BATCH_SIZE=1
   DW_INTERNAL_EVENT__THREAD_COUNT=1
   DUO_WORKFLOW_SERVICE_ENVIRONMENT="development"
   ```

1. Run Snowplow micro with `gdk start snowplow-micro`.
1. Run service with `poetry run duo-workflow-service`.

Visit [the UI dashboard](http://gdk.test:9091) to see the events received by snowplow micro.

### Configuration

There are various configuration options for the Internal Event Tracking.
See `DW_INTERNAL_EVENT` prefixed variables in the env variables settings.

```plaintext
DW_INTERNAL_EVENT__ENABLED # to enable/disable internal events tracking
DW_INTERNAL_EVENT__ENDPOINT # snowplow event collector url
DW_INTERNAL_EVENT__APP_ID # gitlab application id
DW_INTERNAL_EVENT__BATCH_SIZE # batch size for sending events to the collector
DW_INTERNAL_EVENT__THREAD_COUNT # number of event sending threads.
DUO_WORKFLOW_SERVICE_ENVIRONMENT # service environment
```

### Internal Event Interceptor

Some of the fundamental event arguments are collected at `InternalEventsInterceptor` and set to all events automatically.

### Adding New Events

If you are creating any new events, please create an event definition in the `duo_workflow_service/config/events` folder. This documents which events are being tracked in the [Metric dictionary](https://metrics.gitlab.com/events).

If you are updating any existing events with new parameters like adding `label`,`property` or `value` you should document it in the existing event definition file as well. Follow event definition [guide](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/event_definition_guide.html) to structure the event definition file.
