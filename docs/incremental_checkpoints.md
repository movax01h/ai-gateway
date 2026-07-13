# Incremental checkpoints

Tracking issue: [issue #596714](https://gitlab.com/gitlab-org/gitlab/-/issues/596714)

## Why

Every workflow step (`aput`) sends the workflow checkpoint to Rails. Sending the
**entire** compressed checkpoint each time is wasteful: the conversation grows
monotonically, so the bytes per step grow with it — each step re-uploads everything that
came before. The only genuinely new information in most steps is the handful of messages
appended since the previous checkpoint.

Incremental checkpoints send only the channels that changed, and within those channels,
only the newly appended items. This significantly reduces per-step payload size for
append-heavy channels such as `conversation_history`, while Rails can still reconstruct
full state by replaying the deltas.

## Capability gating

The `incremental_checkpoints` client capability gates this feature
(`is_client_capable("incremental_checkpoints")`, see
[`client_capabilities.py`](../duo_workflow_service/client_capabilities.py)). Clients that
don't declare it receive the legacy payload unchanged. The current rollout is a **shadow
write**: `aput` still sends the full `compressed_checkpoint` so reads keep working, and
additionally sends the deltas; Rails persists both but reads from the full checkpoint.

## Payload

When the capability is enabled, the `aput` POST body gains `current_thread` and
`channel_blobs` alongside `compressed_checkpoint`:

```jsonc
{
  "compressed_checkpoint": "...",      // unchanged — full state (shadow write)
  "current_thread": 0,                  // generation counter; bumped on compaction
  "channel_blobs": [
    {
      "channel": "conversation_history",
      "version": "3.0",
      "data": "<base64(zlib(serialized))>",
      "write_type": "msgpack",
      "step_action": "conversation"     // "conversation" = append delta, "compaction" = full replace
    }
  ]
}
```

## How deltas are computed

`_serialize_channel_blobs` walks `new_versions` (LangGraph's set of channels changed this
step) and emits one blob per changed channel:

- **Scalar channels** (`goal`, `project`, …) are skipped — they're tiny, always
  full replacements, and always recoverable from `compressed_checkpoint`. The exception is
  `status`, which is always blobbed because it's required for blob reconstruction.
- **List channels** use `_list_delta`: if the previous value is a prefix of the current
  one, only the appended tail is sent (`step_action="conversation"`); any other change
  (shrink, reorder, in-place edit) sends the full list (`step_action="compaction"`).
- **Dict-of-list channels** (e.g. `conversation_history`, keyed by agent) use
  `_dict_of_list_delta`: per-key appends are sent as a per-key dict of new items; if any
  key's list shrank or its prefix changed, the whole dict is sent as a compaction.

`step_action` is the authoritative append-vs-replace signal for Rails. `current_thread` is
a grouping hint that increments on compaction so Rails restarts reconstruction from that
checkpoint rather than replaying deltas across the boundary — it cannot be authoritative
because it resets to `0` on gateway restart, whereas `step_action` is derived from the
channel values and stays correct.

## State across `aput` calls

To compute deltas, `GitLabWorkflow` caches `_prev_channel_values`, `_prev_checkpoint_id`,
and `_current_thread` between calls. On each `aput`:

1. **Stale-cache detection** — if the incoming parent `checkpoint_id` doesn't match
   `_prev_checkpoint_id`, a checkpoint was missed. The cache is reset and every changed
   channel is serialized as a full replacement (`force_rewrite`).
1. If the step was a compaction (or a stale-cache rewrite), `_current_thread` is bumped.
1. The cache is updated and `channel_blobs` + `current_thread` are attached to the payload.

## Surviving a gateway restart

The cache is in-memory, so a restart (or pickup by another gateway instance) would reset
it and cause a spurious rewrite or a `current_thread` mismatch. `_hydrate_incremental_state`
restores the three cached fields from a fetched checkpoint on every fetch path (REST,
GraphQL latest, and latest-fetch). It accepts both `current_thread` (REST) and
`currentThread` (GraphQL), and tolerates the field being absent (older Rails) or malformed
by keeping the default. After a restart, a fetch followed by a write reuses the server's
`current_thread` and emits a correct delta with no spurious bump.

## Monitoring which strategy is in use

Each `aput` tags its strategy (`incremental` or `full`) in two places so both sides are
searchable in Kibana:

- **Request path**: the checkpoint POST carries a `?checkpoint_strategy=` query parameter.
  Rails doesn't read it (it's ignored by strong parameters), but it appears in
  Rails/workhorse request logs, so the backend side is searchable without any Rails change.
- **Gateway log**: the `Checkpoint saved` log line includes a `checkpoint_strategy` field.
