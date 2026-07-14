# Incremental checkpoints

Tracking issue: [issue #596714](https://gitlab.com/gitlab-org/gitlab/-/issues/596714)

## Why

Every workflow step (`aput`) sends the workflow checkpoint to Rails. Sending the **entire** compressed checkpoint each time is wasteful: the conversation grows monotonically, so the bytes per step grow with it — each step re-uploads everything that came before. The only genuinely new information in most steps is the handful of messages appended since the previous checkpoint.

Incremental checkpoints send only the channels that changed, and within those channels, only the newly appended items. This significantly reduces per-step payload size for append-heavy channels such as `conversation_history`, while Rails can still reconstruct full state by replaying the deltas.

## Gating

A per-workflow property gates this feature: `WorkflowConfig.incremental_checkpoints_enabled`, sourced from the workflow's `incrementalCheckpointsEnabled` field (see [`gitlab_api.py`](../duo_workflow_service/gitlab/gitlab_api.py)) and checked in `aput`. Workflows without it enabled receive the legacy payload unchanged. The current rollout is a **shadow write**: `aput` still sends the full `compressed_checkpoint` so reads keep working, and additionally sends the blobs; Rails persists both but reads from the full checkpoint.

The proposed end state ([issue #605653](https://gitlab.com/gitlab-org/gitlab/-/issues/605653)) is to drop the full checkpoint from storage entirely: Rails would keep a slim **header** and reconstruct `channel_values` from the blobs on read. Storage would use two range-partitioned tables:

- `p_duo_workflows_checkpoint_headers` — one row per checkpoint holding only what's needed to rebuild a `CheckpointTuple` and isn't already in the blobs: the langgraph checkpoint minus `channel_values` (`channel_versions`, `versions_seen`, `v`, `ts`, `updated_channels`) plus its `metadata`.

- `p_duo_workflows_checkpoint_blobs` — the channel deltas described below.

Both would partition by a dedicated `workflow_created_at` column (= the workflow's `created_at`, written on every row), so all of a workflow's rows land in one daily partition. An incremental read filters `workflow_id` + `current_thread` with no `created_at` bound; partitioning by `workflow_created_at` lets that read equality-prune to a single partition instead of scanning every retained daily partition. A 30-day TTL would be enforced by dropping old partitions rather than deleting rows.

## Table structure

### Old — full-checkpoint model

One row per checkpoint holds the whole state, partitioned by `created_at`:

```sql
CREATE TABLE p_duo_workflows_checkpoints (
    id                  bigint      NOT NULL,
    workflow_id         bigint      NOT NULL,
    project_id          bigint,                 -- sharding key (exactly one of
    namespace_id        bigint,                 --   project_id / namespace_id)
    created_at          timestamptz NOT NULL,   -- partition key
    updated_at          timestamptz NOT NULL,
    thread_ts           text        NOT NULL,   -- checkpoint id
    parent_ts           text,                   -- parent checkpoint id
    checkpoint          jsonb       NOT NULL,    -- FULL langgraph checkpoint, incl. channel_values
    metadata            jsonb       NOT NULL,
    ui_chat_log         jsonb,                   -- dead (ui_chat_log is a channel)
    current_thread      integer     NOT NULL DEFAULT 0
) PARTITION BY RANGE (created_at);
```

The whole conversation lives in `checkpoint.channel_values` and is rewritten every step. Partitioning by `created_at` means a workflow's checkpoints scatter across daily partitions, so an incremental read (no `created_at` bound) has to touch every retained partition.

### New — header + blobs

`channel_values` moves out to the blobs table; the checkpoint row becomes a slim header. Both tables partition by `workflow_created_at` so a workflow's rows share one partition.

```sql
CREATE TABLE p_duo_workflows_checkpoint_headers (
    id                  bigint      NOT NULL,
    workflow_id         bigint      NOT NULL,
    project_id          bigint,
    namespace_id        bigint,
    workflow_created_at timestamptz NOT NULL,   -- partition key (= workflow.created_at)
    created_at          timestamptz NOT NULL,
    updated_at          timestamptz NOT NULL,
    current_thread      integer     NOT NULL DEFAULT 0,
    checkpoint          jsonb       NOT NULL,    -- HEADER only: channel_versions, versions_seen, v, ts, updated_channels (no channel_values)
    metadata            jsonb       NOT NULL,
    thread_ts           text        NOT NULL,
    parent_ts           text
) PARTITION BY RANGE (workflow_created_at);
```

```sql
CREATE TABLE p_duo_workflows_checkpoint_blobs (
    id                  bigint      NOT NULL,
    workflow_id         bigint      NOT NULL,
    project_id          bigint,
    namespace_id        bigint,
    workflow_created_at timestamptz NOT NULL,   -- partition key (= workflow.created_at)
    created_at          timestamptz NOT NULL,
    updated_at          timestamptz NOT NULL,
    current_thread      integer     NOT NULL DEFAULT 0,   -- group this blob belongs to
    thread_ts           text        NOT NULL,   -- checkpoint id that produced the blob
    channel             text        NOT NULL,   -- e.g. conversation_history
    version             text        NOT NULL,   -- langgraph channel version
    write_type          text        NOT NULL,   -- "json"
    step_action         text        NOT NULL,   -- "conversation" (append) | "compaction" (replace)
    data                bytea       NOT NULL     -- base64-decoded zlib(json), <= 1 MiB
) PARTITION BY RANGE (workflow_created_at);

-- dedup: a re-sent checkpoint must not duplicate blobs
CREATE UNIQUE INDEX idx_duo_wf_checkpoint_blobs_dedup
    ON p_duo_workflows_checkpoint_blobs
    (project_id, workflow_id, thread_ts, channel, version, step_action, workflow_created_at)
    NULLS NOT DISTINCT;
```

Reconstruction on read: start from the header, then fold the group's blobs — the group-start `compaction` snapshots plus its later `conversation` deltas (see [Self-contained groups](#self-contained-groups)). During the shadow-write transition both models are written; readers still use `p_duo_workflows_checkpoints`, and its rows age out via the 30-day TTL.

## Payload

When the property is enabled, the `aput` POST body gains three fields alongside `compressed_checkpoint`: `current_thread`, `channel_blobs`, and `current_thread_started_at`:

```jsonc
{
  "compressed_checkpoint": "...",       // unchanged — full state (shadow write)
  "current_thread": 0,                  // group counter; bumped on compaction / stale-cache reset
  "current_thread_started_at": "<ISO8601 timestamp>",       // ISO8601 start of this group; still sent but no longer read by Rails (see below)
  "channel_blobs": [
    {
      "channel": "conversation_history",
      "version": "3.0",
      "data": "<base64(zlib(json))>",
      "write_type": "json",             // CustomEncoder JSON, matching the header's channel_values
      "step_action": "conversation"     // "conversation" = append delta, "compaction" = full replace
    }
  ]
}
```

Blobs are encoded as `CustomEncoder` JSON (not langgraph's msgpack serde) so their representation matches the header's `channel_values`, which Rails also stores as JSON. Rails can then merge blob deltas onto the header without reimplementing langgraph's msgpack extension types.

`current_thread_started_at` is derived from the group's first checkpoint ID (a time-ordered UUID) and omitted when the ID isn't time-based. It originally bounded the `created_at` range of the blob read query; that role is obsolete now blobs prune by the `workflow_created_at` partition key (see [Gating](#gating)), so the field is still sent but no longer read by Rails. These three (`current_thread`, `channel_blobs`, `current_thread_started_at`) are the complete set of fields incremental mode adds.

## How deltas are computed

`_serialize_channel_blobs` walks `new_versions` (LangGraph's set of channels changed this step) and emits one blob per changed channel:

- **Scalar channels** (`goal`, `project`, …) are skipped — they're tiny, always full replacements, and always recoverable from `compressed_checkpoint`. The exception is `status`, which is always blobbed because it's required for blob reconstruction.

- **List channels** use `_list_delta`: if the previous value is a prefix of the current one, only the appended tail is sent (`step_action="conversation"`); any other change (shrink, reorder, in-place edit) sends the full list (`step_action="compaction"`).

- **Dict-of-list channels** (e.g. `conversation_history`, keyed by agent) use `_dict_of_list_delta`: per-key appends are sent as a per-key dict of new items; if any key's list shrank or its prefix changed, the whole dict is sent as a compaction.

`step_action` is the authoritative append-vs-replace signal for Rails. `current_thread` is a grouping hint that increments on compaction so Rails restarts reconstruction from that checkpoint rather than replaying deltas across the boundary — it cannot be authoritative because it resets to `0` on gateway restart, whereas `step_action` is derived from the channel values and stays correct.

## Self-contained groups

The per-step deltas above describe only the channels that changed. That's enough while Rails overlays deltas on the full-checkpoint header, but the end state drops the header (see [Gating](#gating)), and then a channel that hasn't changed since a previous `current_thread` group would have no base to fold onto — its history lived in the old group.

To make each group reconstruct on its own, the **start of every group** re-seeds _all_ reconstructable channels as full `compaction` snapshots via `_serialize_all_channels_full`, replacing the per-channel deltas for that one step. A group starts on:

- the workflow's first checkpoint (`_prev_checkpoint_id is None`),

- a stale-cache reset (`force_rewrite`), or

- a compaction.

`_serialize_all_channels_full` mirrors `_serialize_channel_blobs`'s channel selection (list and dict channels plus the `status` scalar; other scalars are dropped as recoverable) and its JSON encoding, but takes versions from the checkpoint's `channel_versions` rather than `new_versions`, since unchanged channels must be re-seeded too. Reconstruction of a group is then: group-start full snapshots + the group's later `conversation` deltas.

The group boundary is keyed on `_prev_checkpoint_id`, **not** on `current_thread_started_at`. The started-at marker is `None` for checkpoint IDs that aren't time-based, which would otherwise re-seed every step. Outside a group boundary, behavior is the per-channel deltas described above.

## State across `aput` calls

To compute deltas, `GitLabWorkflow` caches `_prev_channel_values`, `_prev_checkpoint_id`, `_current_thread`, and `_current_thread_started_at` between calls. On each `aput`:

1. **Stale-cache detection** — if the incoming parent `checkpoint_id` doesn't match `_prev_checkpoint_id`, a checkpoint was missed. The cache is reset and every changed channel is serialized as a full replacement (`force_rewrite`).

1. At a group start (first checkpoint, stale-cache reset, or compaction), the per-channel blobs are replaced with a full re-seed of all channels (see [Self-contained groups](#self-contained-groups)).

1. If the step was a compaction (or a stale-cache rewrite), `_current_thread` is bumped. `_current_thread_started_at` is pinned to this checkpoint's start time whenever a new group starts — on compaction, a stale-cache reset, or the first checkpoint (marker still unset).

1. The cache is updated and `channel_blobs`, `current_thread`, and `current_thread_started_at` are attached to the payload.

## Surviving a gateway restart

The cache is in-memory, so a restart (or pickup by another gateway instance) would reset it and cause a spurious rewrite or a `current_thread` mismatch. `_hydrate_incremental_state` restores the cached fields from a fetched checkpoint on every fetch path (REST, GraphQL latest, and latest-fetch). It accepts both `current_thread` (REST) and `currentThread` (GraphQL), restores `_current_thread_started_at` from the checkpoint's decoded `current_thread_started_at` so a post-restart write doesn't re-pin the marker to a mid-group time and drop the group's earlier blobs, and tolerates fields being absent (older Rails) or malformed by keeping the defaults. After a restart, a fetch followed by a write reuses the server's `current_thread` and emits a correct delta with no spurious bump.

## Monitoring which strategy is in use

Each `aput` tags its strategy (`incremental` or `full`) in two places so both sides are searchable in Kibana:

- **Request path**: the checkpoint POST carries a `?checkpoint_strategy=` query parameter. Rails doesn't read it (it's ignored by strong parameters), but it appears in Rails/workhorse request logs, so the backend side is searchable without any Rails change.

- **Gateway log**: the `Checkpoint saved` log line includes a `checkpoint_strategy` field.
