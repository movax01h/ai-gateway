# Incremental checkpoints

Tracking issue: [issue #596714](https://gitlab.com/gitlab-org/gitlab/-/issues/596714)

## Why

Every workflow step (`aput`) sends the workflow checkpoint to Rails. Sending the **entire** compressed checkpoint each time is wasteful: the conversation grows monotonically, so the bytes per step grow with it — each step re-uploads everything that came before. The only genuinely new information in most steps is the handful of messages appended since the previous checkpoint.

Incremental checkpoints send only the channels that changed, and within those channels, only the newly appended items. This significantly reduces per-step payload size for append-heavy channels such as `conversation_history`, while Rails can still reconstruct full state by replaying the deltas.

## Gating

A per-workflow property gates this feature: `WorkflowConfig.incremental_checkpoints_enabled`, sourced from the workflow's `incrementalCheckpointsEnabled` field (see [`gitlab_api.py`](../duo_workflow_service/gitlab/gitlab_api.py)) and checked in `aput`. Workflows without it enabled receive the legacy payload unchanged. The current rollout is a **shadow write**: `aput` still sends the full `compressed_checkpoint` so reads keep working, and additionally sends the blobs; Rails persists both, and reads the header and blobs for any workflow with the column set.

A second gate layers on top: the `incremental_checkpoints_only` server capability, which says the instance stores only the header and the blobs. `aput` then sends the checkpoint skeleton (the langgraph checkpoint minus `channel_values`) instead of `compressed_checkpoint`. Rails advertises the capability from `duo_workflow_write_incremental_only`, workhorse appends it to `clientCapabilities`, and `is_client_capable` reads it here. An instance too old to advertise it keeps receiving the full payload, and Rails accepts both shapes, so a mid-session change can't break a write.

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
    channel_keys        text[],                  -- live channel membership; filters the blob fold (max 100 entries)
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

`current_thread_started_at` is derived from the group's first checkpoint ID (a time-ordered UUID) and omitted when the ID isn't time-based. It originally bounded the `created_at` range of the blob read query; that role is obsolete now blobs prune by the `workflow_created_at` partition key (see [Gating](#gating)), so the field is still sent but no longer read by Rails. These three (`current_thread`, `channel_blobs`, `current_thread_started_at`) are the complete set of fields incremental mode adds; the incremental-only variant below adds one more.

When the instance also advertises `incremental_checkpoints_only`, `checkpoint` replaces `compressed_checkpoint`: the langgraph checkpoint minus `channel_values` (the same skeleton stored as the header). A `channel_keys` field is added. `current_thread`, `current_thread_started_at`, and `channel_blobs` are unchanged:

```jsonc
{
  "checkpoint": {                       // skeleton — no channel_values
    "id": "...",
    "ts": "...",
    "v": 1,
    "channel_versions": {},
    "versions_seen": {},
    "updated_channels": []
  },
  "channel_keys": ["conversation_history", "status"],   // live channel membership
  "current_thread": 0,
  "current_thread_started_at": "<ISO8601 timestamp>",
  "channel_blobs": [ /* unchanged — see above */ ]
}
```

`channel_keys` is `channel_values.keys()`, captured before the skeleton drops the values. Rails stores it in `channel_keys` on `p_duo_workflows_checkpoint_headers` and uses it to select the live channels from the blob fold ([GitLab issue 613975](https://gitlab.com/gitlab-org/gitlab/-/issues/613975)). Blobs are an append-only log, so the fold is the union of every channel ever written and cannot express a deletion. Rails normally derives the list from `channel_values`, which this mode does not send, and it cannot rebuild it from the rest of the payload: `channel_versions` keeps a channel that was consumed after its value is gone, and each step blobs only the channels that changed. An instance too old to accept the field ignores it.

This shape is tagged with the query parameter `checkpoint_strategy=incremental_only` (see [Monitoring which strategy is in use](#monitoring-which-strategy-is-in-use)).

## How deltas are computed

`_serialize_channel_blobs` walks `new_versions` (LangGraph's set of channels changed this step) and emits one blob per changed channel:

- **Scalar channels** (`status`, `goal`, `branch:to:*`, …) are blobbed as full values, since they have no delta form. No channel is filtered by type: once the header is dropped (see [Gating](#gating)) a skipped scalar is simply lost, and `branch:to:*` — LangGraph's run queue — decides which tasks a resumed graph runs. Because blobs are append-only, the read path must select only the channels the checkpoint header declares live, or deleted channels linger in the reconstructed state ([GitLab issue 613956](https://gitlab.com/gitlab-org/gitlab/-/issues/613956)).

- **List channels** use `_list_delta`: if the previous value is a prefix of the current one, only the appended tail is sent (`step_action="conversation"`); any other change (shrink, reorder, in-place edit) sends the full list (`step_action="compaction"`).

- **Dict channels** (e.g. `conversation_history` keyed by agent, or `last_human_input` holding an events-API payload) use `_dict_of_list_delta`: per-key appends are sent as a per-key dict of new items; if any key's list shrank or its prefix changed, the whole dict is sent as a compaction.

  A **dropped key** also forces the full dict. Rails' `ChannelValuesReconstructor#append` writes only the keys the delta carries, so a per-key delta cannot say "this key is gone" and the reader would keep it forever. Agent-keyed channels only gain keys, but a wholesale-replaced payload like `last_human_input` can lose one between events.

`step_action` is the authoritative append-vs-replace signal for Rails. `current_thread` is a grouping hint that increments on compaction so Rails restarts reconstruction from that checkpoint rather than replaying deltas across the boundary — it cannot be authoritative because it resets to `0` on gateway restart, whereas `step_action` is derived from the channel values and stays correct.

## Self-contained groups

The per-step deltas above describe only the channels that changed. That's enough while Rails overlays deltas on the full-checkpoint header, but the end state drops the header (see [Gating](#gating)), and then a channel that hasn't changed since a previous `current_thread` group would have no base to fold onto — its history lived in the old group.

To make each group reconstruct on its own, the **start of every group** re-seeds _all_ channels as full `compaction` snapshots via `_serialize_all_channels_full`, replacing the per-channel deltas for that one step. A group starts on:

- the lineage's first checkpoint (`state.prev_checkpoint_id is None`),

- a stale-cache reset (`force_rewrite`), or

- a compaction.

`_serialize_all_channels_full` mirrors `_serialize_channel_blobs`'s JSON encoding and also covers every channel, but takes versions from the checkpoint's `channel_versions` rather than `new_versions`, since unchanged channels must be re-seeded too. Reconstruction of a group is then: group-start full snapshots + the group's later `conversation` deltas.

`ui_chat_log` is trimmed at each compaction group start, keeping only the step's new entries and, when compaction ran, the summary card. This keeps the group-start snapshot under the size cap on long sessions. The trim runs behind the `write_incremental_only()` gate in `write_mode.py`, so instances without incremental-only checkpoints are unaffected. Checkpoint state reconstruction stays group-bounded as described above; the Rails `duoMessages` history read is the one reader that folds the write-path deltas and trimmed snapshots across groups, deduplicating by `message_id`.

Because the trim shrinks the channel in state, every other group start (first checkpoint, stale-cache reset, or a non-append history splice such as the cancelled-tool reorder in `ChatAgent`) snapshots at most the entries since the last compaction, about one context window.

The group boundary is keyed on `state.prev_checkpoint_id`, not on `current_thread_started_at`. The started-at marker is `None` for checkpoint IDs that aren't time-based, which would otherwise re-seed every step. Outside a group boundary, behavior is the per-channel deltas described above.

## State across `aput` calls

To compute deltas, `GitLabWorkflow` caches an `_IncrementalCheckpointState` between calls — a small dataclass holding `prev_channel_values`, `prev_checkpoint_id`, and `current_thread`.

One `GitLabWorkflow` instance is the checkpointer for the top-level graph **and** for every nested subgraph invocation it dispatches (e.g. a delegated subagent) — LangGraph resolves each of those under its own `checkpoint_ns`. The cache is therefore keyed by namespace: `self._incremental_state` holds one `_IncrementalCheckpointState` per lineage, reached through `self._incremental_state_for(checkpoint_ns)` (blank `checkpoint_ns` denotes the flow's own top-level lineage). This keeps a nested run's deltas from being computed against — and from clobbering — the top-level lineage's cached previous values.

Each entry holds a copy of its lineage's `channel_values`, and a session can dispatch an unbounded number of nested invocations, so `_trim_nested_incremental_baselines` bounds them: past `MAX_NESTED_INCREMENTAL_BASELINES`, the least recently used nested entries have their cached values dropped. Only the cache is lost — the next write to a trimmed lineage re-seeds every channel as a full snapshot, under a new `current_thread` so it never reopens the group those values belonged to. The top-level lineage is never trimmed: it lives as long as the session, and `_get_latest_checkpoint_status` reads its cached `status`.

On each `aput`, against that call's own namespace's `state`:

1. **Stale-cache detection** — if the incoming parent `checkpoint_id` doesn't match `state.prev_checkpoint_id`, a checkpoint was missed. Every changed channel is serialized as a full replacement (`force_rewrite`).

1. At a group start (first checkpoint, stale-cache reset, or compaction), the per-channel blobs are replaced with a full re-seed of all channels (see [Self-contained groups](#self-contained-groups)).

1. If the step was a compaction (or a stale-cache rewrite), `state.current_thread` is bumped.

1. Once the server accepts the write, the cache is updated; `channel_blobs` and `current_thread` are attached to the payload. `checkpoint_ns` is attached too, but only for nested lineages — see [Checkpoint namespaces](#checkpoint-namespaces).

## Surviving a gateway restart

The cache is in-memory, so a restart (or pickup by another gateway instance) would reset it and cause a spurious rewrite or a `current_thread` mismatch. `_hydrate_incremental_state(gl_checkpoint, decoded_checkpoint, checkpoint_ns)` restores the cached fields — for that specific namespace's `_IncrementalCheckpointState` — from a fetched checkpoint on every fetch path (REST, GraphQL latest, and latest-fetch). It accepts both `current_thread` (REST) and `currentThread` (GraphQL), and tolerates fields being absent (older Rails) or malformed by keeping the defaults. After a restart, a fetch followed by a write reuses the server's `current_thread` and emits a correct delta with no spurious bump.

## Checkpoint namespaces

LangGraph composes a `checkpoint_ns` for every nested subgraph invocation (e.g. a delegated subagent), joining `node:task_uuid` segments with `|`; the top-level graph's own lineage is the blank namespace (`TOP_LEVEL_CHECKPOINT_NS`). `GitLabWorkflow` is the checkpointer for all of these at once, so `aget_tuple`, `aput`, and the delta cache described above key their lookups, writes, and cached state on the `checkpoint_ns` LangGraph passes in `config["configurable"]` — never assuming the top-level lineage.

This matters for incremental checkpoints specifically because a lookup or write that ignored the namespace could seed a nested run with the parent's `channel_values` (duplicating `conversation_history`/`ui_chat_log` entries into the parent) or compute a nested write's delta against — and clobber — the top-level lineage's cached previous checkpoint. Concretely:

- `aget_tuple` skips the session-start `latest_checkpoint` cache for nested lookups (the GraphQL `latestCheckpoint` field only ever resolves the flow's own top-level lineage) and scopes the REST fetch with a `checkpoint_ns` query parameter, sent blank rather than omitted for the top-level case — an omitted parameter asks the list endpoint for the newest checkpoint across *every* lineage. The lineage of what comes back is then verified client-side rather than assumed, because an instance that ignores the parameter answers with a row from another lineage (see below); such a row is discarded, so the nested lineage reads as empty and its subgraph starts from scratch. Fetching a checkpoint by `thread_ts` needs no such check — that names one checkpoint in the whole flow.

- `aput` sends `checkpoint_ns` in the payload for nested lineages only; it's omitted for the top-level lineage so the payload stays identical to what GitLab instances predating `checkpoint_ns` already accept.

- Every returned `CheckpointTuple` reports the namespace it was resolved for, in both `config` and `parent_config`. This isn't cosmetic: LangGraph's `patch_checkpoint_map` reads `configurable["checkpoint_ns"]` unconditionally whenever a checkpoint has parents (i.e. for every nested run).

- `alist` and `checkpoints_reversed` stay deliberately unfiltered — they return every lineage mixed together, as before — but each returned tuple now reports its own namespace, read off the checkpoint itself.

- `aput_writes` drives the Rails workflow state machine off the `status` channel, and a nested invocation writes that same top-level channel. Session-*ending* events (`TERMINAL_WORKFLOW_STATUS_EVENTS`: `finish`, `drop`, `stop`) are therefore only honoured for the flow's own top-level lineage — otherwise the first of several parallel subagents to reach its terminal node would end the session the user is still in. The filter stops there on purpose: events meaning "blocked awaiting the user" (`require_input`, `require_tool_call_approval`, `require_plan_approval`, `pause`) still propagate from nested lineages, because that is how the client learns it has to prompt — a subagent pausing on tool approval genuinely blocks the whole session.

GitLab versions before 19.3 don't declare the `checkpoint_ns` parameter on the create/list checkpoint endpoints: Grape silently drops it on write, and ignores it on read, falling back to the pre-existing unfiltered behavior. Those versions also omit the field from the checkpoints they return, which is what makes the client-side lineage check above safe: their rows read as top-level, so only lookups for a *nested* namespace are refused. A nested lineage therefore cannot be resumed at all before 19.3, rather than being resumed from the wrong state; the flow's own top-level lineage is unaffected on every version.

## Monitoring which strategy is in use

Each `aput` tags its strategy (`incremental_only`, `incremental` or `full`) in two places so both sides are searchable in Kibana:

- **Request path**: the checkpoint POST carries a `?checkpoint_strategy=` query parameter. Rails doesn't read it (it's ignored by strong parameters), but it appears in Rails/workhorse request logs, so the backend side is searchable without any Rails change.

- **Gateway log**: the `Checkpoint saved` log line includes a `checkpoint_strategy` field.
