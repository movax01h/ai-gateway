# ADR-003: User input reaches agents through a per-run inbox

- Status: proposed
- Date: 2026-10-29
- Deciders: `john-slaughter`, `mikolaj_wawrzyniak` (chat engine)
- Consulted:
  - `thomas-schmidt` (continued sessions, [epic 24022](https://gitlab.com/groups/gitlab-org/-/work_items/24022));
  - `rathid` (the server-side queue and steering, [epic 23954](https://gitlab.com/groups/gitlab-org/-/work_items/23954),
    [issue 633236](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236)).
- Related: [ADR-002](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7747) shapes the message,
  [ADR-004](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7749) renders it. Together they replace the draft in
  [!7578](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7578)

## Context

A user message reaches a graph in one of two ways today, and the steering spike needs a third. Each has its own gap:

1. **At the turn boundary**, as graph input. It's durable in the input checkpoint, but the run can't receive anything
   more until it ends.
1. **At a wait**, through `Command(resume=...)`. The resume value isn't stored, so a crash mid-resume asks the user
   again, and a root `Command(update=...)` can't reach a nested subagent's wait.
1. **Mid-run, as a steer** ([issue 633236](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236)).
   Nothing outside a graph can write LangGraph state while a node runs, so a steer has nowhere to go.

On `main`, a v1 flow takes the first two from the client's gRPC start request: `goal` for a message, and `approval`
for a decision. Rails isn't in that path. Only legacy workflows read the session mailbox (`duo_workflows_events`),
through `get_event`. Epic 24022 continues remote sessions without it: Rails starts the next CI job, and the session
page holds a message that's waiting ([issue 633289](https://gitlab.com/gitlab-org/gitlab/-/work_items/633289)). The
server-side queue comes with steering ([epic 23954](https://gitlab.com/groups/gitlab-org/-/work_items/23954)).

The steering spike converged on that mailbox as the one durable path
([summary](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236#note_4096641617)). A message is consumed with the
checkpoint that contains it, under the run lease. Only receivers read it, before their next model call, and stopping a
session hands queued messages back. Mikolaj asked for the synchronization scenarios before any binding decision
([note](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236#note_4097286679)). §6 decides the gate one, and the
stage one is an open question.

The draft in !7578 kept the message in a top-level `FlowState` key with a claims reducer. Two prototypes replaced
that, and they agree:

- [!7710](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7710) keeps
  the message out of state, in an inbox passed as LangGraph runtime context, and wires it into the real nodes.
- [!7719](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7719) and
  [!7722](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7722) add
  durable rows, the ack, addressing and decision rows, tested against the real `GitLabWorkflow` saver.

Terms: a run is one graph invocation, and this ADR says run throughout. A turn that pauses takes several runs. A node
claims a row when it acts on it, and reads a row when it only looks. A row is acked when Rails marks it consumed.

Constraints:

- DWS's saver stores only interrupt writes (`GitLabWorkflow.aput_writes`). A task that finished next to a failed or
  paused sibling runs again on resume, unlike with LangGraph's `InMemorySaver`.
- Saves are asynchronous by default, so they overlap the next step, and LangGraph saves the next checkpoint even when
  the previous save failed.
- `chat` and chat-partial behave the same, and v1 changes stay additive.

## Decision

### 1. The engine delivers, components render

The engine puts user input in the inbox, and the receiving component renders it through its own prompt into its own
history. The engine doesn't read prompts, component types or node names, and it writes no component state.
ADR-002 covers how the engine shapes the message, and ADR-004 covers how the tail renders it. How flow authors
reference inbound messages in config is a follow-up.

### 2. On the mailbox path, every user input is a row first

Messages, steers and every gate decision, including an approval with no text, are rows in the session's mailbox before
DWS sees them. Rails holds the durable copy. A row is an ADR-002 message, typed as the table in
[ADR-002 §2](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7747) says.

Rows come in two steps: messages first, then decisions without text, because those need every client to change.
Until the second step, a text-less decision rides the resume, as today.

With the mailbox (§8), the engine takes messages only from rows. The request's `goal` and `approval` still drive entry
classification, but they're delivered as input only where §8 says so, so nothing arrives twice.

### 3. Each run gets an inbox, outside state

Flow's override of `_resolve_graph_invocation` creates the inbox, seeds it with the session's pending rows, and passes
it to the graph as runtime context. It's the only place rows enter DWS. The inbox lives for one run, and rows nothing
claimed stay pending in the mailbox.

The inbox is a store, and it doesn't know components: each node decides whether it claims a row or only reads it. A
node claims synchronously, keyed by its task ID. A task that runs again in the same run reads its claim back, and a
sibling in the same superstep never sees what it claimed. DWS sets no retry policy, so the retry that matters is a new
run, where a row that wasn't acked is still pending and is claimed again. Claimed rows never return to pending within
the run. A claim means the task acts on the row: an agent renders it, a gate records the decision, and a component
runs a directive. An agent claims every pending message it would take at once, and renders them together as one user
message (ADR-004 §6).

A gate writes the UI log entry for each decision it reads, feedback included, with its own attribution, as today. For
any other row, the renderer writes the USER UI log entry in the same update as the render, so the entry is
checkpointed with the ack.

Rendering a message that was there when the run started begins a new cycle budget. That includes a gate answer or
feedback, because the user is back in the loop, whereas `chat_engine.md`'s stage 5 resets it only for plain input. A
steer is a row the inbox receives after the run started, in arrival order. How it reaches a running gateway is the
steering work's transport (epic 23954). A steer resets neither the budget nor `cancelled_turn`.

### 4. A row is consumed with the first durable checkpoint after the task that claimed it

- When a task finishes, the checkpointer parks the rows it claimed under the checkpoint ID the task ran on.
- The save built on that ID carries them to Rails, which marks them consumed in the same transaction, under the run
  lease ([#2877](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2877)).
- A row stays due on every later save in its namespace until one succeeds. That also covers a reply lost after Rails
  committed.
- A task that fails or pauses acks nothing, so the row comes back on the next run. The checkpointer only ever sees
  task IDs and opaque row IDs.

This rests on four LangGraph behaviors:

- each task's `aput_writes` pairs with the `aput` built on the same checkpoint ID;
- parking before the first `await` lands before that `aput` builds its body;
- the next save runs after a failed one;
- DWS doesn't use `exit` durability today (`AbstractWorkflow._checkpoint_durability`). A flow that needs it works too,
  if the ledger acks every row a finished task claimed on the exit save, with its own guard test.

!7719 and !7722 verify them, with controls that fail without the right rule. Those tests ship as guard tests with the
implementation and run on the pinned LangGraph, so an upgrade that changes the pairing fails CI. That keeps the
dependency raised in [#633236](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236#note_4091181150) explicit and
tested.

Two of the concerns in that note need more than tests:

- **Fire-and-forget saves** ([issue 596713](https://gitlab.com/gitlab-org/gitlab/-/work_items/596713)). The rule holds
  as long as the saver learns each POST's outcome, so a background POST has to report back before its IDs stop being
  due. A save that never reports keeps them due.
- **Subgraphs.** Nothing bubbles up. One saver serves every namespace, so a nested claim is acked on the nested save
  (§7). The guard tests don't cover parallel subagents yet, which stay behind `dap_parallel_subagents`.

### 5. Rows carry an address the engine never reads

`recipient` names the component the client was showing when the user hit send. The engine never reads or writes it.

- A node claims the rows addressed to its own name.
- A row with no address goes to the component the turn is at: the flow's entry on a new turn and mid-run, and the
  component that paused on an interrupt resume. When several components paused in one turn, the first to resume takes
  it. That's the limit of unaddressed rows, and the client capability removes it.
- A row with an `in_reply_to` is never claimed by address: only a lookup by that ID reads it (§6). So an answer
  ignores any `recipient` (ADR-002 §1). Forwarding a `HumanInputComponent` answer to its receiver would add one to
  answers, as an additive field (see the open question on how that answer reaches its receiver).

### 6. Every decision is a row with one claimer

- A tool approval is identified by the ID of the `AIMessage` it guards. The request node puts it on every card it
  writes, so it doesn't depend on which calls were pre-approved. An `AIMessage` without an ID gets one before it's
  stored, so the gate and the agent can read it from history. Rails and the client carry it opaquely in
  `in_reply_to`.
- The gate reads the row whose `in_reply_to` names it before it pauses, so a RETRY with no resume value still
  proceeds.
- The gate claims a decision without feedback, and answers rejected calls itself, as today. A decision with feedback,
  the gate only reads.
- The gate records the decision's approval attribution for each tool call, as today (ADR-002 §1).
- The agent the gate routes to claims the feedback by the `AIMessage` ID in its own history. It answers the rejected
  calls, then renders the feedback.
- A tool-approval gate doesn't ask while a row its own agent would claim under §5 is still pending
  ([note](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236#note_4094197694)). The request node checks before
  it writes any card or status, so no card is left with nobody to answer it. It routes back to the agent, which
  answers the skipped calls itself, saying they weren't run because the user sent a message, then renders the message.
- **`HumanInputComponent` gates work the same way.** Its question gets an ID that the client stamps on the answer.
  `FetchNode`'s write step records the decision only, and the answer's receiver claims and renders it. A target that
  isn't an agent renders no tail, so for it `FetchNode` writes the answer into its history as today. A text-less
  reject keeps today's "User rejected this action" message, written by the gate. `message_template` stays the
  question the gate shows.

### 7. Nested runs see a view of the parent's inbox

A dispatch forwards the rows addressed to its subagent into a child inbox that shares the parent's claims. On an
interrupt resume it also forwards unaddressed rows (§5). A nested render is acked under the nested checkpoint
namespace. A dispatch clears `cancelled_turn` from the subagent's state, so a subagent never reads the user's
cancelled transcript.

Delegations a supervisor creates are not mailbox rows. They're addressed to the subagent and never acked. The
supervisor still writes the subsession `goal`, but the subagent's tail renders the delegated message, because a
resume doesn't rewrite `goal`.

### 8. Opt-in, compatibility and the bridge

- A version opts in by using new-format prompts (ADR-004 §5). An old-format version keeps today's delivery path for
  its whole life, and existing flows are unchanged. Old-format chat-partial configs keep running on the legacy Agentic
  Chat engine, as unlisted ones do today, and a chat-partial version listed for the engine that uses the old format
  refuses to build. So the inbox has two inbound paths, the bridge and the mailbox, and no old-format one. A session
  never switches mid-life: Rails pins the resolved version (`chat_engine.md` "Sessions in
  flight", [issue 630374](https://gitlab.com/gitlab-org/gitlab/-/work_items/630374)). `developer/4.0.0` is the
  exception: it isn't released before the new pipeline ships, so it moves to the new format in place, and sessions
  started behind its feature flag before the move aren't protected.
- **Capabilities are pinned per session.** The engine records the capabilities a session started with in
  engine-private state. Each run uses the intersection of those and its own, so a session can lose a capability
  mid-life but never gain one.
- **Rails capability.** Older GitLab versions talk to the same gateway, so the listing alone can't tell. GitLab
  declares the mailbox as a server capability, as it does for `idle_status_event`
  ([ADR-001](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7419)).
  Without it, the run takes the bridge. With it, the engine takes messages only from rows (§2). The mailbox path needs
  the server-side queue (epic 23954) to put Rails in the delivery path: today the client sends `goal` and `approval` on
  the stream and Rails never sees them. So every rule here that has Rails consume a row is a Rails prerequisite, not
  something the gateway can do. A DWS older than its Rails isn't supported, so only older Rails needs a fallback.
- **Client capability.** Clients that stamp `recipient`, `in_reply_to` and a typed reply's `decision` advertise it.
  Without it, rows are unaddressed, and decisions ride the resume as today, a typed reply included, which the gate
  takes as a rejection with feedback. The gate puts that text into the inbox as the user's `answer` with a reject
  `decision`, an engine ID (ADR-002 §2) and the guarded `AIMessage` ID as `in_reply_to`, so the agent claims and
  renders it as it would a row (§6). Like a delegation, it isn't a row, so it's never acked. For those clients a crash
  mid-resume asks again. Rails consumes such a decision's row before the resume reaches the gateway, so nothing claims
  it a second time.
- **Bridge.** Where GitLab doesn't declare the mailbox, which is every GitLab version until the server-side queue
  ships, the engine seeds the inbox from the graph input, with an engine ID (ADR-002 §2), and the entry claims and
  renders it as it would a row. It's the first production path for a version that opts in. The bridged
  message lives in a state key until the entry renders it, and the render clears it in the same update. A retry from
  the input checkpoint seeds it again, and a retry from a later checkpoint doesn't. That's a state key like the one the
  first alternative rejects, and it's fine here: it holds one message, written only at turn start and never mid-run.
  Without it, a crash before the render would lose a turn-start message, which is durable in the input checkpoint
  today.
  Decisions ride the resume as today, there are no steers, and there's no row to ack. Nothing is refused, so
  ADR-001's support for older GitLab holds.

## Consequences

- **Rails and clients:**
  - every input becomes a row, including turn-start messages and, in a second step, text-less decisions. Approvals
    travel on the gRPC request today, so text-less approvals as rows are new client and Rails scope, bought so that a
    crash mid-resume doesn't ask again;
  - clients with the capability stamp `recipient`, `in_reply_to` and a typed reply's `decision`;
  - the checkpoint POST gains the consumed row IDs;
  - GitLab declares the mailbox capability.

  Rows, the ack and the capability belong with the server-side queue, which epic 24022 places in epic 23954, plus a
  checkpoint API change. Who owns them there is an open question below. Epic 24022's team offered to own the client
  changes, after aligning with the client teams.
- **Production wiring.** `AbstractWorkflow` builds the saver before the inbox exists, so the saver needs a late-bound
  reference to the current inbox. The rows replace the prototype's in-memory puts (!7710). If both delivered the same
  input, it would render twice.
- **Audit.** Seeding the inbox emits `UserInputReceivedEvent` per row, as `get_event` does today, and decision rows
  keep the gate's approval tracking.
- **A row left pending at a clean turn end** is Rails's to deliver as the next turn (epic 24022,
  [issue 633289](https://gitlab.com/gitlab-org/gitlab/-/work_items/633289)). After a stop, a rollback or a regenerate,
  pending rows are handed back to the user as unsent, as the steering spike agreed; that comes with the server-side
  queue (epic 23954). The engine decides neither: it only leaves unclaimed rows pending.
- **On the mailbox path, the raw message is no longer in any checkpoint.** The row keeps it, under the session's
  retention, and the checkpoint holds the render.
- **A rollback never un-consumes a row.** When a stop discards the checkpoint that consumed it, the model sees the row
  only in `cancelled_turn`, as its render.
- **Checkpoints after a tool rejection with feedback change shape.** The calls stay unanswered until the agent runs. A
  gateway without this change fills them with a synthetic "interrupted" `ToolMessage` (`restore_message_consistency`).

## Open questions

- **Who owns the mailbox prerequisites in epic 23954:** rows for every input, the ack in the checkpoint transaction,
  the dedupe on the fallback paths and the capability. Asked on !7748.
- **Delegations that fail before they render.** A delegation that fails before its subagent renders it isn't
  delivered again: v1 `DelegationNode` doesn't rerun on a retry, and a v2 dispatch seeds its task only on a fresh
  dispatch. One direction: make delegations mailbox rows that the subagent acks. That would also make them auditable,
  and let a supervisor steer a subagent mid-run. It needs Rails rows for input created inside the graph.
- **How a `HumanInputComponent` answer reaches its receiver.** For a tool approval, the agent finds its feedback by
  the `AIMessage` ID in its own history. A HumanInput question rarely comes with tool calls, so it needs its own ID
  source, and the receiver doesn't hold it. The gate has to hand it over, by recording it where the receiver reads it
  or by forwarding the answer addressed to the receiver. The ID also has to be reproducible on re-entry.
- **The stage race.** A steer pending when a stage completes, or when a `HumanInputComponent` gate asks, in a staged
  `chat` flow ([note](https://gitlab.com/gitlab-org/gitlab/-/work_items/633236#note_4097286679)).
- Telling the entry apart in staged flows that opt in.
- Authors reference inbound messages through `inputs`, with a default input (follow-up MR).
- Rethinking cancellation on top of the inbox
  ([issue 2998](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2998)).

## Alternatives considered

1. **A top-level `FlowState` key with a claims reducer** (the draft in !7578). Claims and history checkpoint together,
   but the key holds one message, a run can't take more mid-run, and nested waits are out of reach.
1. **A destructive pop from an in-memory queue.** A resume, a retry or a failed task loses the message.
1. **Dropping a row when the run ends cleanly.** A durable render followed by a failure in the same run is rendered
   again next turn, and consumers don't dedupe. !7719 shows it with a control.
1. **Consumed IDs in state.** A sibling in the same superstep can't see them until the next one, and the set needs a
   bound.
1. **No address, so the first claimer takes everything** (!7710 as first proposed). Parallel claimers and parallel
   resumes take each other's messages. Unaddressed rows keep that limit on an interrupt resume (§5), and only there.
1. **Separate child inboxes for nested runs.** The parent's checkpointer never sees the nested claim, so a follow-up
   rendered before a second pause renders twice. !7722 shows it with a control.
1. **Acking outside the checkpoint transaction,** by reading the `tasks` stream or with a separate call. There's a
   window where Rails shows a message as queued after the agent read it.
1. **Acking the row IDs the saved checkpoint contains,** with each render carrying its row ID. The saver would read
   the shape of state, gate decisions and directives claim a row without rendering one, and compaction drops message
   IDs.
1. **`durability="sync"` for engine flows,** which `SYNC_CHECKPOINT_FLOWS` already supports. It removes the overlap
   and the parking-order dependency, at a latency cost on every step.
1. **Identifying a tool approval by its first tool call ID** (!7722). The request node drops pre-approved calls before
   it writes cards, so the client may never see that ID.
1. **A gate subclass chosen by which config fields are absent** (!7578). It ties the gate's question to its answer
   handling and adds a validation trap.
1. **Today's delivery path inside the engine for old-format chat-partial versions.** It's a third inbound path that
   does the bridge's job. Routing those configs to the legacy Agentic Chat engine keeps the engine at two.
