# ADR-002: The engine shapes every user input into a typed message, and never renders it

- Status: proposed
- Date: 2026-10-29
- Deciders: `john-slaughter`, `mikolaj_wawrzyniak` (chat engine)
- Consulted:
  - `thomas-schmidt` (continued sessions, [epic 24022](https://gitlab.com/groups/gitlab-org/-/work_items/24022));
  - `rathid` (the server-side queue and steering, [epic 23954](https://gitlab.com/groups/gitlab-org/-/work_items/23954)).
- Replaces: the draft ADR-002 in
  [!7578](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7578), never
  accepted, together with ADR-003 and ADR-004
- Related: [ADR-003](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7748) delivers the message,
  [ADR-004](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7749) renders it

## Context

[`chat_engine.md`](../chat_engine.md) has the engine shape every user message with the same pure stages before the
graph runs, and the ingestion prototype
([!7467](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7467)) built
them:

1. claim attachments into content blocks;
1. tell a platform directive from a message;
1. render the message once.

Rendering in the engine means the engine has to know receivers, prompt templates and node names, which that
prototype showed. ADR-003 moves delivery into a per-run inbox, and ADR-004 moves rendering into the receiving
component's prompt.

Input comes from several places: a turn-start message, a steer mid-run, a directive, a gate answer, a rejection with
feedback, and a supervisor's delegation. The component that renders it needs one shape to read, and a type to branch
on. Today `goal` doubles as the message, so templates can't tell the session's purpose from the message being
answered.

Constraints:

- v1 is a stable public API. New behavior is additive and opt-in.
- Checkpoints resume across deploys and rollbacks in both directions.
- State is checkpointed, as snapshots or deltas, and mailbox rows live in Rails, so media can't travel as bytes.

## Decision

### 1. One shape for every message

Every input is one of three kinds, and each kind carries only the fields it uses:

| Field | Meaning | `message` | `directive` | `answer` |
|---|---|---|---|---|
| `type` | The kind, as templates see it | `message` | `directive` | `answer` |
| `id` | Unique per input, set by its writer (§2) | required | required | required |
| `content` | Typed parts, such as text | required | none | required, except with a `decision`: none for an approval, optional feedback for a rejection |
| `text` | The text parts joined with newlines | computed | as the user typed it | computed |
| `sender` | Who wrote it: the user, or a named component | required | the user: only the user runs a directive | the user: only the user answers |
| `recipient` | The addressee: stamped by the client, or set by a supervisor (ADR-003 §5, §7) | optional | optional | ignored: a reply is found by `in_reply_to` |
| `in_reply_to` | The approval or question it answers (ADR-003 §6) | none | none | required |
| `decision` | A decision on an approval: `choice`, `approval_source` and `policy_ref` | none | none | optional |
| `command`, `argument` | The directive and its instruction, such as `compact` and `keep the plan` | none | parsed from `text` | none |

- A part is an object with a `type`, such as `{type: text, text: ...}`, and maps one to one to a LangChain content
  block. It holds only JSON values. A media part holds a reference, never bytes (§6), and a part type the gateway
  doesn't read is refused (§8).
- A message's fields can't be reassigned once it's built, and its parts are copies of what the writer sent. Nested
  values aren't frozen, so a component treats what it claims as read-only, and a retry reads what the writer sent.
- The type can't contradict the fields: only the user's text can be a `directive`, and an `answer` always has an
  `in_reply_to`.
- `sender` is the user for an input that came through the user's own channel, even when the engine or a gate builds
  the message, and whether or not a human typed it. Every other writer names itself as a component, as a supervisor
  does. Only a `message` can come from a component: a `directive` and an `answer` are always the user's, and a saved
  one that names anyone else doesn't load. So a component's input never acts with the user's authority. A component
  without a name, or a blank `recipient`, is invalid.
- A `decision`'s choice is approve or reject and nothing else, because a gate can't guess what an unknown choice means
  for consent. `in_reply_to` stays outside the `decision`, because an input gate's answer has one without a decision.
- A decision carries its approval attribution, so the gate still records it for each tool call. It's the same
  attribution the request carries today: `approval_source`, which DWS resolves with `ApprovalSource.from_approval`,
  and `policy_ref`, the client's provenance. No writer can claim a session approval: the engine reads one as
  `user_explicit`, as `from_approval` does. Who resolves the rest of `approval_source` for a row is an open question
  below.
- The Python representation lives in `duo_workflow_service/agent_platform/core/inbox/message.py`, and the code is
  authoritative for how it enforces these rules.

### 2. The type follows from the fields

| Input | Type | Built by |
|---|---|---|
| A message at the turn boundary, or a steer mid-run | `message` | The engine, from the mailbox row |
| A platform directive, such as `/compact` | `directive` | The engine, from the mailbox row (stage 2) |
| An input or external gate answer, or an approval decision with or without feedback | `answer` | The engine, or the gate when the feedback rides the resume (ADR-003 §8) |
| A delegation from a supervisor to its subagent | `message` | The supervisor, inside the graph, as its `sender`. It isn't a mailbox row (ADR-003 §7) |

The type follows from the fields, whoever builds the message:

- A row from the user with no `in_reply_to` that the engine identifies as a platform directive (stage 2) is a
  `directive`. A reply is always an `answer`, even when its text is a directive. Only the user runs platform
  directives, so a component's message never is one.
- Any other row with neither an `in_reply_to` nor a `decision` is a `message`.
- A row with a `decision` is an `answer`, with or without feedback. Its readers tell a rejection from an approval by
  the `decision`.
- A row with an `in_reply_to`, text and no `decision` is an `answer`, as an input gate's answer is.
- When the user types instead of choosing at a tool approval, the client stamps a reject `decision`. Only the client
  knows it showed an approval card, and the engine doesn't read gate kinds (ADR-003 §1). Stamping it is part of the
  client capability (ADR-003 §8).
- A supervisor's delegation is a `message` to its subagent, with the supervisor as `sender`.

`id` comes from the writer. For a mailbox row it's the row's `correlation_id`, a UUID the client creates, so a message
sent twice maps to the row that already exists. For a delegation it's the `delegate_task` call ID. An input that arrives
without a row, through the bridge or on a resume (ADR-003 §8), gets its ID from the engine. The bridge stores it with
the graph input.

### 3. Input is validated where it enters DWS

The engine validates each row when it reads it into the inbox, never inside a component. A row must have an ID, at
least one part unless it's a decision, and an `in_reply_to` when it carries a decision. A row with an approve decision
has no parts, because an approval carries nothing but the decision. A row must also follow the field rules in §1 and
§6, such as a known `choice`, no blank `recipient` and no bytes in a part.

A row that fails validation never reaches the inbox. When it's the input that would start or resume a run, such as a
message at the turn boundary or a gate's answer, the run doesn't start: the session stays `IDLE` or `INPUT_REQUIRED`, and the client gets an error that says what to fix, so the user
can send it again. A steer that fails goes back to the user as unsent, and the run in flight carries on. Rows only exist where GitLab declares the mailbox (ADR-003 §8), and a legacy client's decisions still ride the resume,
so validation never sees what older clients and older GitLab send.

### 4. The engine shapes, and renders nothing

`chat_engine.md`'s stages 1 and 2 stay: attachments are claimed into content parts, and platform directives are told
from messages. Stage 3 changes from "render the user message once" to "build the typed message". The receiving
component renders it (ADR-004).

The engine doesn't run a directive either. It delivers it as a `directive` message, and the component that claims it
runs it, because running `/compact` needs the component's internals.

### 5. Envelopes stay in `context:inputs`

Additional context, such as page context, is flow-scoped. It stays in `context:inputs`, refreshed at the
start of every run that carries input, a turn-start message or a resume (stage 4). A steer can't write state mid-run,
so it doesn't refresh them. The message has no context field.

### 6. Media parts hold references, never bytes

A media part carries a reference, such as an upload ID. Each kind of media is its own part type, such as a
`FileReferencePart`, so the `inputs` follow-up can read parts the way it reads text.

The prompt registry turns a reference into bytes when it builds the model call, close to the model clients, so the
engine never couples to them. Bytes never sit in a row or in state.

### 7. The engine writes `cancelled_turn` with the row it belongs to

The engine writes `context:inputs.cancelled_turn` at the start of every run that carries input, a turn-start message
or a resume, in the run's graph input or resume update. It writes it together with the ID of the input it belongs to:
the oldest one pending when the run starts that a component will render. That rules out a delegation, a `directive` and
an input with no parts. It's `None` when nothing was rolled back. When the run has no such input, as when a text-less
approval resumes it, the transcript doesn't reach the model.

No render clears it. Only the render that includes that input carries the transcript (ADR-004 §6), so a retry, or a
second render in the same run, never repeats it. A steer arrives mid-run and leaves it alone (ADR-003).

### 8. Fields are additive only

Fields and part types can be added, never removed or renamed. A consumer ignores fields it doesn't know. A tail that
doesn't branch on a message's `type` renders it like any other message (ADR-004).

A part type is different, because nothing in a gateway can render one it doesn't read: a media part's reference only
turns into bytes in that gateway's prompt registry (§6). So the engine refuses an input with a part type it doesn't
read, and logs the type. The client gets an error that names the type and says to upgrade the AI Gateway or send the
input without it (§3), rather than a model that silently never sees the part. A gateway supports a part type before a
client sends it, which is the usual deploy order: the Duo Workflow Service first, then GitLab.

## Consequences

- **`chat_engine.md`.** Stage 2 still tells a directive from a message, but delivers it instead of running it. Stage 3
  is rewritten, and stage 5's budget reset moves to the agent that renders (ADR-003). Recovery's "cleared once read"
  goes too: the engine binds the envelope to one input, and only the render that includes that input carries it
  (ADR-004 §6).
- **The resume no longer needs a rendered-content field on `FlowEvent`.** `chat_engine.md` planned one, but decisions
  now arrive as rows (ADR-003 §6). On the bridge they stay text, as today.
- **Legacy `chat.Workflow` consumes the same stages today, and renders at read time.** It needs stage 3's render step
  kept as a function it calls itself.
- **New v1 surface.** The message shape becomes readable by templates through ADR-004, and later by flow config
  through the `inputs` follow-up.

## Open questions

These come from !7578:

- When do per-message envelopes leave `context:inputs`, for example an @-file becoming a `FileReferencePart`? That
  includes a steer's @-file, which can't reach `context:inputs` mid-run.
  ([thread](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7578#note_4052592432))
- How does the prompt registry cache media bytes?
  ([thread](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7578#note_4052593050))
- Can flow authors read the message, or is it framework-only until there's a consumer? The `inputs` follow-up takes
  this one.
  ([thread](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7578#note_4052590267))

New here:

- Who resolves `approval_source` for a row: whoever writes the row, or DWS from the raw approval fields, as it does
  from the request today? The answer decides whom the field trusts.
- What else does a `sender` change, besides ruling out directives and answers: the USER log entry, the budget
  reset (ADR-003 §3) and how a template frames the message? The first MR that puts a delegation in an inbox answers
  it (ADR-003 §7).
  ([thread](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7747#note_4140175504))
- Can an `answer` come from a component, such as a supervisor answering its subagent's question? Today an answer's
  `sender` is always the user. Allowing a component later is additive (§8): a gateway that doesn't allow it refuses a
  saved component's answer instead of reading it as the user's.
  ([thread](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7881#note_4176072804))

## Alternatives considered

1. **The engine renders each message at ingestion,** stage 3 as `chat_engine.md` plans it. It works, but the engine
   has to know receivers, prompt templates and node names.
1. **The engine runs directives itself,** as `chat_engine.md` plans for `/compact`. Running one needs a component's
   internals.
1. **The engine appends a raw `HumanMessage`.** The author's `user` template can't frame later messages, and the
   engine writes component state.
1. **Per-component markers** (`context:<name>.last_inbound_id`) instead of one shape. They depend on two keys changing
   together, and a dropped message leaves no trace.
1. **The message under `context`.** `context` is deep-merged, so a new message mixes with an unconsumed one, and it's
   copied into every dispatched subagent.
1. **A string message with side fields for attachments.** Every new part type needs another field.
1. **A `feedback` type for a rejection with text, and a `task` type for a delegation.** No reader of feedback needs
   its own type: the gate branches on the `decision`, and the agent finds it by `in_reply_to`. A delegation differs
   from a message only in who sent it, which `sender` says for any writer.
1. **`from` and `to`.** `from` is a Python keyword, so the field would need an alias with a different name in Python.
   `sender` and `recipient` pair without one.
1. **`recipient` with a reply flag instead of `in_reply_to`.** A flag says a row is a reply, not which question it
   answers. The gate binds a decision to the exact `AIMessage` the user saw, and finds it by that ID on a retry
   (ADR-003 §6).
