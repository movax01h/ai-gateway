# ADR-004: Prompts declare a prefix and a tail

- Status: proposed
- Date: 2026-10-29
- Deciders: `john-slaughter`, `mikolaj_wawrzyniak` (chat engine)
- Consulted: AI Gateway prompt registry maintainers, AI Catalog (`.luke`, `Imjaydip`: the validator),
  `thomas-schmidt` (Duo Developer)
- Related: [ADR-002](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7747) shapes the message the tail
  renders, and [ADR-003](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7748) delivers it

## Context

A user message reaches the model differently depending on the path:

- Legacy `chat.Workflow` stores each message raw and renders the platform's `user` partial around it on every call.
- `Flow` renders an agent's `user` role at the head of every call from `context.goal`, the session's first message.
- The chat engine design has the engine render each message and append it to the entry component's history
  (`chat_engine.md`, stage 3). Nothing on `main` does that yet.

Today's format can't express what an inbox needs. `prompt_template` maps each role to a string or a list of strings,
`user` included, so:

1. **`user` can't hold parts.** Attachments (images, video, audio, files) have nowhere to go.
1. **Whether `user` renders on every call or once per message is decided by the flow, not declared by the prompt.**
   The draft in !7578 inferred it from role layout (`Split`, `NoTail`, `Unsplittable`), which only works while the
   layout is unambiguous.

The rest of what goes wrong comes from elsewhere:

- `goal` doubles as the message being answered. ADR-003 delivers the message on its own.
- Legacy chat renders at read time, so a template change rewrites old messages.
- A per-turn variable rendered on every call changes the prompt prefix and breaks caching. It belongs in what renders
  once.

Constraints:

- Prompts are versioned, and shipped versions don't change. v1 is a stable public API, and flow configs define
  prompts inline too.
- Most flow configs pin prompts by caret range (`^1.0.0`).
- `chat` and chat-partial behave the same, with legacy `chat.Workflow` as the parity reference.
- Each message is rendered once into immutable history.

## Decision

### 1. Prefix and tail are defined by when they render

| Block | Key | Rendered | Stored |
|---|---|---|---|
| Prefix | `system` | On every model call | No |
| Tail | `message` | Once per claim, over the messages its component claims together (ADR-003) | Yes, appended to the component's history |

A model call sends the prefix, then history, and rendered tails are part of history. Nothing renders after history,
so a prefill can't be written. Both blocks come from one resolution and render through the same pipeline, so model
variables render the same in either.

### 2. The keys fix the roles

- **`system`** renders one system message, from its parts in order. Anthropic, for one, takes system content only at
  the top, and this keeps it there.
- **`message`** renders one user message per claim, so it fits right after history and never leaves an assistant
  message for the next call to continue. It's capped at one entry for now, which leaves room to extend it.
- **A tail before every call.** A component in a flow that opts in calls the model only after rendering at least one
  tail, so the prefix is never the last thing sent.

Neither block holds an assistant turn, so there's no few-shot exchange or prefill. The schema enforces the first two
rules, and the component enforces the third.

### 3. The format is declared, never inferred

- A new-format prompt uses a top-level `template` key instead of `prompt_template`. A prompt that has both is
  rejected. The key is the marker, so there's no `format` or `version` field, and nothing is inferred from which roles
  appear.
- `template` holds `system` and an optional `message`, each a list of entries. There's no history placeholder.
- An entry is a part: an object with a `type` and its `content`. A string is sugar for `{type: text, content: ...}`.
  That's a change from `prompt_template`, where a `system` list means separate system messages: under `template`,
  it's the parts of one.
  Parts map one to one to LangChain content blocks, so new part types (audio, files) are additive. Media parts hold
  references, which turn into bytes only when the model call is built (ADR-002 §6).
- The platform appends the claimed messages' attachments after `message`'s parts, in the same user message, as
  `assemble_user_message` does today.
- Every model-family variant of a prompt version declares the same format.
- Only agent-flow consumers read `template`. Other consumers of the shared `PromptConfig` keep reading
  `prompt_template`.
- The new format renders through the same `Prompt` path as the old one, so one place decides the layout. The
  tool-output security block is its own system message after `system`'s, and `Prompt` merges them per model
  (`merges_system_messages`), as it does today
  ([!7709](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7709)).
- Author system content renders before anything per-session, and nothing per-turn renders before history.
  Per-session author content goes in a later part of `system`, so the first part stays the same across sessions.
- Cache breakpoints are per message today (`default_cache_control_injection_points`). A breakpoint on the whole system
  message would cache the per-session parts too, so the new format needs a breakpoint after the first part (open
  question below). A test renders two sessions and compares the bytes through that part.

The agreed shape, on a generic prompt with the default `message`. The `chat/agent` migration renders its legacy
parity partials in `message` instead (§5):

```yaml
name: Example Agent
model:
  params:
    max_tokens: 32_768
unit_primitive: duo_chat
template:
  system:
    - type: text
      content: |
        {% include 'chat/agent/partials/system_static/1.0.0.jinja' %}
    - |
      {% include 'chat/agent/partials/system_dynamic/1.0.0.jinja' %}
  message:
    - "{{ inbound_messages.text }}"
```

### 4. One prompt object serves both formats

The registry resolves a prompt once into a typed object: a per-call render, a tail render or none, and the variables
each one uses. Callers don't branch on format:

- **New format:** the per-call render is `system`, and `message` renders once per claim.
- **Old format:** the per-call render is today's whole template, with `user` and history wherever it puts them, and
  there's no tail. Existing flows run byte for byte as today.

### 5. A version opts in by using the new format

- A flow version opts into the inbox (ADR-003 §8) by using new-format prompts. The config is the signal, and nothing
  translates an old-format prompt.
- A version is all new format or all old. One that mixes them refuses to build.
- An old-format version keeps today's delivery path for its whole life. Old-format chat-partial configs keep running
  on the legacy Agentic Chat engine, as unlisted ones do today, and a chat-partial version listed for the engine that
  uses the old format refuses to build. So the inbox never handles an old-format prompt.
- A new-format prompt with no `message` gets the platform default, `{{ inbound_messages.text }}`, followed by the
  attachments, as one user message. The cancelled-turn block is added by §6.
- The default isn't legacy chat's `user` partial. A flow that needs legacy parity, such as the engine-owned chat
  configs, ships a `message` that renders the additional-context and slash-command partials as new-format partials
  (§7).
- A prompt that needs a prefill or a few-shot exchange stays on the old format, so its flow can't opt in. No shipped
  flow prompt uses either today.

### 6. The tail renders the messages, and `goal` stays the session's purpose

- The tail gets every variable the prefix gets, plus `inbound_messages` and `cancelled_turn`. `inbound_messages` holds
  the messages claimed together, each with the same keys whatever its kind (`type`, `text`, `content` and
  `decision`), and `inbound_messages.text` joins their text.
- Per-turn values, such as plan mode, stay out of the prefix, so it stays stable. A tail can render them as user-role
  content, frozen when the message renders, or the client can send a change once, as its own message.
- ADR-002 defines the message, and ADR-003 delivers it. The `inputs` follow-up decides how authors reference it. The
  tail's `inbound_messages` should be whatever that input resolves to, under the same name.
- The engine writes `cancelled_turn` with the input it belongs to (ADR-002 §7). The standard `<cancelled-turn>` block
  goes on the render that includes that input, and no other. It's skipped when the tail places it itself, which means
  the template references `cancelled_turn`, or when the author points the input elsewhere.
- `goal` is the session's purpose, and `inbound_messages` are the messages being answered. A tail that renders `goal`
  repeats the first message every turn. That's the author's call, and the docs spell out the difference.
- A `directive` isn't rendered: the component that claims it runs it (ADR-002 §4), so it's never in
  `inbound_messages`.

### 7. Flows move by new versions, in a rolling migration

- Shipped prompts and flow versions don't change. `developer/4.0.0` isn't released before the new pipeline ships, so
  it moves to the new format in place (ADR-003 §8).
- New-format prompts ship as a new major version. A minor version would reach flows that pin by caret range and
  haven't opted in.
- A flow moves when it ships a new version that references new-format prompts, one flow at a time.
- A session never switches format mid-life: Rails pins its flow version, and the flow version pins a prompt major.
  Every version in a major keeps one format, which the validator checks. The one exception is a `developer/4.0.0`
  session started behind its feature flag before the move, which isn't protected (ADR-003 §8).
- Text partials work unchanged inside `system`. Wrappers written for the `user` role, such as the one legacy chat
  renders around each message, move into `message`, around `inbound_messages.text`.

### 8. A dry-run validator marks every flow

A validator loads every flow config, resolves its prompts, and reports for each flow:

- whether it can opt in as it is;
- whether it needs new-format prompts;
- if it can't, the blocker: a prefill, a few-shot exchange or a factory-built prompt.

It also checks the `template` schema, §5's one format per version, §6's tail variables, and that every variant of a
version declares one format. It runs in CI for every flow and prompt in this repository, and behind DWS's
`ValidateFlowConfig`, which the AI Catalog uses to validate customer flows, so customer flows join the rolling
migration too. Components refuse at build time as a backstop.

The rollout is staged. For customer 0, the new format is limited to chat-partial: the validator rejects `template` in
other environments, in CI and behind `ValidateFlowConfig`. Customer flows can't use it until two things change: the
engine builds only versions listed in this repository (`ENGINE_OWNED_CONFIG_VERSIONS`), and `ValidateFlowConfig` skips
the chat-partial dry run. Other environments follow once the first release is verified in production.

Classifying old-format layouts lives here, as a migration aid, not in the runtime.

## Consequences

- **The prompt registry API changes.** It returns the typed prompt object from one resolution, and validates the new
  format. The prompt registry maintainers own that part, in its own MR.
- **Two formats live in the registry for a long time.** Old prompts, inline ones in customer flows included, keep
  working as long as v1 does.
- **The docs describe the new format only.** The old format moves out of the Flow Registry docs into its own
  document, marked deprecated, which opens with how to migrate.
- **Shipped chat-partial configs need new versions to opt in,** because opting in means new-format prompts.
- **Old-format chat-partial configs keep running on the legacy Agentic Chat engine,** not the new pipeline, so the
  inbox has no old-format path (ADR-003 §8).
- **Factory-built prompts need new-format equivalents.** `agentic_chat/2.0.0` can't opt in on `chat/agent`, and the
  same goes for `workflow/*`.
- **Authors learn a second format.** The validator's messages and the docs have to make that cheap.
- **Caching.** The prefix stays stable across calls as long as per-turn values stay out of it, and a template change
  never rewrites stored messages. The breakpoint after `system`'s first part (§3) is new work, and it has to land
  before a prompt that puts per-session content in a later part, such as `developer/4.0.0`, moves.
- **One user message per claim.** Messages claimed together render into one user message, so two quick steers don't
  produce consecutive user messages.
- **A prefix that renders `goal` repeats the first message once,** in the system prompt and in the first tail. Chat
  prompts should reference `goal` in the prefix only when they need the session's purpose on every call.
- **Failure moves to validation time.** A new-format prompt is valid by construction, or the validator rejects it, in
  CI or in the AI Catalog.

## Open questions

- Whether attachments get a part type of their own in `message`, so the author places them, instead of the platform
  appending them.
- How the breakpoint after `system`'s first part lands: a per-part breakpoint, or `Prompt` sending the parts as
  separate system messages to models with `uses_cache_breakpoints`, the inverse of `merges_system_messages`.
- How a batch orders a pending directive, such as `/compact`, against the messages around it.
- How providers treat a user message right after tool results, which some send as a user turn. Batching makes it one
  user message per claim, and a self-hosted smoke test settles the rest before steering ships.
- Something clearer than a non-list input for excluding the cancelled-turn transcript.

## Alternatives considered

1. **Classify role layouts** (`Split`, `NoTail`, `Unsplittable`, the draft in !7578). It needs no new format, but the
   structure is guessed after the fact, and a prefix with a few-shot exchange reads as unsplittable.
1. **General prefix and tail lists** (`format: 2` with `prefix:` and `tail:` lists of role messages, an earlier draft
   of this ADR). They allow few-shot exchanges in the prefix and assistant turns in a tail, but no flow prompt uses
   either, and it's more format to learn. A key can be added if a case comes up.
1. **An explicit `format` or `version` field.** The Flow Registry's `version` keyword already confused authors, and a
   distinct top-level key says the same thing.
1. **Make the existing `user` key an object in new versions.** It fixes attachments, but whether `user` renders on
   every call or once still depends on the flow.
1. **Messages rendered one at a time,** each as its own user message, merged per model at the `Prompt` level. It hides
   the merge from the author. A batch lets the template decide how several messages read.
1. **A closed variable set for the tail** (`inbound_messages` and `cancelled_turn` only). Per-turn values like plan
   mode would have to live in the prefix and break the cache, and declared inputs couldn't reach the tail.
1. **Render at read time,** as legacy chat does. A template change re-renders old messages on deploy, which silently
   rewrites history and breaks the cache.
1. **Keep `user` a string and add a separate attachments key.** It fixes media but not extensibility, and every new
   part type needs another key.
1. **Accept old-format system-only prompts under opt-in, with the default tail** (an earlier draft of this ADR). It
   needs a translation rule from the old format to the new, which adds upkeep and new failure modes, and the config no
   longer says which delivery path a version takes.
1. **A validator ban on `goal` in a tail.** The tail is the author's to write, and documenting how `goal` and
   `inbound_messages` differ is enough.
1. **A boolean mode on each component, or a config field.** Either puts the structure in the flow instead of the
   prompt.
