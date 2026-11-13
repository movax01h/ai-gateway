# ADR-005: MCP tools reach a `tools:` component only through explicit selectors

- Status: proposed
- Date: 2026-10-19

## Context

A YAML flow config defines a flow in GitLab Duo Agent Platform (DAP). It lists the `components` that do the work (for
example an `AgentComponent`), the `routers` that connect them, and an `environment`. The environment sets how much a
human takes part: `chat` and `chat-partial` flows are conversations, and `ambient` flows run in the background
([flow registry v1](../v1.md)). The flow registry has two versions, `v1` and `experimental`. Duo Workflow Service (DWS)
runs the flows.

An `AgentComponent` lists the tools it may call in its `toolset:`. Each entry is a tool name such as
`get_merge_request`, or a tool name with option overrides. The names refer to tools that DWS
registers itself (`duo_workflow_service/components/tools_registry.py`). These are the native tools. Today
`toolset:` can't select tools by server or by pattern.

Tools can also come from MCP servers. The Model Context Protocol (MCP) lets an outside server offer
tools to an agent ([GitLab documentation](https://docs.gitlab.com/user/gitlab_duo/model_context_protocol/mcp_clients/)).
DWS receives MCP tools from these servers:

| Server            | Sent by                                              | Named by               |
|-------------------|------------------------------------------------------|------------------------|
| `gitlab`, `orbit` | Rails                                                | The server name        |
| `slack`           | Rails, for the Slack assistant only                  | The server name        |
| Catalog servers   | Rails, for the servers an agent selects              | The server name        |
| Local servers     | The client (an IDE or the CLI), read from `mcp.json` | Nothing: one flat list |

The message that carries each MCP tool to DWS (the tool message) has a name and no server field, so DWS can't tell
which local server a tool came from.

MCP tools don't appear in `toolset:`. Instead, DWS adds every tool from a connected MCP server to each component that
declares a `toolset:`, in `chat` and `chat-partial` flows. In `v1`, an `AgentComponent` with no `toolset:` gets them
too. These are the automatic MCP tools. `ambient` flows stopped getting them
in [!6362](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6362). That
split (on in `chat` and `chat-partial`, off in `ambient`) was a stop-gap, so that users with servers in
`.gitlab/duo/mcp.json` keep their tools
([#2412](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2412#note_3646541880)).
It leaves three problems:

- Cost: automatic MCP tools raised the cost of a flat-rate flow.
- Access control: a flow gets tools it didn't declare
  ([epic discussion](https://gitlab.com/groups/gitlab-org/-/work_items/23110#note_3680233027)).
- Inconsistency: custom agents get only the tools they declare. Custom flows don't.

The design discussion is in
[ai-assist#2667](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2667).

## Decisions

These decisions describe the end state. [Phased delivery](#phased-delivery) says what each phase ships, so some
examples below use syntax that arrives in phase B or C.

**Create a new `tools:` key.** The GitLab MCP server already uses `toolset` for its tool groups
([documentation](https://docs.gitlab.com/user/model_context_protocol/mcp_server/#select-tool-groups-toolsets)), so the
new key has a different name. Keep `toolset:` as the legacy key, so that existing flows keep working as they do today.
GitLab deprecates it after a cut-off, through its standard deprecation process, which tells customers first. The
automatic MCP tools end with it, because a `tools:` component gets none.

**A flow uses one key, `toolset:` or `tools:`.** A component can't have both keys, and a flow can't mix the two kinds
across the components it defines (the mix rule). An item that the flow's top-level `include` section brings in, such
as a subagent from the workspace or the AI Catalog, keeps its own key and is exempt
([proposed by `mikolaj_wawrzyniak`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7550#note_4159806885)).
Each such item comes with a `toolset:` today, so without the exemption a `tools:` flow couldn't include a workspace
agent at all. The `experimental` flows have no `include` section, so the mix rule covers every component in them,
except a `tool_name:` step (see below).

**A `v1` `AgentComponent` with neither key counts as a `toolset:` component.** DWS treats it as `toolset: []`, so it
gets the automatic MCP tools in `chat` and `chat-partial` flows. A flow that has `tools:` on any component and leaves
an `AgentComponent` without either key is a config error, because it breaks the mix rule above.

**A `DeterministicStepComponent` with `tool_name:` is exempt from the mix rule.** It names one tool, which the step
calls with fixed arguments. It gets no automatic MCP tools in either kind of flow, so a `tools:` flow keeps its
deterministic steps as they are.

**A component with a `tools:` list gets an MCP tool only when an entry selects it.** It gets no automatic MCP tools,
and `tools: []` gives it no selectable tools, only its internal tools (see `server: gitlab` below). An empty list that
selected every tool would fail open. The rule is the same in every environment, because behavior that differs per
environment confuses authors. A `toolset:` component behaves as today: it keeps the automatic MCP tools it gets in
`chat` and `chat-partial` flows until its config moves to `tools:`. That includes the GitLab MCP tools that Rails
sends, such as the tools selected on an AI Catalog agent. Pairs apply to a `toolset:` component too (see below).

**Every entry in `tools:` is a selector, so there's one way to write an entry.** A selector has an `origin`
(`instance` or `client`), a `server`, and a `name`. `server` is one string. `name` is a list of tool names without the
`<server>_` prefix that Workhorse puts on MCP tool names. `name` and `server` are required. A selector carries no tool
options (the `ops` field) until phase C. Until then a flow with a component that needs a tool option stays on
`toolset:` in every component, because of the mix rule. For example:

```yaml
tools:
    - origin: instance
      server: gitlab                   # a tool GitLab provides, native or MCP
      name: [ "get_merge_request" ]
    - origin: instance
      server: gitlab                   # exact server, exact and pattern names
      name: [ "list_work_items", "list_merge_request_*" ]
    - origin: client
      server: jira                     # a local server, under its own name
      name: [ "*" ]                    # bare `*` name: every tool jira offers
    - origin: instance
      server: "corp-*"                 # pattern server: every server named corp-something
      name: [ "read_issue" ]
    - origin: client
      server: "*"                      # bare `*` server: any local server
      name: [ "search_*" ]
```

**Both `name` and `server` allow patterns.** A pattern uses `*` for any run of characters, so `list_*` matches
`list_work_items` and `list_projects`. A bare `*` matches anything. The syntax is the same for every server. An author
who can't name a tool or a server ahead of time uses a pattern, for example a generic flow that takes the servers an
end user adds. A pattern also selects a tool that a server adds later, so a flow that needs a tight tool set lists
exact names. A flow that GitLab ships and that reads untrusted input, such as merge request descriptions or comments,
lists exact tool names.

**`server: gitlab` covers every tool that GitLab provides.** A flow author doesn't track whether DWS implements a tool
natively or the GitLab MCP server offers it, so both sit under `server: gitlab`. That includes native tools with no MCP
version, such as the plan, task, UI, shell and web search tools. A native tool matches only `origin: instance`.
Internal tools are the exception: a component adds the tools its own features need, such as the delegate task tool
when it has subagents and the response schema tool when it has a response schema. DWS always includes them, an author
can't remove them, and a selector never matches them.

**MCP tool migration: a native tool and an MCP migration tool form a pair.** Pairs are formed in cases where a native
and MCP tool exist, for example, the native `create_branch` and MCP `add_branch`.
The [design doc](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/21789) lists the pairs and
decides when an MCP tool replaces the native tool in a pair (_the pair is on_). Pairing works one way only: named
native tools can be replaced with the paired MCP tool but not the other way around. In either a `toolset:` or `tools:`
key:

- Naming a native tool returns the native tool if the pair is off, or an MCP tool if the pair is on and the paired MCP
  tool is available.

In just a `tools:` key (where MCP tools are selectable):

- Naming a paired MCP tool always returns the MCP tool if available. This never falls back to a native tool.
- If both paired tools are named, only the MCP tool is used if available, else the native tool is used.
- If `name: ["*"]` is used, each pair gives the same tool as naming the native tool: the MCP tool when the pair is on
  and the MCP tool is available, else the native tool.
- Naming a tool without a pair selects that tool.

For example, `name: ["create_branch"]` gets native `create_branch` when the pair is off, and MCP `add_branch` when the
pair is on and the GitLab MCP server sent it.

**Entries add up.** A component gets every tool that any entry names (a set union), so each entry reads on its own and
no entry removes a tool. A tool that no config may select needs an admin rule: governance denies it, not the config.
For example, this component gets `get_merge_request`, `list_work_items` and `get_work_item` from `gitlab`, and
`create_issue` from `jira`:

```yaml
tools:
    - origin: instance
      server: gitlab
      name: [ "get_merge_request", "list_work_items" ]
    - origin: instance
      server: gitlab
      name: [ "list_work_items", "get_work_item" ]
    - origin: client
      server: jira
      name: [ "create_issue" ]
```

**An admin sets allow, ask or deny for GitLab MCP tools.** Tool governance exists today. It lists the tools of the
GitLab MCP server and sets one of three modes for each: Always Allow, Always Ask or Always Deny. In a governed session,
a mode applies whatever a flow's `tools:` list selects, and a tool set to Always Deny never reaches the agent.

**A new governance toggle blocks every client MCP tool.** Admin rules accept only tool names that GitLab knows. They
can't name a client tool: GitLab doesn't know it, and a client can rename it. One toggle blocks every tool with
`origin: client`, whatever a flow's `tools:` list selects. Which phase carries the toggle is an
[open question](#open-questions).

**`server` is the server's name, carried as its own field.** For a server that Rails configures, it's the configured
name, which Workhorse already uses as the `<server>_` prefix of each tool name. For a local server, it's the name in
`mcp.json`. It isn't the `serverInfo.name` that a server reports about itself: that name isn't unique across servers,
and a server can't be fully trusted to set it. DWS can't derive the server from the prefix, because a server name can
contain `_`. So the tool message carries the server name as its own field. Today `RunMCPTool`, the action that DWS
sends to run an MCP tool, carries only the prefixed name, and Workhorse finds the session by it. The new field is a
contract change in DWS, Workhorse, and the clients.

**`origin` tells whether an MCP tool comes from the instance or the client.** It has two values. `instance` is a
server that Rails configured, which includes `gitlab`, `orbit`, `slack` and the catalog servers. `client` is a tool
the client supplies. `origin` says who configured the server, not who owns it. Workhorse sets `origin` on each tool.
On a tool the client supplies it sets `client`, whatever the client sent, as it already clears the `trusted` flag
there (Rails sets `trusted` on `gitlab` and `orbit` tools, see [phase A](#phase-a-explicit-selectors)). So a client
can't choose its origin. Code that holds a DWS token, such as a CI job, could connect to DWS directly and send
`origin` itself. DWS can tell such a request apart, because it issued that token. That's why DWS reads `origin` only
on a request that Workhorse handled. On any other request every tool counts as `client`, and DWS drops them all. On a
GitLab version that sends `origin`, a tool message without `origin` counts as `client`. On an older version, DWS uses
the phase A rule instead (`trusted` plus the `<server>_<name>` prefix, see
[phase B](#phase-b-v1-support-and-client-servers), part 2). In the config, a selector without `origin` means
`origin: instance`. It never matches a client tool, so a local server needs an explicit `origin: client`. For example:

```yaml
tools:
    - origin: instance                 # a server that Rails configured
      server: gitlab
      name: [ "get_merge_request" ]
    - origin: client                   # a local server from the user's mcp.json
      server: jira
      name: [ "create_issue" ]
```

**`include` and `exclude` narrow a field.** `server` and `name` can each take an object with `include` and `exclude`
instead of a plain value. This `include` is a field inside a selector, not the flow's top-level `include` section. A
value matches when it matches `include` and doesn't match `exclude`. `exclude` works inside its own selector. It never
removes a tool that another entry adds. For example:

```yaml
tools:
    - origin: instance
      server:
          include: "*"                   # any server
          exclude: "gitlab"              # except gitlab
      name: [ "*" ]
    - origin: instance
      server: gitlab
      name:
          include: [ "list_*" ]          # the list tools from gitlab
          exclude: [ "list_projects" ]   # except this one
```

## Phased delivery

The Decisions are the end state. They ship in three phases, and
[epic 23110](https://gitlab.com/groups/gitlab-org/-/work_items/23110) tracks the release for each. A phase can ship
in more than one release. A config written for one phase stays valid in the next, because each phase adds to the syntax
and removes nothing.

### Phase A: explicit selectors

Phase A ships to the `experimental` version of the flow registry only. Later phases build on it.

- Introduce the `tools:` key in the `experimental` version of the flow registry. A selector only accepts `origin: instance`.
- `server` is `gitlab` or `orbit`, and `name` lists exact tool names. Any other `server`, or `origin: client`, is a
  config error.
- `server: gitlab` also selects native tools by exact name, so a `tools:` component can still declare them. DWS has no
  pair data yet (Phase C), so it looks for a native tool first and then for an MCP tool.
- Workhorse doesn't send a server name yet. So DWS selects a tool when the tool is `trusted` and its name is
  `<server>_<name>` (the phase A rule). Rails sets `trusted` only on `gitlab` and `orbit`. DWS does this only on a
  GitLab version that clears `trusted` on client tools (19.3.6, 19.4.3, and 19.5.0 or later). On any other version a
  selector selects nothing, and DWS logs it (whether that fails the flow is an open question). DWS also counts no tool
  as `trusted` on a request that Workhorse didn't handle.
- If two MCP tools from the instance, or two client tools, arrive with the same name, DWS drops every tool with that
  name and logs it ([!7721](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7721)).
  Workhorse routes a call by name alone, so DWS can't send a call to one of the tools. A client tool that shares a name
  with an instance tool still loses, as it does today: Workhorse runs the name it registered itself, so the instance
  tool stays. The drop replaces the current rule, which keeps one of the colliding tools within an origin.
- A component with a `tools:` list gets no automatic MCP tools, in every environment, and `tools: []` means no
  selectable tools. A component without `tools:` behaves as today.
- A flow that mixes `toolset:` and `tools:` components is a config error. `experimental` flows have no `include`
  section, so every component in them counts, except a `tool_name:` step.
- Rails adds an `AllowList` for the rollout of `tools:` to `experimental` flows, starting with one flow. Rails sends
  the GitLab MCP server to the flows on the list (see [Consequences](#consequences)).

### Phase B: v1 support and client servers

Phase B ships in two parts. Part 1 needs no Workhorse change. Part 2 needs the Workhorse contract change.

Part 1, no Workhorse change. The selectors use the phase A rule, so `server` is `gitlab` or `orbit` and `name` lists
exact tool names:

- The `v1` parser reads `tools:`. It rejects a flow that mixes `toolset:` and `tools:` components. It counts an
  `AgentComponent` with neither key as a `toolset:` component and ignores a `tool_name:` step. The check covers the
  components the author wrote, before the `include` section adds any, so a subagent from the workspace or the AI
  Catalog doesn't count.
- Rails adds a foundational flow (a flow GitLab ships) that doesn't rely on automatic MCP tools to the `AllowList`
  once its components use `tools:`, and sends the GitLab MCP server to the flows on the list. Existing `chat` and
  `chat-partial` flows using the `toolset:` key maintain existing functionality, and the shipped configs that rely on
  automatic MCP tools move in phase C. Custom flows get the server in part 2, once the Rails schema accepts `tools:`
  (see [Consequences](#consequences)).

Part 2, needs the contract change and Rails side change:

- DWS reads `server`, `origin`, and drops the phase A rule, on a GitLab version that sends them.
- On a GitLab version that doesn't send them, DWS keeps the phase A rule, with the same version check and the same
  `trusted` condition. Only `server: gitlab` and `server: orbit` select tools there. Any other selector for an MCP tool
  selects nothing, and DWS logs it. An instance that hasn't upgraded GitLab keeps its GitLab MCP tools when DWS ships
  this part.
- Any server name works. `origin: client` selects local servers, and `origin: instance` covers catalog servers and
  `slack`.
- DWS drops client tools on a request that Workhorse didn't handle.
- If `RunMCPTool` carries the server, DWS routes a call by server and name, and the drop rule for MCP tools that share a
  name ends. If it doesn't, the rule stays.
- `server` and `name` accept a bare `*`, so a flow can select every tool or every local server.
- `include` / `exclude` on both `server` and `name` fields. They take exact names and a bare `*`.
- Rails accepts `tools:` in custom flows. The AI Catalog flow schema allows the key, and the catalog tracking and audit
  code read it. Like the `v1` parser, the schema rejects a flow that mixes `toolset:` and `tools:` components, and it
  checks only the components the author wrote. Rails sends the GitLab MCP server to a custom flow that uses `tools:`.

### Phase C: pairs, new configs, tool protection, patterns, version and ops

- Pairs declared for each MCP tool that replaces a native tool, with pairing logic implemented in DWS.
- DWS ships a new major version of each shipped `chat` and `chat-partial` config that relies on automatic MCP tools.
  The new version uses `tools:`. DWS ships it first. A later GitLab release asks for the new major version, and only
  GitLab versions that send `origin` (phase B, part 2) ask for it. [Consequences](#consequences) says why a major
  version and not a minor one.
- Protection against a tool that its provider changes after the author wrote the flow. This could be a lock on the
  tool definition or an opt-in pin. Which of them is an open question (see [open questions](#open-questions)).
- A tool version in the selector. This is a separate question from the lock (see [open questions](#open-questions)).
- Patterns other than a bare `*` in `server` and `name`, such as `list_*`, and patterns inside `include` and `exclude`.
- `ops` on selectors.

## Consequences

- At the time of writing, 13 configs declare `chat` or `chat-partial`: 7 under
  `duo_workflow_service/agent_platform/v1/flows/configs/`, 3 under `ai/features/code/developer/config/`, 2 under
  `ai/features/insights/analytics_agent/config/` and 1 under `ai/features/chat/slack_assistant/config/`. `agentic_chat`
  ships `toolset: []` and relies on automatic MCP tools today.
- At the time of writing, three configs set tool options in `toolset:`: `fix_pipeline`, `security_review` and
  `workplan`. They stay on `toolset:` in every component until `ops` ships, because a selector carries no options and a
  flow can't mix the two keys. Until then such a flow can't give one component a local MCP server through `tools:`.
- A client can report any server name, including `gitlab`. A selector with `origin: instance` never matches a client
  tool, so a client tool that reports `gitlab` can't pass as a tool from the instance.
- In phase A a client could give a tool a name that starts with `gitlab_`. DWS counts only `trusted` tools, and only on
  a GitLab version that clears `trusted` on client tools, so a client tool can't pass as a `gitlab` tool. On other
  versions a selector selects nothing.
- Rails checks flow configs against a schema that lists only `toolset`, and the catalog tracking and audit code reads
  only `toolset`, so both must accept `tools:`. Custom flows are `v1`, so they also wait for the `v1` parser.
- A renamed or removed tool stops matching. DWS logs an entry that selects nothing. The GitLab MCP server keeps an old
  tool name as an alias, but `tools/list` returns only the current names, and selectors match those names. A selector
  with an old name selects nothing, although the server still accepts that name. The exception is an old native name
  that an MCP tool replaces: the pair covers it. A CI test checks that the tool names in `v1` configs exist in the
  registry. Selectors and the `experimental` configs need a matching check. CI covers only configs that GitLab ships,
  so how GitLab finds a customer flow whose selector selects nothing is an open question.
- Rails pre-approves GitLab MCP tools by their `readOnlyHint`, so `name: ["*"]` on `gitlab` approves a tool GitLab ships
  next.
- Rails sends the GitLab MCP server only to Agentic Chat, to agents with their own tool list, to the flows on the
  `AllowList`, and from phase B part 2 to custom flows that use `tools:`. Other flows get none, so `server: gitlab`
  selects native tools only there.
- A `toolset:` component in a `chat` or `chat-partial` flow keeps getting the tools of every connected MCP server
  automatically, as it does today. Phase B doesn't add the GitLab MCP server to a `toolset:` flow that doesn't get it
  today. Agentic Chat, AI Catalog agents, and foundational agents whose AI Catalog version lists MCP tools keep it, and
  there a native tool and its MCP copy both stay until pairs ship (phase C).
- The contract change (phase B, part 2) goes in this order: the proto in ai-assist, a new `gopb` version, the bump in
  the GitLab repository (Workhorse), then DWS reads the new fields. A GitLab version without the change sends none of
  them, so DWS keeps the phase A rule for `gitlab` and `orbit` there. Workhorse only forwards client tools, so the
  clients (`gitlab-lsp` and the CLI) must also send the name of a local server. A selector for a local server, such as
  `jira`, selects nothing for users on a client release that doesn't send it.
- A GitLab version that doesn't send `origin` can't select local MCP servers. DWS ships a new major version of each
  shipped `chat` or `chat-partial` config that relies on automatic MCP tools. The new version uses `tools:` and names
  each tool once. DWS resolves each name through the pair, so a session without the GitLab MCP server keeps the native
  tool. Rails asks for most of these configs by range, such as `^1.0.0`, and DWS serves the highest version in the
  range. So a minor version would reach older GitLab versions too. Only a GitLab version that sends `origin` asks for
  the new major version. Older versions keep the current definition with `toolset:`, so they keep their local MCP
  servers. A flow that stays on `toolset:` keeps the problems under [Context](#context). This ADR proposes docs that
  show how to move it to `tools:`.
- A native tool and an MCP tool can share a name, such as `list_work_items`. Under `server: gitlab` one name selects one
  tool, and a flow gets the MCP tool once its pair is on. DWS logs which tool it chose.
- Tool governance covers the tools of the GitLab MCP server, and a rule on a native tool name also covers the MCP tool
  that replaces it. A session without governance applies no rules to native tools, as in IDE and Duo CLI sessions.
  Local tool governance brings the rules to those sessions when it ships. A catalog server has no per-tool rule. A
  group Owner can still block a whole catalog server in the MCP Registry.
- This ADR relies on the pairs in the design doc that makes the MCP server the source of DAP tools. That doc is still
  proposed.

## Alternatives considered

| Alternative                                                                  | Why not                                                                                                                                                                                                      |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A flag such as `accept_local_mcp_servers`                                    | It can conflict with selectors: with the flag on and `server: jira`, it's unclear whether other local servers get in.                                                                                        |
| Automatic MCP tools per environment                                          | `tools: []` would behave differently in each environment.                                                                                                                                                    |
| A `deny` entry in the config                                                 | A ban that must hold whatever a config says is an admin rule (see [Decisions](#decisions)).                                                                                                                                |
| `server: client` for all local servers                                       | A flow couldn't select one local server and not another.                                                                                                                                                     |
| Derive `server` or `origin` in DWS from `trusted` and the `<server>_` prefix | Not as the end state: a server name can contain `_`, so the prefix is ambiguous. Phase A uses it for `gitlab` and `orbit` only, where Rails sets `trusted`. Phase B keeps it only for older GitLab versions. |
| Exact names only for `gitlab` and `orbit` until a lock exists                | The syntax would differ by server, which confuses authors. An author who needs a tight tool set lists exact names.                                                                                           |
| Allow, ask or deny levels for client tools                                   | Admin rules accept only tool names that GitLab knows, so they can't name a client tool. One toggle blocks every client tool.                                                                                 |
| Reuse `client_injected` instead of adding `origin`                           | DWS reads a missing `client_injected` as `false`, which means from the instance. A missing `origin` counts as `client`.                                                                                      |
| A selector with no `server` for native tools                                 | An author would track whether GitLab implements a tool natively or in the GitLab MCP server, and the line would move when GitLab migrates the tool.                                                          |
| Name the MCP tool, or both tools, in the shipped `tools:` definitions        | Naming only the MCP tool leaves a session without the GitLab MCP server with no tool. Naming both lists two names for one tool. One name resolved through the pair avoids both.                              |
| Plain names in `tools:`, as in `toolset:`                                    | Two ways to write an entry confuse authors. The cost: a flow with a component that needs a tool option stays on `toolset:` until `ops` ships (phase C).                                                      |
| Mixing `toolset:` and `tools:` components in one flow                        | A reader can't tell which tools the flow gets, because a `toolset:` component keeps its automatic MCP tools and a `tools:` component gets none. It also delays the end of `toolset:`. The cost: a custom flow can't move one component at a time. |

## Open questions

1. How a flow detects that a tool changed after the author wrote the flow (phase C). A server can change the
   description or input schema of a tool without renaming it, and the description goes straight into the prompt. Two
   proposals exist: a lock and an opt-in pin. Open points: whether to take one or both.
    - Freeze the tool definition. This works for a tool from any server.
        - A lock,
          [proposed by `allparts`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2667#note_4047050521).
          When the author publishes the flow, the flow stores a hash of each tool's description, input schema and
          annotations. DWS doesn't admit a tool whose hash no longer matches until the author publishes the flow again.
          It doesn't bring back the old tool. Whether the flow then fails depends on the required or optional question
          below. Detecting and rejecting a changed tool is more robust than showing the model a description that no
          longer matches it
          ([source](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7550#note_4157700930)).
        - An opt-in pin,
          [suggested by Tim](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2667#note_4049451318),
          similar to the
          [pin in Anthropic's MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector#pin-mcp-tool-list).
          The flow stores a copy of the tool list, with each tool's name, description and input schema, and uses the
          copy. The flow doesn't ask the server for its tools again. The model sees the same tool when the server
          changes it, and the pin doesn't detect the change. The stored copy can differ from the tool that runs, so the
          model can call a tool with a description or input schema that no longer matches it.
1. Whether a selector can name a tool version, as
   [proposed by `mikolaj_wawrzyniak`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2412#note_3611889751)
   (phase C). GitLab can replace or remove a tool, which breaks a component that has the tool's arguments written in
   its config. A selector names a version, or a range such as `version: "^1.0"`, so a flow can keep a legacy version of
   a tool. Without `version`, the flow gets the latest. A strict component, such as a deterministic step, keeps working
   when GitLab replaces a tool. The provider must keep more than one version of a tool, and MCP has no per-tool
   version. Other frameworks rely on aliases
   ([research](https://gitlab.com/gitlab-org/gitlab/-/work_items/570762#note_3514552813)). So this applies only to tools
   that GitLab owns. It trusts the version that the provider reports, so it doesn't detect a change that keeps the same
   version. A version is a separate concern from a lock
   ([source](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7550#note_4157700930)).
   A flow author uses it to declare how much change the flow accepts: a strict version calls for a lock, and a
   permissive range may not. A version also lets a tool author ship a breaking change without breaking a flow that
   names an older version. Open points: which tools carry a version, how GitLab keeps and retires an old version, and
   how a version fits with the lock.
1. How to apply `ops`, for native tools and for MCP tools. The three configs under Consequences that set tool options
   use native tools, and the cut-off for `toolset:` waits on `ops`.
1. How subagents inherit MCP tools. Workspace subagents get a `toolset:` today
   (`duo_workflow_service/agent_platform/v1/catalog/sources/promotion.py`), so the cut-off for `toolset:` also covers
   them.
1. How to recognize GitLab-owned servers and trust `server`.
1. Whether `RunMCPTool` carries the server and the plain tool name, or keeps the prefixed name. The answer also
   decides whether DWS keeps dropping tools that share a name. Carrying the raw fields lets a downstream system
   compose any derivative it needs, but it can't rebuild the raw attributes from a derivative without coupling to the
   upstream system
   ([source](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7550#note_4157700948)).
1. What DWS does with a client tool that has no server name, for example from a client release that doesn't send it.
1. Whether the `chat` and `chat-partial` configs that GitLab ships need automatic MCP tools, for example whether GitLab
   supports client MCP tools in them. If they do, the new `tools:` definition of each config declares them. If they
   don't, users lose their local MCP servers in those flows, so the notice in the next question applies to them too.
1. For custom flows that rely on automatic MCP tools, how customers learn about the change in behavior. The standard
   deprecation process announces the removal of `toolset:`. It doesn't say that `tools:` adds no MCP tools, so a flow
   must declare the ones it needs before the cut-off.
1. Which native tools `server: gitlab` includes. The tools a flow has enabled include the default tools (plan,
   task and UI tools), the no-op tools (hand-over and user clarification) and the capability tools (shell and web
   search), so `name: ["*"]` selects them. Web search is pre-approved. The flow's privileges still cap the list. A tool
   that a component adds for its own features is internal, so `name: ["*"]` never matches it (see
   [Decisions](#decisions)). Whether hand-over and user clarification should count as internal is open: a toolset
   includes them today only when it lists them.
1. Which phase carries the one toggle for client tools.
1. Whether a selector can name a fallback tool, a different tool used when the first isn't available. Under
   `server: gitlab`, an entry that names a native tool already falls back to it when the paired MCP tool doesn't
   arrive, so this asks about other cases. If it can, a fallback for an `instance` tool never comes from a `client`
   tool. Open points: the syntax, how it fits "Entries add up", and whether a fallback may come from any other source
   that is less trusted than the first tool.
1. Whether tools can carry tags or belong to groups, and whether a selector can select by them. A tag would work the
   same way for native tools and for tools from any server. Open points: who sets a tag, and how it relates to the
   groups that exist today.
1. Whether a selector can be marked required or optional. A required selector that selects nothing fails the flow at
   start. DWS skips an optional one, as it skips unknown names today. The default is open.
1. Whether and how GitLab finds a customer flow whose selector selects nothing, for example after GitLab renames or
   removes a tool. GitLab could check the flow when a user saves it, review the AI Catalog flows on a cadence and flag
   their owners, or do both. A check at save time can't cover a local server. Open points: where the flag shows, and
   whether the same review also flags flows that still use `toolset:` before the cut-off.
1. Whether DWS maps an alias to its tool for `server: gitlab`. The GitLab MCP server resolves an alias when a
   tool is called, but `tools/list` leaves aliases out, so DWS can't see them. The pair that the design doc carries in
   `_meta` gives DWS the aliases that replace a native tool.
   [GitLab work item 609451](https://gitlab.com/gitlab-org/gitlab/-/work_items/609451)
   plans a migration for stored tool names when MCP tools are renamed or removed. Open points: where the list of aliases
   between MCP tools comes from, and whether a selector with an old name warns or selects the tool.
1. When to set the cut-off for the deprecation of the legacy `toolset:` key. It can't come before `ops` ships, because
   configs that set tool options can't move to `tools:` without it.
1. Whether pairs apply to a `tool_name:` step (phase C). The step calls its tool with fixed arguments, so the paired
   MCP tool would have to accept the same arguments.
