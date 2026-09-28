---
name: rename-tool
description: >
  Rename an existing Duo Workflow Service tool's `name` across both the AI Gateway
  and Rails (AI Catalog) repos, including the post-deploy governance migration.
  Use when asked to rename a tool, to change a tool's `name` string, or to open
  the companion ai-assist and gitlab-org/gitlab MRs for a tool rename.
argument-hint: "[old-tool-name] [new-tool-name]"
---

# Renaming a tool

Two repos, two MRs. The full step-by-step process lives in
[`docs/renaming_a_tool.md`](../../../docs/renaming_a_tool.md) — read it before making any
changes, don't re-derive the steps from scratch here.

## Before you start

Confirm with the user: the old tool name, the new tool name, and whether this is a
straight rename or should be a
[Tool Supersession](../../../docs/adding_new_tool.md#4-tool-supersession-optional)
instead (same name, different implementation: that's a different guide, not this one).

## Where the two repos are

- **AI Gateway** — this repo.
- **Rails** — `gitlab-org/gitlab`. If not already checked out, look for it as a sibling
  GDK checkout at `$GDK_ROOT/gitlab` before asking the user where it is.

## Workflow

1. Read `docs/renaming_a_tool.md` in full.
2. Make the AI Gateway changes (tool class, workflow tool lists, flow configs, tests)
   and open that MR first.
3. Make the Rails changes (catalog entry, privilege group mapping, public tools table,
   the post-deploy data migration for existing `ai_tool_rules` rows) and open that MR,
   targeting the same milestone.
4. Cross-link the two MRs in their descriptions.

Don't skip the migration step even if no admin has configured governance rules on the
tool yet. The doc explains why it's still required.

If you need a reference implementation use:

- AI Gateway: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6283
- Rails: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/254136
