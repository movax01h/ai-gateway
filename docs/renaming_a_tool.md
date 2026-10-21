# Renaming a tool

This guide covers renaming an existing tool's `name` (the string an LLM uses to call it, for example
`run_command`). It assumes the tool's behavior stays the same.

If you're also changing behavior, read [Adding a New Tool](adding_new_tool.md) too.

If you wish to change the behavior of an existing tool without updating its tool name,
read [Tool Supersession](adding_new_tool.md#4-tool-supersession-optional).

A tool name is persisted in several places outside the code that defines it. A rename that only changes the class
touches none of these, and silently breaks the ones tied to the old name.

## AI Gateway (Duo Workflow Service) side

1. **Rename the tool file and class**, if the file is named after the tool:

   ```shell
   git mv duo_workflow_service/tools/old_tool_name.py duo_workflow_service/tools/new_tool_name.py
   ```

1. **Update the `name` field** on the tool class in that file:

   ```python
   class YourTool(DuoBaseTool):
       name: str = "new_tool_name"  # was "old_tool_name"
   ```

1. **Rename any input field that describes the old name**, if one exists (for example
   `previous_session_id` → `session_id`), and update its `Field(description=...)` text.

1. **Update the tool's `description`** to match the new name and behavior. This is read by the LLM, so an
   inconsistent name and description confuses it.

1. **Update the import** in `duo_workflow_service/tools/__init__.py`:

   ```python
   from .new_tool_name import *  # was: from .old_tool_name import *
   ```

1. **Update every workflow tool list** that references the tool by its string name. These are defined per workflow under
   `duo_workflow_service/workflows/*/workflow.py`, for example `CHAT_SESSION_CONTEXT_TOOLS` in `chat/workflow.py` and
   `CONTEXT_BUILDER_TOOLS`/`PLANNER_TOOLS` in `software_development/workflow.py`.

1. **Update flow config YAML files** under `duo_workflow_service/agent_platform/v1/flows/configs/` that list the tool
   or mention it in a prompt.

1. **Update tests**: `tests/duo_workflow_service/components/test_tools_registry.py`, the tool's own test file (rename
   it alongside the tool file), and any workflow test that asserts on the tool list.

1. **Search for the old name across the repo** before opening the MR, to catch call sites outside the lists above:

   ```shell
   grep -rn "old_tool_name" .
   ```

## Rails (AI Catalog) side

The AI Catalog keeps its own built-in tool registry, independent of the AI Gateway's. If the tool is registered there,
update it to match:

1. **Update the tool's `name` and `title`** in `ee/lib/ai/catalog/built_in_tool_definitions.rb`. Keep the numeric
   `id` unchanged. `Ai::Catalog::ItemVersion#def_tools` persists tool selections by that number, not by name, so
   changing it would silently drop the tool from every agent that already selected it.

1. **Update the tool's key** in the privilege group mapping in `ee/lib/ai/tool_rules/registry.rb`. This one is
   functional, not cosmetic: a stale key here drops the tool out of `all_tool_names`, `default_permission_of`,
   `category_for`, and `action_type_of`. Also check `MCP_TOOL_NAME_FOR` in the same file. If the tool appears there
   as a key or a value, rename it too, or `to_mcp_tool_names` silently stops rewriting it.

1. **Update the public tools table** in `doc/user/duo_agent_platform/agents/tools.md`, keeping alphabetical order by
   title. Amend with a `<history>` noting the rename and the milestone in which it happened.

1. **Update stale comments and specs** referencing the old name, for example in
   `ee/app/graphql/ee/types/query_type.rb` and `ee/spec/requests/api/graphql/ai/catalog/built_in_tools_spec.rb`.

1. **Add a post-deploy data migration** if admins can set governance rules on the tool (`ai_tool_rules`). Rules are
   keyed by the tool's name string, not its numeric identifier, so a code-only rename orphans any existing
   Allow/Deny/Ask rule under the old name. `Ai::ToolRule` validates `tool_name` against the same tool list, so an
   orphaned row fails that validation on any save. The settings UI doesn't render it, and only a console can
   re-save or delete it. The tool itself falls back to its group default, which is auto-allow if it's in
   `DEFAULT_PREAPPROVED_GROUPS`.

   The migration should, for every `ai_tool_rules` row on the old name:
    - Rename it to the new name, if no row already exists for that namespace/project under the new name.
    - Otherwise, merge it into the existing new-named row. Keep the columns the new-named row already has explicitly
      set (an admin may have set them before or during the migration). Backfill only the columns it left unset, using
      values from the old row. Delete the old row once merged.
    - Handle the race where a new-named row appears between your conflict check and the update. For example, rescue
      the unique-index violation and retry the merge.

> [!note]
> Between the rename deploying and the migration running, the renamed tool shows the privilege group's default instead
> of the admin's actual setting. For example, renaming `get_previous_session_context` to `get_session_context`: if an
> admin had set `get_previous_session_context` to `[deny, null, deny]`, after the code deploy `get_session_context`
> shows `[allow, allow, allow]` instead: the `read_only_gitlab` group default, since no row exists yet under the new
> name. If an admin notices this and sets a value, that creates a new database row, resulting in a duplicate the
> migration needs to resolve. Self-managed admins can extend this window further with
> `SKIP_POST_DEPLOYMENT_MIGRATIONS=true`.

## Rollout order

Target the same milestone for both MRs. Until the Rails MR merges, the AI Catalog, the public tools table, and
`ai_tool_rules` governance all still key the tool by its old name.

The AI Catalog also passes tool selections to the AI Gateway by name. If Rails advertised the new name before the
AI Gateway understood it, a custom agent selecting that tool would fail to get it. Merge the AI Gateway side first
to avoid that, then open the Rails MR right away. Keep the gap between the two MRs as short as review allows.

> [!note]
> Once the AI Gateway re-name happens there will be a misalignment between the tool name in AI Gateway and Rails.
> In that window, an admin's Allow, Ask, or Deny on the tool silently stops enforcing in the governance UI.

If either MR risks missing the milestone, postpone the rename. Target the next milestone you feel confident both
will land in together.
