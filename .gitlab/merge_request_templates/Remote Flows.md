## What does this merge request do and why?

<!-- Briefly describe what this merge request does and why. -->

%{first_multiline_commit}

## How to set up and validate locally

1. [Set up Duo Agent Platform](https://gitlab-org.gitlab.io/gitlab-development-kit/howto/duo_agent_platform/)
1. [Set up the Flow Registry in your GDK](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/main/docs/flow_registry?ref_type=heads#debugging-flows)
1. [Run a Remote Flow](https://docs.gitlab.com/user/duo_agent_platform/flows/#available-flows)
1. Check your Flow Output in `Automate > Agent sessions`

## Merge request checklist

- [ ] Tests added for new functionality. If not, please raise an issue to follow up.
- [ ] Documentation added/updated, if needed.
- [ ] If this change requires executor implementation: verified that issues/MRs exist for
  both [Go executor](https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-executor)
  and [Node executor](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp) or confirmed that changes are
  backward-compatible and don't break existing executor functionality.

/label ~"group::agent foundations"
/label ~"section::ai"
/label ~"devops::ai-powered"
/label ~"Category:Duo Agent Platform"

<!-- Select a type -->
<!-- /label ~"type::bug" -->
<!-- /label ~"type::feature" -->
<!-- /label ~"type::maintenance" -->

/assign me
