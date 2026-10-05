# Adding and moving features

New prompts and flows go under `ai/features/<domain>/<feature>/`. Prompts that several
features use go under `ai/shared/<name>/`. The legacy roots,
`ai_gateway/prompts/definitions/` and `duo_workflow_service/agent_platform/v1/flows/configs/`,
are frozen: a lint rejects new directories there.

The steps below work today. [Module boundaries](module_boundaries.md) describes the target
design, and some parts of that design are not implemented yet.

## Where files go

| Asset | Location | Registered as |
|-------|----------|---------------|
| Prompt | `ai/features/<domain>/<feature>/prompts/<family>/<version>.yml`, plus `.jinja` includes | prompt IDs `<feature>` and `<domain>/<feature>` |
| Shared prompt | `ai/shared/<name>/prompts/<family>/<version>.yml`, plus `.jinja` includes | prompt ID `<name>` |
| Flow configuration | `ai/features/<domain>/<feature>/config/<version>.yml` | flow ID `<feature>` |
| Feature tools | `ai/features/<domain>/<feature>/components/__init__.py`, in `FEATURE_TOOLS` | tools in the tools registry |
| Tests | `ai/features/<domain>/<feature>/tests/` | collected by `pytest` |
| Serving surface | `ai/features/<domain>/<feature>/serving.py`, in `SERVING_SURFACE` | not read yet |

Follow these rules:

- The directory name is the ID. Feature and shared directory names must be unique across
  `ai/features/` and `ai/shared/`. The AI Gateway fails at boot on a duplicate, and
  `tests/test_feature_ids_unique.py` catches it in CI.
- Name a domain after the capability, not after a team. `cli` and `insights` exist today.
  `foundational_agents` holds flows that are fetched from the AI Catalog at image build, so
  do not add bundled features there.
- A nested prompt ID fixes its domain. `chat/react` lives in `ai/features/chat/react/`,
  because the second registered ID, `<domain>/<feature>`, must match the name that callers
  use.
- The two prompt IDs of a feature point at the same `prompts/` directory. A flow that needs
  a second file-based prompt defines it inline in the flow configuration, or puts it in a
  second feature directory.
- Use Python identifiers for domain and feature names. The tools registry skips a feature
  whose name it cannot import.

### Shared prompts

A prompt that several features use goes to `ai/shared/<name>/prompts/` and registers as
`<name>`. Include-only directories move the same way, and their include strings stay the
same: `ai/shared/common/prompts/` serves `common/branch_naming/1.0.0.jinja`.

Keep `ai/shared/` small. A definition that only one feature uses belongs in that feature.

### Not supported yet

Response schemas cannot live in a feature or in `ai/shared/`. They stay in
`ai_gateway/response_schemas/definitions/`.

## Add a version to a legacy feature

A new version of a feature that still lives in a legacy root goes next to its existing
versions. The lint allows a new file in a directory that the allowlist already lists.
`scripts/update_prompt_version.sh` copies the latest version into the same directory.

Do not put the new version under `ai/features/`. The prompt registry resolves a prompt ID
from one root only, so a feature directory for that ID hides every legacy version.

A new family directory, for example `<feature>/claude_sonnet_5/`, is a new directory. The
lint rejects it, so move the feature first.

## Add a prompt feature

1. Create the feature directory. Add empty `__init__.py` files to the domain and the
   feature, like the existing features:

   ```plaintext
   ai/features/<domain>/__init__.py
   ai/features/<domain>/<feature>/__init__.py
   ai/features/<domain>/<feature>/serving.py
   ai/features/<domain>/<feature>/prompts/base/1.0.0.yml
   ai/features/<domain>/<feature>/prompts/system/1.0.0.jinja
   ```

1. Write the prompt YAML. The format is the same as in the legacy root. See the
   [prompt configuration reference](aigw_prompt_registry.md#ai-gateway-prompt-configuration-reference).
   An include starts with the prompt ID:

   ```yaml
   prompt_template:
     system: |
       {% include '<feature>/system/1.0.0.jinja' %}
   ```

1. Declare the serving surface in `serving.py`. For a prompt that the AI Gateway serves
   over REST:

   ```python
   from duo_workflow_service.agent_platform.serving_surface import ServingSurface

   SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
   ```

   Nothing reads this declaration yet.

1. Add the feature or its domain to `.gitlab/CODEOWNERS`.

1. Run the checks:

   ```shell
   make check-legacy-roots
   poetry run pytest tests/test_feature_ids_unique.py
   ```

1. Call the prompt. Follow [Verifying a prompt move locally](module_boundaries.md#verifying-a-prompt-move-locally)
   with your prompt ID.

## Add a flow feature

1. Create the flow configuration at `ai/features/<domain>/<flow>/config/1.0.0.yml`. Add
   empty `__init__.py` files to the domain and the feature. The flow ID is `<flow>`. For the
   configuration format, see [Flow Registry v1](flow_registry/v1.md).

1. Keep prompts inline in the flow configuration. See
   [Locally Defined Prompts](flow_registry/v1.md#locally-defined-prompts). If the flow needs
   one file-based prompt, put it in `prompts/` in the same directory and set `prompt_id` to
   the flow ID:

   ```plaintext
   ai/features/<domain>/<flow>/
     __init__.py
     serving.py
     config/1.0.0.yml          # prompt_id: "<flow>"
     prompts/base/1.0.0.yml
   ```

1. Put the tools that only this flow uses in `components/`. Export them from
   `components/__init__.py` in a `FEATURE_TOOLS` mapping, keyed by the agent privilege that
   gates them. The tools registry merges them at startup:

   ```python
   FEATURE_TOOLS: dict[str, list[type[BaseTool]]] = {
       "read_only_gitlab": [MyTool],
   }
   ```

1. Put the tests in `tests/`. `pytest` collects `ai/features/*/*/tests`.

1. Declare the serving surface in `serving.py`, with `transport="grpc"` and
   `deployable="dws"`.

1. Add the feature or its domain to `.gitlab/CODEOWNERS`.

1. Regenerate the flow graph documentation and commit the result. The
   `check-duo-workflow-docs` job fails when they are out of date:

   ```shell
   make duo-workflow-docs
   ```

1. Run the flow with the GitLab Duo CLI:

   ```shell
   duo run \
     --flow-config ai/features/<domain>/<flow>/config/1.0.0.yml \
     --flow-config-schema-version v1 \
     -g "Your goal description here"
   ```

## Move a legacy feature

Before you start, check where the feature goes:

- A prompt ID with a slash moves to the domain named by its first part. `chat/react` goes
  to `ai/features/chat/react/`, and `workflow/planner` goes to `ai/features/workflow/planner/`.
- A shared prompt goes to `ai/shared/<name>/prompts/`. This includes the include-only
  directories such as `common/`.
- A feature with response schemas moves its prompts and its configuration. Its schemas stay
  in `ai_gateway/response_schemas/definitions/`.
- Ask the group that owns the feature to review the move.

Move the whole feature in one merge request:

1. Move every version with `git mv`, so Git keeps the history. Create the parent directory
   first, then run the line that applies:

   ```shell
   # a prompt
   git mv ai_gateway/prompts/definitions/<feature> ai/features/<domain>/<feature>/prompts
   # a nested prompt ID, for example chat/react
   git mv ai_gateway/prompts/definitions/chat/react ai/features/chat/react/prompts
   # a shared prompt or include directory
   git mv ai_gateway/prompts/definitions/<name> ai/shared/<name>/prompts
   # a flow
   git mv duo_workflow_service/agent_platform/v1/flows/configs/<feature> ai/features/<domain>/<feature>/config
   ```

   Includes such as `<feature>/system/1.0.0.jinja` or `chat/react/system/1.0.0.jinja` keep
   working without a change.

1. Run `make check-legacy-roots`. From `scripts/legacy_roots_allowlist.txt`, delete each line
   that it reports as stale.

1. Add the `__init__.py` files, `serving.py`, and the CODEOWNERS entry, as for a new feature.
   A shared prompt needs only the CODEOWNERS entry.

1. Move the tools of the feature into `components/` and declare them in `FEATURE_TOOLS`.
   Remove them from the static tool lists in `duo_workflow_service/components/tools_registry.py`.
   Keep a deprecation shim in the old module for callers outside this repository, for example
   `duo_workflow_service/tools/get_glql_schema.py`. Update every import in this repository,
   including `agent_tests/`.

1. Move the tests of the feature into `tests/`.

1. Find the references to the old path and update them. CI rules, scripts, and documentation
   often name it:

   ```shell
   git grep -n -e "definitions/<feature>" -e "flows/configs/<feature>"
   ```

1. For a flow, run `make duo-workflow-docs` and commit the result.

1. Check that the feature serves the same versions from the new location. For a prompt,
   follow [Verifying a prompt move locally](module_boundaries.md#verifying-a-prompt-move-locally).
   For a flow, run this command on `main` and on your branch, and compare the output:

   ```shell
   poetry run python -c "from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig, discover_feature_flow_configs, list_flow_configs; discover_feature_flow_configs(); print(sorted(e['flow_version'] for e in list_flow_configs(FlowConfig) if e['flow_identifier'] == '<feature>'))"
   ```

Two completed moves show the result:

- [`analytics_agent`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/commit/af40a8846aa18834fd4d56b21ceaf5e2c42507cc)
  is a flow with its own tools and tests.
- [`glab_ask_git_command`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/commit/80d178887f48fef8cf1eb02ec74a6a3c7a7dab7d)
  is a prompt feature. This commit also added prompt discovery, so only its file moves apply
  to a new move.

## The legacy-roots lint

`make check-legacy-roots` compares the directories that hold files in the legacy roots with
`scripts/legacy_roots_allowlist.txt`. A new file in a listed directory passes. The lint
fails in these cases:

- A new directory in a legacy root. Create the feature under `ai/features/`, or move the
  existing feature first.
- A listed directory that holds no files of its own. Delete that line. A move always leaves
  such lines.
- In CI, a line added to the allowlist, compared with the target branch. The allowlist only
  gets shorter.

The `lint:legacy_roots` CI job and the `lefthook` pre-commit hook run the lint.
