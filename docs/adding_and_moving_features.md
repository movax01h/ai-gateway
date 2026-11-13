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
| One of several prompts of a feature | `ai/features/<domain>/<feature>/prompts/<directory>/<family>/<version>.yml`, listed in `prompts/subprompts_index.yml` | the prompt ID that the index lists |
| Shared prompt | `ai/shared/<name>/prompts/<family>/<version>.yml`, plus `.jinja` includes | prompt ID `<name>` |
| Flow configuration | `ai/features/<domain>/<feature>/config/<version>.yml` | flow ID `<feature>` |
| Feature tools | `ai/features/<domain>/<feature>/components/__init__.py`, in `FEATURE_TOOLS` | tools in the tools registry |
| Tests | `ai/features/<domain>/<feature>/tests/` | collected by `pytest` |
| Serving surface | `ai/features/<domain>/<feature>/serving.py`, in `SERVING_SURFACE` | the flows that the Duo Workflow Service serves |

Follow these rules:

- The directory name is the ID. Feature and shared directory names, and the prompt IDs in
  each `subprompts_index.yml`, must be unique across `ai/features/` and `ai/shared/`.
  `tests/test_feature_ids_unique.py` catches a duplicate in CI. When both duplicates
  register a prompt, the AI Gateway also fails at boot.
- Name a domain after the capability, not after a team. `cli` and `insights` exist today.
  `foundational_agents` holds flows that are fetched from the AI Catalog at image build, so
  do not add bundled features there.
- A nested prompt ID fixes its domain. `chat/react` lives in `ai/features/chat/react/`,
  because the second registered ID, `<domain>/<feature>`, must match the name that callers
  use.
- The two prompt IDs of a feature point at the same `prompts/` directory. A flow with
  several file-based prompts lists them in an index. See
  [Several prompts in one feature](#several-prompts-in-one-feature).
- Use Python identifiers for domain and feature names. The tools registry skips a feature
  whose name it cannot import.

### Several prompts in one feature

A flow with several file-based prompts keeps each prompt in a directory in `prompts/`, and
lists it in `prompts/subprompts_index.yml`. Each line maps a prompt ID to its directory:

```yaml
# ai/features/pipeline/fix_pipeline/prompts/subprompts_index.yml
fix_pipeline_create_plan: create_plan
fix_pipeline_execution: execution
```

```plaintext
ai/features/pipeline/fix_pipeline/
  config/1.0.0.yml                # prompt_id: "fix_pipeline_create_plan", ...
  prompts/
    subprompts_index.yml
    common/job_log/1.0.0.jinja    # include-only, not in the index
    create_plan/
      base/1.0.0.yml
      system/1.0.0.jinja
    execution/
      base/1.0.0.yml
      system/1.0.0.jinja
```

Follow these rules:

- Each ID in the index registers its directory. A directory that the index does not list
  registers no prompt ID. The feature's own IDs, `<feature>` and `<domain>/<feature>`,
  still point at `prompts/`.
- The prompt ID namespace is flat, so start a new ID with the feature name. A prompt that
  moves from a legacy root keeps its ID.
- An include starts with the listed ID, for example
  `fix_pipeline_create_plan/system/1.0.0.jinja`. A file in a listed directory resolves only
  under the listed ID, so `fix_pipeline/create_plan/system/1.0.0.jinja` does not resolve.
  A file outside the listed directories resolves under the feature ID, for example
  `fix_pipeline/common/job_log/1.0.0.jinja`.
- Write one `<prompt_id>: <directory>` entry on each line, with letters, digits, and
  underscores only, and one space or more after the colon. Comments, blank lines, and a
  `---` before the first entry are allowed. The AI Gateway and the scripts read the file
  without a YAML library, so other YAML syntax fails.
- The AI Gateway fails at boot when an entry is malformed, or when the listed directory has
  no `base/<version>.yml` file. `tests/test_feature_ids_unique.py` fails when a
  `prompts/<directory>/base/` directory holds a `.yml` file and the index has no entry for
  `<directory>`.

### Shared prompts

A prompt that several features use goes to `ai/shared/<name>/prompts/` and registers as
`<name>`. Include-only directories move the same way, and their include strings stay the
same: `ai/shared/common/prompts/` serves `common/tool_output_security/1.0.0.jinja`.

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

   A prompt that the AI Gateway serves over HTTP sets `models_from` in each version. See
   [Models of prompts served over HTTP](aigw_prompt_registry.md#models-of-prompts-served-over-http).

1. Declare the serving surface in `serving.py`. For a prompt that the AI Gateway serves
   over REST:

   ```python
   from duo_workflow_service.agent_platform.serving_surface import ServingSurface

   SERVING_SURFACE = [ServingSurface(transport="rest", deployable="aigw")]
   ```

   The prompt routes do not read this declaration yet. A test checks that a feature whose
   prompt versions set `models_from` declares it.

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

   If the flow needs several file-based prompts, list them in an index. See
   [Several prompts in one feature](#several-prompts-in-one-feature).

   Do not set `models_from` in these prompts: the flow selects its models. See
   [Models of prompts served over HTTP](aigw_prompt_registry.md#models-of-prompts-served-over-http).

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
   `deployable="dws"`. Without it, the Duo Workflow Service fails at boot.

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

### Experimental-tier flows

A flow on the **experimental** schema tier lives in a `config/experimental/`
subfolder of the same feature. The folder picks the tier: `config/<version>.yml`
is v1, `config/experimental/<version>.yml` is experimental.

```plaintext
ai/features/<domain>/<flow>/
  prompts/                    # shared by both tiers
  config/
    1.0.0.yml                 # version: v1
    experimental/
      1.0.0.yml               # version: experimental
```

- The `version:` field must match the folder (`v1` in `config/`, `experimental`
  in `config/experimental/`); the tier's `FlowConfig` validates it.
- The two tiers keep independent version spaces, so the same `<version>.yml` name
  can exist in both without clashing.
- Promote a flow from experimental to v1 with `git mv config/experimental/X.yml
  config/X.yml` and flip its `version:` to `v1`.
- The machinery is `BaseFlowConfig.FEATURE_CONFIG_SUBDIR` (`""` for v1,
  `"experimental"` for experimental); the config listers are tier-agnostic.

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
   # one of several prompts of a flow, then list it in prompts/subprompts_index.yml
   git mv ai_gateway/prompts/definitions/<prompt_id> ai/features/<domain>/<feature>/prompts/<directory>
   # a shared prompt or include directory
   git mv ai_gateway/prompts/definitions/<name> ai/shared/<name>/prompts
   # a flow
   git mv duo_workflow_service/agent_platform/v1/flows/configs/<feature> ai/features/<domain>/<feature>/config
   ```

   Includes such as `<feature>/system/1.0.0.jinja` or `chat/react/system/1.0.0.jinja` keep
   working without a change.

   An include string that changes its prefix in the move is part of the Flow Registry public
   API, because custom flows in the AI Catalog include it by the old name. For example,
   `common/branch_naming/1.0.0.jinja` became `developer/common/branch_naming/1.0.0.jinja`.
   Add the old name to `LEGACY_INCLUDE_ALIASES` in `ai_gateway/prompts/feature_roots.py`,
   so the old name still renders. `make check-include-names INCLUDE_NAMES_BASE=origin/main`
   lists every include name that stopped resolving. A fragment that you delete on purpose
   goes to `RETIRED_INCLUDE_NAMES` in the same module instead; check first that no AI
   Catalog flow includes it.

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
   poetry run python -c "from ai.serving.grpc.flows import register_served_flows; from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig, list_flow_configs; register_served_flows(); print(sorted(e['flow_version'] for e in list_flow_configs(FlowConfig) if e['flow_identifier'] == '<feature>'))"
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
