"""Helpers for the Flow Creator benchmark suite.

This module is deliberately free of pytest imports so that every check in it can
be exercised directly against real flow configs, without an LLM and without a
test runner. The tests are thin wrappers around the functions here.

It provides three things:

1. A bounded multi-turn conversation runner, plus an on-disk cache of the YAML
    each test case produced, so that ``test_smoke`` and ``test_hard_rules``
    score the same generated YAML without paying for it twice.
2. YAML extraction and completeness checks (``yaml_code_blocks``,
    ``find_truncation_markers``).
3. One ``check_rule_N`` function per hard rule from
    https://gitlab.com/gitlab-org/gitlab/-/work_items/604709. Each returns a
    list of human-readable violations; an empty list means the rule passed.

The checks are unit tested in
``tests/agent_tests/flow_creator/test_helpers.py``, which runs on
every pipeline; this suite itself only runs as a manual job.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml
from langchain_core.messages import HumanMessage, ToolMessage

from agent_tests.flow_creator.cases import FlowCase
from agent_tests.helpers import AgentResult, ToolCall

# Rule 1 only applies to components that call a GitLab API tool, so the check is
# only as good as its notion of what a GitLab API tool is. Rather than curate a
# list that silently goes stale, derive it from the tool privileges the service
# itself ships: every tool reachable through the `read_only_gitlab` and
# `read_write_gitlab` privileges is a GitLab API tool. Imported privately
# because the registry exposes no public accessor for the privilege map.
from duo_workflow_service.components.tools_registry import _AGENT_PRIVILEGES

_GITLAB_PRIVILEGES = ("read_only_gitlab", "read_write_gitlab")

# Not every GitLab API tool is project-scoped - `get_current_user` and
# `gitlab_documentation_search` need no project context, and demanding
# `project_id` for them would be a false positive. A tool is treated as
# project-scoped when its argument schema accepts a project identifier.
_PROJECT_SCOPE_ARG_NAMES = frozenset(
    {
        "project_id",
        "project_path",
        "project_full_path",
        # Most DWS GitLab tools accept a resource URL instead of an ID, and the
        # search tools take the project or group as `id`.
        "url",
        "id",
    }
)

# A tool that targets a group or namespace and names no project field is not
# project-scoped, however its identifier argument is spelled. Without this, the
# epic tools match on their bare `id`/`url` and Rule 1 fails flows whose only
# GitLab tool never needs a project.
_GROUP_SCOPE_ARG_NAMES = frozenset({"group_id", "namespace", "namespace_id"})

_EXPLICIT_PROJECT_ARG_NAMES = frozenset(
    {"project_id", "project_path", "project_full_path"}
)

# Group- and instance-scoped tools whose schemas carry no group field to key off:
# `gitlab_group_project_search` searches within a group, and `gitlab__user_search`
# searches users across the instance.
_NEVER_PROJECT_SCOPED = frozenset(
    {"gitlab_group_project_search", "gitlab__user_search"}
)

# Generic API tools take a raw endpoint or GraphQL query, so their schemas name
# no project field even though Rule 1 calls them out explicitly.
_ALWAYS_PROJECT_SCOPED = frozenset({"gitlab_api_get", "gitlab_graphql"})

STANDARD_CONTEXT_CATEGORY = "agent_platform_standard_context"

BRANCH_CONTEXT_KEYS = ("primary_branch", "workload_branch")

# The sections the agent's own prompt tells it to "always include". Deliberately
# stricter than PartialFlowConfig, which makes `flow` and `routers` optional for
# chat-partial flows: the benchmark scores the agent against the standard it was
# instructed to follow.
REQUIRED_TOP_LEVEL_KEYS = (
    "version",
    "environment",
    "components",
    "routers",
    "flow",
    "prompts",
)

REQUIRED_APPROVAL_ROUTES = ("approve", "modify", "reject", "default_route")

REQUIRED_HUMAN_INPUT_UI_LOG_EVENTS = ("on_user_input_prompt", "on_user_response")

HUMAN_INPUT_TYPE = "HumanInputComponent"

AGENT_TYPE = "AgentComponent"

TERMINAL_COMPONENTS = frozenset({"end", "abort"})


def _model_field_default(model_cls: Any, field_name: str) -> Any:
    """Return the declared default of a Pydantic model field, without instantiating."""
    model_fields = getattr(model_cls, "model_fields", None)
    if not model_fields:
        return None
    return getattr(model_fields.get(field_name), "default", None)


def _project_scoped_gitlab_tool_names() -> frozenset[str]:
    """Return the names of GitLab API tools that require project context."""
    names: set[str] = set()

    for privilege in _GITLAB_PRIVILEGES:
        for tool_cls in _AGENT_PRIVILEGES.get(privilege, []):
            name = _model_field_default(tool_cls, "name")
            if not isinstance(name, str) or not name:
                continue

            if name in _NEVER_PROJECT_SCOPED:
                continue

            args_schema = _model_field_default(tool_cls, "args_schema")
            arg_names = set(getattr(args_schema, "model_fields", {}) or {})

            if not arg_names & _PROJECT_SCOPE_ARG_NAMES:
                continue

            # `group_id` with no project field means the tool addresses a group,
            # and its `id`/`url` argument identifies that group, not a project.
            if arg_names & _GROUP_SCOPE_ARG_NAMES and not (
                arg_names & _EXPLICIT_PROJECT_ARG_NAMES
            ):
                continue

            names.add(name)

    return frozenset(names | _ALWAYS_PROJECT_SCOPED)


GITLAB_API_TOOLS = _project_scoped_gitlab_tool_names()


# ===== YAML extraction =====

_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[\w+.-]*)[ \t]*\r?\n(?P<body>.*?)```",
    re.DOTALL,
)

_YAML_LANGS = frozenset({"yaml", "yml"})

_TOP_LEVEL_COMPONENTS_RE = re.compile(r"(?m)^components:[ \t]*(?:#.*)?$")

# Signals that the agent elided part of the flow instead of writing it out.
# Deliberately conservative: these must not fire on prose inside a generated
# prompt, which is why bare "etc." and "TODO" are not listed.
_TRUNCATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?m)^[ \t]*#[ \t]*\.\.\..*$", "an elided '# ...' comment line"),
    (r"(?m)^[ \t]{2,}\.\.\.[ \t]*$", "a bare '...' line"),
    (
        r"(?i)\.\.\.[ \t]*(?:rest|remaining|remainder|same as|as above|other"
        r"|others|more|and so on|etc)\b",
        "an ellipsis followed by 'rest of' / 'same as' / 'etc'",
    ),
    (
        r"(?i)(?:rest|remainder)[ \t]+of[ \t]+the[ \t]+"
        r"(?:flow|yaml|config|configuration|components|prompts)",
        "a 'rest of the flow' placeholder",
    ),
    (
        r"(?i)<[ \t]*(?:snip|elided|truncated|omitted|unchanged)",
        "a '<snip>'-style placeholder",
    ),
    (
        r"(?i)\([^)]{0,40}(?:omitted|unchanged|as before|truncated)[^)]{0,40}\)",
        "an '(omitted)' / '(unchanged)' placeholder",
    ),
)


def looks_like_flow_config(body: str) -> bool:
    """Return True when a code block body looks like a flow config document."""
    return bool(_TOP_LEVEL_COMPONENTS_RE.search(body))


def yaml_code_blocks(text: str) -> list[str]:
    """Return the YAML code blocks in an agent response.

    Blocks tagged ``yaml`` or ``yml`` always count. Untagged blocks count only
    when they look like a flow config, so that shell or JSON examples in the same
    response are not mistaken for a second YAML document.
    """
    blocks: list[str] = []
    for match in _FENCE_RE.finditer(text):
        lang = (match.group("lang") or "").lower()
        body = match.group("body")
        if lang in _YAML_LANGS or (not lang and looks_like_flow_config(body)):
            blocks.append(body)
    return blocks


def flow_config_blocks(text: str) -> list[str]:
    """Return the YAML code blocks that look like a whole flow config."""
    return [block for block in yaml_code_blocks(text) if looks_like_flow_config(block)]


def has_unterminated_code_fence(text: str) -> bool:
    """Return True when a code fence was opened and never closed.

    This is the signature of a response that hit the model's output token limit part-way through the YAML, which is
    worth distinguishing from an agent that simply chose not to emit YAML.
    """
    return text.count("```") % 2 == 1


def find_truncation_markers(text: str) -> list[str]:
    """Return descriptions of any elision or truncation markers found."""
    found: list[str] = []
    for pattern, description in _TRUNCATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append(f"{description}: {match.group(0).strip()!r}")
    return found


def parse_flow_yaml(yaml_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a flow config document, returning ``(config, error)``."""
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return None, f"YAML did not parse: {exc}"

    if not isinstance(parsed, dict):
        return None, f"YAML parsed to {type(parsed).__name__}, expected a mapping"

    return parsed, None


def missing_top_level_keys(config: dict[str, Any]) -> list[str]:
    """Return the required flow config sections that are absent."""
    return [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]


# ===== Config accessors =====


def components(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the component mappings declared in a flow config."""
    raw = config.get("components") or []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def component_name(component: dict[str, Any]) -> str:
    """Return a component's name, or a placeholder when it has none."""
    return str(component.get("name") or "<unnamed>")


def components_of_type(
    config: dict[str, Any],
    component_type: str,
) -> list[dict[str, Any]]:
    """Return the components declared with the given ``type:``."""
    return [
        component
        for component in components(config)
        if component.get("type") == component_type
    ]


def agent_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the AgentComponents declared in a flow config."""
    return components_of_type(config, AGENT_TYPE)


def supervisor_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the AgentComponents that supervise sub-agents.

    A supervisor is not its own component type: an ``AgentComponent`` becomes one by declaring a non-empty
    ``subagents:`` list, which is what gives it ``delegate_task``. See "Supervisor Mode" in
    ``docs/flow_registry/v1.md``.
    """
    supervisors = []
    for component in agent_components(config):
        subagents = component.get("subagents")
        if isinstance(subagents, list) and subagents:
            supervisors.append(component)
    return supervisors


def component_toolset(component: dict[str, Any]) -> list[str]:
    """Return the tool names in a component's toolset.

    Tolerates both documented spellings (``- tool_name`` and
    ``- tool_name: {options}``) and the case where a toolset was written as a
    comma-less YAML flow sequence, which YAML collapses into a single
    space-separated scalar.
    """
    raw = component.get("toolset")
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.split()

    names: list[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, str):
            names.extend(entry.split())
        elif isinstance(entry, dict):
            names.extend(str(key) for key in entry)
    return names


def gitlab_api_tools_in(component: dict[str, Any]) -> list[str]:
    """Return the project-scoped GitLab API tools a component can call."""
    return sorted(set(component_toolset(component)) & GITLAB_API_TOOLS)


def component_inputs(component: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a component's input mappings."""
    raw = component.get("inputs") or []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def input_sources(component: dict[str, Any]) -> list[str]:
    """Return the ``from:`` values of a component's inputs."""
    return [str(entry.get("from")) for entry in component_inputs(component)]


def template_variable_name(from_key: str, alias: str | None) -> str:
    """Return the template variable an input is exposed as.

    Mirrors ``IOKey.template_variable_name``: the ``as:`` alias when set,
    otherwise the last dotted subkey of the source.
    """
    if alias:
        return alias
    tail = from_key.split(":", 1)[-1]
    return tail.split(".")[-1] if tail else from_key


def input_variables(component: dict[str, Any]) -> dict[str, str]:
    """Return ``{template_variable: source}`` for a component's inputs."""
    variables: dict[str, str] = {}
    for entry in component_inputs(component):
        from_key = str(entry.get("from") or "")
        alias = entry.get("as")
        alias_str = str(alias) if isinstance(alias, (str, int)) else None
        variables[template_variable_name(from_key, alias_str)] = from_key
    return variables


def inline_prompts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the prompts defined inline in the flow config."""
    raw = config.get("prompts") or []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def prompt_label(prompt: dict[str, Any]) -> str:
    """Return a readable identifier for a prompt."""
    return str(prompt.get("prompt_id") or prompt.get("name") or "<unnamed prompt>")


def inline_prompt_for(
    config: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the inline prompt a component uses, or None.

    A component with ``prompt_version`` set resolves its prompt from the prompt
    registry instead of the flow config, so there is no inline prompt to check.
    """
    prompt_id = component.get("prompt_id")
    if not prompt_id or component.get("prompt_version"):
        return None

    for prompt in inline_prompts(config):
        if prompt.get("prompt_id") == prompt_id:
            return prompt
    return None


def prompt_template(prompt: dict[str, Any]) -> dict[str, Any]:
    """Return a prompt's ``prompt_template`` block."""
    raw = prompt.get("prompt_template")
    return raw if isinstance(raw, dict) else {}


def template_block_text(prompt: dict[str, Any], role: str) -> str:
    """Return the text of one role block of a prompt template.

    A role may map to a list of strings (used for prompt caching), in which case the parts are joined.
    """
    content = prompt_template(prompt).get(role)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part for part in content if isinstance(part, str))
    return ""


def template_text(prompt: dict[str, Any]) -> str:
    """Return all message text of a prompt template, ignoring placeholders."""
    return "\n".join(
        template_block_text(prompt, role)
        for role in prompt_template(prompt)
        if role != "placeholder"
    )


def rendered_template_for(
    config: dict[str, Any],
    component: dict[str, Any],
) -> tuple[str, str] | None:
    """Return ``(label, template_text)`` for the template a component renders.

    Agents and one-off components render an inline prompt; a
    ``HumanInputComponent`` renders its ``message_template``. Both are rendered
    with the same Jinja2 environment, so both are subject to Rule 7.
    """
    prompt = inline_prompt_for(config, component)
    if prompt is not None:
        return prompt_label(prompt), template_text(prompt)

    if component.get("type") == HUMAN_INPUT_TYPE:
        message_template = component.get("message_template")
        if isinstance(message_template, str):
            return (
                f"message_template of '{component_name(component)}'",
                message_template,
            )

    return None


# ===== Placeholder syntax =====

_JINJA_EXPRESSION_RE = re.compile(r"{{(?P<body>.*?)}}", re.DOTALL)
_JINJA_STATEMENT_RE = re.compile(r"{%-?(?P<body>.*?)-?%}", re.DOTALL)
_ANGLE_PLACEHOLDER_RE = re.compile(r"<<[ \t]*(?P<name>\w+)[ \t]*>>")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_TEMPLATE_INHERITANCE_RE = re.compile(r"{%-?[ \t]*(?:extends|include)\b")

# Only used to filter Jinja statements, where control keywords appear alongside
# variables. Expressions are not filtered, so an input alias that happens to
# collide with a keyword or filter name is never reported as missing.
_JINJA_STATEMENT_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "block",
        "call",
        "elif",
        "else",
        "endblock",
        "endcall",
        "endfilter",
        "endfor",
        "endif",
        "endmacro",
        "endraw",
        "endset",
        "endwith",
        "extends",
        "false",
        "filter",
        "for",
        "from",
        "if",
        "import",
        "in",
        "include",
        "is",
        "macro",
        "none",
        "not",
        "or",
        "raw",
        "set",
        "true",
        "with",
        "without",
    }
)


def jinja_placeholders(text: str) -> set[str]:
    """Return the variable names a Jinja2 template references.

    Both ``{{ var }}`` expressions and ``{% if var %}`` statements count, so a
    variable used only as a condition is not reported as missing.
    """
    names: set[str] = set()

    for match in _JINJA_EXPRESSION_RE.finditer(text):
        # Only the root of a dotted or filtered expression is the variable name.
        root = match.group("body").split("|")[0]
        identifier = _IDENTIFIER_RE.search(root)
        if identifier:
            names.add(identifier.group(0))

    for match in _JINJA_STATEMENT_RE.finditer(text):
        # String literals hold template paths, not variable names.
        body = _STRING_LITERAL_RE.sub(" ", match.group("body"))
        for identifier in _IDENTIFIER_RE.findall(body):
            if identifier.lower() not in _JINJA_STATEMENT_KEYWORDS:
                names.add(identifier)

    return names


def uses_template_inheritance(text: str) -> bool:
    """Return True when a template pulls in another template.

    A template that uses ``{% extends %}`` or ``{% include %}`` may declare its
    placeholders in the parent, which is not visible from the flow config alone,
    so placeholder coverage cannot be verified from the config text.
    """
    return bool(_TEMPLATE_INHERITANCE_RE.search(text))


def angle_placeholders(text: str) -> set[str]:
    """Return names written in the inert ``<<name>>`` form.

    The agent's own system prompt documents placeholders as ``<<name>>`` so that
    its template is not rendered when the prompt itself is loaded. That form is
    literal text at runtime, so generated YAML that uses it silently substitutes
    nothing.
    """
    return {match.group("name") for match in _ANGLE_PLACEHOLDER_RE.finditer(text)}


# ===== Routing =====


def routers(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the router mappings declared in a flow config."""
    raw = config.get("routers") or []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def routers_from(config: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return the routers whose ``from:`` is the given component."""
    return [router for router in routers(config) if router.get("from") == name]


def router_routes(router: dict[str, Any]) -> dict[str, str]:
    """Return the ``{state_value: target}`` map of a conditional router."""
    condition = router.get("condition")
    if not isinstance(condition, dict):
        return {}
    routes = condition.get("routes")
    if not isinstance(routes, dict):
        return {}
    return {str(key): str(value) for key, value in routes.items()}


def router_targets(router: dict[str, Any]) -> list[str]:
    """Return every component a router can send the flow to."""
    if "condition" in router:
        return list(router_routes(router).values())
    target = router.get("to")
    return [str(target)] if target else []


def entry_point(config: dict[str, Any]) -> str | None:
    """Return the flow's entry point component name, if declared."""
    flow = config.get("flow")
    if not isinstance(flow, dict):
        return None
    value = flow.get("entry_point")
    return str(value) if value else None


def components_running_before(config: dict[str, Any], name: str) -> set[str]:
    """Return the components that can run before ``name`` in the graph.

    Walks forward from the entry point without expanding past ``name``, so a
    component reachable only through ``name`` - for example the target of a
    ``modify`` loop-back - is correctly excluded.
    """
    start = entry_point(config)
    if not start:
        return set()

    seen: set[str] = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        if node == name:
            # The gate is a boundary: anything reachable only through it has not
            # run at the point the gate fires.
            continue
        for router in routers_from(config, node):
            queue.extend(router_targets(router))

    return seen - {name} - TERMINAL_COMPONENTS


def walk_strings(value: Any) -> Iterable[str]:
    """Yield every string *value* in a nested config structure.

    Mapping keys are not yielded, so declaring ``primary_branch`` in a
    ``flow.inputs`` schema does not count as referencing it.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


# ===== Prompt resolution =====

# The component types that drive an LLM and so must resolve a prompt. Both
# declare `prompt_id: str` and `prompt_version: Optional[str]`; the other
# component types run code and need no prompt at all.
PROMPT_BEARING_TYPES = frozenset({"AgentComponent", "OneOffComponent"})


def components_missing_inline_prompts(config: dict[str, Any]) -> list[str]:
    """Prompt-bearing components whose prompt is not defined in this flow.

    A custom flow is sent to the service inline, and its author cannot add files to the AI Gateway prompt registry at
    ``ai_gateway/prompts/definitions/`` — that takes a merge request against the AI Gateway itself. Every agent in a
    generated flow must therefore carry its own prompt in the flow's ``prompts:`` block.

    This matters because the rules that read prompts (4 to 7) would otherwise skip a component whose prompt does not
    resolve, reporting nothing at all for output that cannot run. ``missing_top_level_keys`` already catches a flow
    with no ``prompts:`` key; what it cannot see is a per-component mismatch inside a flow that has one.
    """
    violations: list[str] = []

    for component in components(config):
        component_type = component.get("type")
        if component_type not in PROMPT_BEARING_TYPES:
            continue

        name = component_name(component)
        prompt_id = component.get("prompt_id")

        if not prompt_id:
            violations.append(
                f"component '{name}' is an {component_type} but declares no 'prompt_id', "
                f"so it has no prompt to run"
            )
            continue

        prompt_version = component.get("prompt_version")
        if prompt_version:
            violations.append(
                f"component '{name}' sets prompt_version '{prompt_version}', which resolves "
                f"'{prompt_id}' from the AI Gateway prompt registry; a custom flow cannot add "
                f"prompts there, so it must define the prompt inline instead"
            )
            continue

        if inline_prompt_for(config, component) is None:
            declared = [prompt_label(prompt) for prompt in inline_prompts(config)]
            violations.append(
                f"component '{name}' references prompt_id '{prompt_id}', but the flow declares "
                f"no matching inline prompt (inline prompts: {declared})"
            )

    return violations


# ===== Hard rule checks =====
# Each check returns a list of violations; an empty list means the rule passed.
# The rules are defined in
# https://gitlab.com/gitlab-org/gitlab/-/work_items/604709.


def check_rule_1_project_id_threaded(config: dict[str, Any]) -> list[str]:
    """Rule 1: components with GitLab API tools must receive ``project_id``."""
    violations: list[str] = []

    for component in components(config):
        api_tools = gitlab_api_tools_in(component)
        if not api_tools:
            continue

        name = component_name(component)
        sources = input_sources(component)
        if "context:project_id" not in sources:
            violations.append(
                f"component '{name}' calls GitLab API tools {api_tools} but has "
                f"no 'context:project_id' input (inputs: {sources or 'none'})"
            )

        prompt = inline_prompt_for(config, component)
        if prompt is None:
            if component.get("prompt_id") and not component.get("prompt_version"):
                violations.append(
                    f"component '{name}' references prompt_id "
                    f"'{component['prompt_id']}', which is not defined in "
                    f"'prompts', so its project ID reference cannot be verified"
                )
            continue

        user_block = template_block_text(prompt, "user")
        if "project_id" not in user_block and "Project ID" not in user_block:
            violations.append(
                f"prompt '{prompt_label(prompt)}' (component '{name}') does not "
                f"reference the project ID in its 'user' block"
            )

    return violations


def branch_context_inputs(config: dict[str, Any]) -> list[str]:
    """Return the branch context keys the flow actually reads.

    Only component ``inputs`` count. Mentioning ``primary_branch`` in prose
    inside a prompt is not an access of the standard context and does not
    require a declaration, so scanning all strings would produce false
    positives.
    """
    sources = [
        source
        for component in components(config)
        for source in input_sources(component)
    ]
    return [
        key
        for key in BRANCH_CONTEXT_KEYS
        if any(source.endswith(f".{key}") for source in sources)
    ]


def check_rule_2_flow_inputs_declared(config: dict[str, Any]) -> list[str]:
    """Rule 2: branch context requires a ``flow.inputs`` declaration."""
    referenced = branch_context_inputs(config)
    if not referenced:
        return []

    flow = config.get("flow")
    flow_inputs = flow.get("inputs") if isinstance(flow, dict) else None
    if not isinstance(flow_inputs, list) or not flow_inputs:
        return [f"flow references {referenced} but declares no 'flow.inputs' stanza"]

    declared: set[str] = set()
    for entry in flow_inputs:
        if not isinstance(entry, dict):
            continue
        if entry.get("category") != STANDARD_CONTEXT_CATEGORY:
            continue
        schema = entry.get("input_schema")
        if isinstance(schema, dict):
            declared.update(str(key) for key in schema)

    if not declared:
        return [
            f"flow references {referenced} but 'flow.inputs' declares no "
            f"'{STANDARD_CONTEXT_CATEGORY}' category with an input_schema"
        ]

    missing = [key for key in referenced if key not in declared]
    if missing:
        return [
            f"flow references {missing} but the '{STANDARD_CONTEXT_CATEGORY}' "
            f"input_schema only declares {sorted(declared)}"
        ]

    return []


def check_rule_3_human_input_wiring(config: dict[str, Any]) -> list[str]:
    """Rule 3: every HumanInputComponent must be fully wired."""
    violations: list[str] = []

    for component in components_of_type(config, HUMAN_INPUT_TYPE):
        name = component_name(component)
        violations.extend(_check_gate_declaration(component, name))
        violations.extend(_check_gate_router(config, name))
        violations.extend(_check_gate_response_target(config, component, name))

    return violations


def _check_gate_declaration(component: dict[str, Any], name: str) -> list[str]:
    """Rule 3.1 and 3.2: interaction_type and ui_log_events on the gate itself."""
    violations: list[str] = []

    if not component.get("interaction_type"):
        violations.append(f"gate '{name}' does not set 'interaction_type'")

    raw_events = component.get("ui_log_events")
    events = (
        {str(event) for event in raw_events} if isinstance(raw_events, list) else set()
    )
    missing = [
        event for event in REQUIRED_HUMAN_INPUT_UI_LOG_EVENTS if event not in events
    ]
    if missing:
        violations.append(
            f"gate '{name}' is missing ui_log_events {missing} "
            f"(has: {sorted(events) or 'none'})"
        )

    return violations


def _check_gate_router(config: dict[str, Any], name: str) -> list[str]:
    """Rule 3.3 and 3.4: a conditional router covering all approval routes."""
    violations: list[str] = []

    outgoing = routers_from(config, name)
    if not outgoing:
        violations.append(f"gate '{name}' has no outgoing router")

    for router in outgoing:
        if "condition" not in router:
            violations.append(
                f"gate '{name}' is followed by a simple 'to: {router.get('to')}' "
                f"router; a conditional router is required"
            )
            continue

        routes = router_routes(router)
        missing = [route for route in REQUIRED_APPROVAL_ROUTES if route not in routes]
        if missing:
            violations.append(
                f"router after gate '{name}' is missing routes {missing} "
                f"(has: {sorted(routes)})"
            )
        if routes.get("modify") in TERMINAL_COMPONENTS:
            violations.append(
                f"router after gate '{name}' sends 'modify' to "
                f"'{routes['modify']}'; it must route back to an agent"
            )

    return violations


def _check_gate_response_target(
    config: dict[str, Any],
    component: dict[str, Any],
    name: str,
) -> list[str]:
    """Rule 3.5: sends_response_to must point at a component that already ran."""
    target = component.get("sends_response_to")
    if not target:
        return [f"gate '{name}' does not set 'sends_response_to'"]

    already_run = components_running_before(config, name)
    if not already_run:
        if entry_point(config) is None:
            return [
                f"gate '{name}' sets sends_response_to='{target}' but the flow "
                f"declares no 'flow.entry_point', so nothing can be verified to "
                f"run before the gate"
            ]
        return [
            f"gate '{name}' sets sends_response_to='{target}' but no component "
            f"runs before the gate (it is the entry point, or is unreachable "
            f"from it); this raises KeyError('{target}') at runtime"
        ]

    if str(target) not in already_run:
        return [
            f"gate '{name}' sets sends_response_to='{target}', which does not run "
            f"before the gate (components that run first: {sorted(already_run)}); "
            f"this raises KeyError('{target}') at runtime"
        ]

    return []


def check_rule_5_unit_primitives(config: dict[str, Any]) -> list[str]:
    """Rule 5: every inline prompt must declare ``unit_primitives``."""
    violations: list[str] = []

    for prompt in inline_prompts(config):
        if "unit_primitives" not in prompt:
            violations.append(
                f"inline prompt '{prompt_label(prompt)}' has no 'unit_primitives' key"
            )
        elif not isinstance(prompt["unit_primitives"], list):
            violations.append(
                f"inline prompt '{prompt_label(prompt)}' has 'unit_primitives' of "
                f"type {type(prompt['unit_primitives']).__name__}, expected a list"
            )

    return violations


def check_rule_6_history_placeholder(config: dict[str, Any]) -> list[str]:
    """Rule 6: every inline prompt template must declare ``placeholder: history``."""
    violations: list[str] = []

    for prompt in inline_prompts(config):
        template = prompt_template(prompt)
        if not template:
            violations.append(
                f"inline prompt '{prompt_label(prompt)}' has no 'prompt_template' block"
            )
            continue

        placeholder = template.get("placeholder")
        if isinstance(placeholder, str):
            declared = {placeholder}
        elif isinstance(placeholder, list):
            declared = {str(item) for item in placeholder}
        else:
            declared = set()

        if "history" not in declared:
            violations.append(
                f"inline prompt '{prompt_label(prompt)}' does not declare "
                f"'placeholder: history' (found: {placeholder!r})"
            )

    return violations


def check_rule_7_aliases_match_placeholders(config: dict[str, Any]) -> list[str]:
    """Rule 7: every input ``as:`` alias must appear as a ``{{alias}}`` placeholder.

    Jinja2 is the only template syntax the framework renders, so the inert
    ``<<alias>>`` form is reported as a violation with an explicit hint: it
    substitutes nothing at runtime and fails silently.

    Templates that use ``{% extends %}`` or ``{% include %}`` are skipped, since
    their placeholders may live in a template the flow config does not contain.
    """
    violations: list[str] = []

    for component in components(config):
        rendered_template = rendered_template_for(config, component)
        if rendered_template is None:
            continue

        label, text = rendered_template
        if uses_template_inheritance(text):
            continue

        name = component_name(component)
        rendered = jinja_placeholders(text)
        inert = angle_placeholders(text)

        for variable, source in input_variables(component).items():
            if variable in rendered:
                continue
            if variable in inert:
                violations.append(
                    f"component '{name}' input '{source}' is exposed as "
                    f"'{variable}' and {label} writes '<<{variable}>>', which the "
                    f"framework does not render; it must be '{{{{{variable}}}}}'"
                )
            else:
                violations.append(
                    f"component '{name}' input '{source}' is exposed as "
                    f"'{variable}' but {label} has no '{{{{{variable}}}}}' "
                    f"placeholder (placeholders found: {sorted(rendered) or 'none'})"
                )

    return violations


def system_prompts_to_validate(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(label, system_prompt)`` for every inline prompt.

    Used by the Rule 4 check, the only rule that cannot be verified deterministically and so is delegated to the LLM
    judge.
    """
    prompts: list[tuple[str, str]] = []
    for prompt in inline_prompts(config):
        system = template_block_text(prompt, "system")
        if system.strip():
            prompts.append((prompt_label(prompt), system))
    return prompts


# ===== Conversation runner and response cache =====

CACHE_ROOT = (
    Path(__file__).resolve().parents[2]
    / ".test-reports"
    / "agent_tests"
    / "flow_creator"
)

MAX_AGENT_TURNS_PER_USER_TURN = 12

# Tool results are truncated in the transcript artifact only. The agent always
# receives the untruncated result.
MAX_TRANSCRIPT_TOOL_RESULT_CHARS = 2_000


@dataclass
class GeneratedFlow:
    """The flow YAML one test case produced, plus how it was produced."""

    case_id: str
    response: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    user_turns: int = 1
    from_cache: bool = False
    # False when the agent never stopped calling tools, which is the failure mode
    # a missing stopping instruction (Rule 4) produces.
    agent_finished: bool = True

    @property
    def yaml_blocks(self) -> list[str]:
        """Every YAML code block in the response."""
        return yaml_code_blocks(self.response)

    @property
    def flow_blocks(self) -> list[str]:
        """The YAML code blocks that look like a whole flow config."""
        return flow_config_blocks(self.response)

    @property
    def yaml_text(self) -> str | None:
        """The single flow config document, or None if there is not exactly one."""
        blocks = self.flow_blocks
        return blocks[0] if len(blocks) == 1 else None

    @property
    def config(self) -> dict[str, Any] | None:
        """The parsed flow config, or None when absent or unparsable."""
        yaml_text = self.yaml_text
        if yaml_text is None:
            return None
        config, _ = parse_flow_yaml(yaml_text)
        return config

    @property
    def parse_error(self) -> str | None:
        """Why no config is available, or None when one is."""
        blocks = self.flow_blocks
        if not blocks:
            if not self.agent_finished:
                reason = (
                    "the agent never stopped calling tools, so it produced no "
                    "final answer"
                )
            elif has_unterminated_code_fence(self.response):
                reason = (
                    "the response ends inside an unclosed code fence, so the YAML "
                    "was cut off by the model's output token limit"
                )
            else:
                reason = "the response contains no flow config YAML block"
            return f"after {self.user_turns} user turn(s), {reason}"
        if len(blocks) > 1:
            return f"agent produced {len(blocks)} flow config YAML blocks, expected 1"
        _, error = parse_flow_yaml(blocks[0])
        return error


def _cache_dir(execution_model: str) -> Path:
    return CACHE_ROOT / (execution_model or "unknown-model")


def _cache_path(case_id: str, execution_model: str) -> Path:
    return _cache_dir(execution_model) / f"{case_id}.json"


def load_cached_flow(case_id: str, execution_model: str) -> GeneratedFlow | None:
    """Return a previously generated flow for this case and model, if any."""
    try:
        payload = json.loads(_cache_path(case_id, execution_model).read_text())
    except (OSError, ValueError):
        return None

    response = payload.get("response")
    if not isinstance(response, str):
        return None

    return GeneratedFlow(
        case_id=case_id,
        response=response,
        transcript=payload.get("transcript") or [],
        user_turns=int(payload.get("user_turns") or 1),
        from_cache=True,
        agent_finished=bool(payload.get("agent_finished", True)),
    )


def save_cached_flow(flow: GeneratedFlow, execution_model: str) -> None:
    """Persist a generated flow, its transcript, and its YAML for inspection.

    The transcript is always written because it is the primary debugging artifact. The reusable cache entry is only
    written when the agent actually produced a flow config, so that a degenerate run is retried rather than replayed.
    """
    directory = _cache_dir(execution_model)
    try:
        directory.mkdir(parents=True, exist_ok=True)

        transcript = "\n\n".join(
            f"## {turn['role']}\n\n{turn['content']}" for turn in flow.transcript
        )
        (directory / f"{flow.case_id}.transcript.md").write_text(
            f"# {flow.case_id}\n\n{transcript}\n"
        )

        if not flow.flow_blocks:
            return

        _cache_path(flow.case_id, execution_model).write_text(
            json.dumps(
                {
                    "case_id": flow.case_id,
                    "execution_model": execution_model,
                    "user_turns": flow.user_turns,
                    "agent_finished": flow.agent_finished,
                    "response": flow.response,
                    "transcript": flow.transcript,
                },
                indent=2,
            )
            + "\n"
        )

        yaml_text = flow.yaml_text
        if yaml_text is not None:
            (directory / f"{flow.case_id}.yml").write_text(yaml_text)
    except OSError:
        # The cache is an optimisation and a debugging aid; failing to write it
        # must never fail the benchmark.
        pass


def message_text(message: Any) -> str:
    """Return the text content of a message, flattening content blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


@dataclass
class ConversationResult:
    """The outcome of a multi-turn conversation with an agent."""

    result: AgentResult
    transcript: list[dict[str, str]]
    user_turns: int
    # False when the agent was still calling tools after
    # MAX_AGENT_TURNS_PER_USER_TURN, that is, it never produced a final answer.
    completed: bool = True
    # Every piece of prose the agent emitted, in order, including text sent
    # alongside tool calls. `result.content` only holds the last message of the
    # last turn, which drops YAML the agent emitted while still calling tools.
    agent_messages: list[str] = field(default_factory=list)


async def _invoke_tool(
    tools_by_name: dict[str, Any],
    tool_call: dict[str, Any],
) -> ToolMessage:
    """Run one tool call, reporting failures back to the agent as tool output."""
    tool_call_id = tool_call.get("id") or ""
    tool = tools_by_name.get(tool_call["name"])
    if tool is None:
        return ToolMessage(content="Tool not found", tool_call_id=tool_call_id)

    try:
        output = await tool.ainvoke(tool_call.get("args", {}))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Mirrors ask_agent: a tool error is context for the agent, not a test
        # failure, so the conversation continues.
        return ToolMessage(content=f"Error: {exc}", tool_call_id=tool_call_id)

    return ToolMessage(content=str(output), tool_call_id=tool_call_id)


def _format_tool_exchange(
    tool_call: dict[str, Any],
    tool_message: ToolMessage,
) -> str:
    """Render one tool call and its result for the transcript.

    Tool results are truncated because the mocked doc tools return whole documentation pages, which would otherwise
    dwarf the agent's own reasoning in the artifact.
    """
    args = tool_call.get("args", {})
    result = str(tool_message.content)
    if len(result) > MAX_TRANSCRIPT_TOOL_RESULT_CHARS:
        omitted = len(result) - MAX_TRANSCRIPT_TOOL_RESULT_CHARS
        result = (
            f"{result[:MAX_TRANSCRIPT_TOOL_RESULT_CHARS]}\n"
            f"... [{omitted} more characters omitted]"
        )

    return f"Arguments:\n\n```json\n{json.dumps(args, indent=2, default=str)}\n```\n\nResult:\n\n```\n{result}\n```"


async def run_conversation(
    agent: Any,
    initial_state_factory: Callable[..., dict],
    user_messages: Sequence[str],
    stop_when: Callable[[str], bool] | None = None,
    validation_model: str = "",
) -> ConversationResult:
    """Hold a bounded multi-turn conversation with an agent.

    Each user message is sent in turn, and after each one the agent's tool calls
    are executed until it produces a text response - exactly as
    ``agent_tests.helpers.ask_agent`` does for a single turn. The conversation
    stops early when ``stop_when`` accepts the agent's latest response, which is
    how the suite avoids spending follow-up turns once the agent has answered.

    Args:
        agent: The ChatAgent instance.
        initial_state_factory: The ``initial_state`` fixture factory.
        user_messages: The user turns to send, in order.
        stop_when: Predicate on the agent's response text; stop when it is True.
        validation_model: Model used by the LLM judge for later assertions.

    Returns:
        ConversationResult with the final AgentResult, the transcript, and the
        number of user turns actually spent.
    """
    if not user_messages:
        raise ValueError("run_conversation requires at least one user message")

    agent_name = agent.name
    state = initial_state_factory(user_messages[0], agent_name=agent_name)
    history = state["conversation_history"][agent_name]
    tools_by_name = {tool.name: tool for tool in agent.prompt_adapter._tools}

    all_tool_calls: list[ToolCall] = []
    transcript: list[dict[str, str]] = []
    agent_messages: list[str] = []
    result: dict[str, Any] | None = None
    ai_message: Any = None
    user_turns = 0
    completed = True

    for index, user_message in enumerate(user_messages):
        if index > 0:
            history.append(HumanMessage(content=user_message))
            state = {**state, "conversation_history": {agent_name: history}}
        transcript.append({"role": "user", "content": user_message})
        user_turns += 1

        answered = False
        for _ in range(MAX_AGENT_TURNS_PER_USER_TURN):
            result = await agent.run(state)
            ai_message = result["conversation_history"][agent_name][-1]

            if not ai_message.tool_calls:
                answered = True
                break

            # Record any prose the agent emitted alongside its tool calls. The
            # agent sometimes answers and calls a verification tool in the same
            # message, and that text is otherwise lost: only the final message
            # of the turn reaches `response_text` below.
            interim_text = message_text(ai_message)
            if interim_text.strip():
                agent_messages.append(interim_text)
                transcript.append({"role": "agent (interim)", "content": interim_text})

            tool_messages: list[ToolMessage] = []
            for tool_call in ai_message.tool_calls:
                all_tool_calls.append(
                    ToolCall(
                        name=tool_call["name"],
                        args=tool_call.get("args", {}),
                        id=tool_call.get("id") or "",
                    )
                )
                tool_message = await _invoke_tool(tools_by_name, tool_call)
                tool_messages.append(tool_message)
                transcript.append(
                    {
                        "role": f"tool call: {tool_call['name']}",
                        "content": _format_tool_exchange(tool_call, tool_message),
                    }
                )

            history.append(ai_message)
            history.extend(tool_messages)
            state = {**state, "conversation_history": {agent_name: history}}

        if not answered:
            # The agent is still calling tools. Its last message is already in
            # the history with unanswered tool calls, so the conversation cannot
            # continue; stop and report it.
            completed = False
            transcript.append(
                {
                    "role": "note",
                    "content": (
                        f"Agent was still calling tools after "
                        f"{MAX_AGENT_TURNS_PER_USER_TURN} turns and never "
                        f"produced a final answer."
                    ),
                }
            )
            break

        history.append(ai_message)
        state = {**state, "conversation_history": {agent_name: history}}
        response_text = message_text(ai_message)
        agent_messages.append(response_text)
        transcript.append({"role": "agent", "content": response_text})

        # Checked against every message of this turn, not just the last one: the
        # agent may emit the flow YAML in a message that also carries a tool
        # call, and spending follow-up turns after that is wasted budget.
        if stop_when is not None and any(stop_when(text) for text in agent_messages):
            break

    if ai_message is None or result is None:
        raise RuntimeError("Agent did not produce any response")

    agent_result = AgentResult(
        ai_message=ai_message,
        state=result,
        validation_model=validation_model,
    )
    agent_result.tool_calls = all_tool_calls

    return ConversationResult(
        result=agent_result,
        transcript=transcript,
        user_turns=user_turns,
        completed=completed,
        agent_messages=agent_messages,
    )


async def generate_flow(
    agent: Any,
    initial_state_factory: Callable[..., dict],
    case: FlowCase,
    follow_ups: Sequence[str],
    execution_model: str,
    validation_model: str = "",
    use_cache: bool = True,
) -> GeneratedFlow:
    """Ask the agent to design one case's flow and return the YAML it produced.

    The case prompt is sent first. Because the agent is instructed to ask how the
    flow will be triggered before designing it, the follow-ups are replayed until
    it emits a flow config - at most ``len(follow_ups)`` extra turns.

    Results are cached on disk per case and execution model so that the smoke
    tests and the hard-rule tests score the same YAML without generating it
    twice.
    """
    if use_cache:
        cached = load_cached_flow(case.case_id, execution_model)
        if cached is not None:
            return cached

    conversation = await run_conversation(
        agent,
        initial_state_factory,
        [case.prompt, *follow_ups],
        stop_when=lambda text: bool(flow_config_blocks(text)),
        validation_model=validation_model,
    )

    flow = GeneratedFlow(
        case_id=case.case_id,
        response=_response_bearing_flow(conversation),
        transcript=conversation.transcript,
        user_turns=conversation.user_turns,
        agent_finished=conversation.completed,
    )
    save_cached_flow(flow, execution_model)
    return flow


def _response_bearing_flow(conversation: ConversationResult) -> str:
    """Return the agent message the flow YAML should be scored against.

    Normally that is the final response. When the agent emits its flow in a message that also carries a tool call, the
    final response is a follow-up remark instead, and scoring it would report "no flow config YAML block" for a run that
    did produce one. In that case the most recent message that does contain a flow config is used.
    """
    final = conversation.result.content
    if flow_config_blocks(final):
        return final

    for text in reversed(conversation.agent_messages):
        if flow_config_blocks(text):
            return text

    return final
