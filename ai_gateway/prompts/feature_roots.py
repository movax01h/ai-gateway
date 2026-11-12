"""Registry of prompt roots under ``ai/features/`` and ``ai/shared/``.

A feature owns its prompts in ``ai/features/<domain>/<feature>/prompts/``. A
feature with several file-based prompts lists them in
``prompts/subprompts_index.yml``. A definition that several features use lives
in ``ai/shared/<name>/prompts/``.
Both discovery paths consult this registry by prompt id:

- version-file discovery in ``LocalPromptRegistry._resolve_id``
- Jinja ``{% include %}`` resolution via ``FeatureRootLoader`` in ``base.py``

Registration strips the prompt-id prefix, so a self-namespaced include such as
``glab_ask_git_command/system/1.0.0.jinja`` keeps working unchanged.
"""

from pathlib import Path

from lib.feature_roots import default_features_dir
from lib.subprompts_index import read_subprompts_index

__all__ = [
    "LEGACY_INCLUDE_ALIASES",
    "RETIRED_INCLUDE_NAMES",
    "discover_feature_prompts",
    "feature_prompt_root",
    "nested_prompt_roots",
    "register_prompt_root",
]

# Include names that a move retired, mapped to the current name of the file.
# Custom AI Catalog flows include by the old name, so the Flow Registry must
# keep serving it (ai-assist#3494).
LEGACY_INCLUDE_ALIASES: dict[str, str] = {
    "common/api_assistant/system/1.0.0.jinja": "slack_assistant/common/api_assistant/system/1.0.0.jinja",
    "common/branch_naming/1.0.0.jinja": "developer/common/branch_naming/1.0.0.jinja",
    "common/developer/system/1.0.0.jinja": "developer/common/system/1.0.0.jinja",
    "common/git_commits/1.0.0.jinja": "developer/common/git_commits/1.0.0.jinja",
    "common/merge_request_guidelines/1.0.0.jinja": "developer/common/merge_request_guidelines/1.0.0.jinja",
    "deep_code_review_reviewer/system/1.0.0.jinja": "standard_code_review_reviewer/system/1.0.0.jinja",
    "deep_code_review_reviewer/user/1.0.0.jinja": "standard_code_review_reviewer/user/1.0.0.jinja",
    "deep_code_review_summarizer/system/1.0.0.jinja": "standard_code_review_summarizer/system/1.0.0.jinja",
    "deep_code_review_summarizer/user/1.0.0.jinja": "standard_code_review_summarizer/user/1.0.0.jinja",
    "review/deep_code_review_reviewer/system/1.0.0.jinja": "standard_code_review_reviewer/system/1.0.0.jinja",
    "review/deep_code_review_reviewer/user/1.0.0.jinja": "standard_code_review_reviewer/user/1.0.0.jinja",
    "review/deep_code_review_summarizer/system/1.0.0.jinja": "standard_code_review_summarizer/system/1.0.0.jinja",
    "review/deep_code_review_summarizer/user/1.0.0.jinja": "standard_code_review_summarizer/user/1.0.0.jinja",
}

# Include names deleted on purpose, with no replacement. Only
# scripts/check_include_names.py reads this. Before you add a name, check that
# no AI Catalog flow includes it.
RETIRED_INCLUDE_NAMES: frozenset[str] = frozenset(
    {
        "common/project_activity_fetch_issues/1.0.0.jinja",
        "common/project_activity_fetch_merge_requests/1.0.0.jinja",
        "project_activity_create_summary_issue/system/1.0.0.jinja",
        "project_activity_create_summary_issue/user/1.0.0.jinja",
        "project_activity_fetch_issues_closed/user/1.0.0.jinja",
        "project_activity_fetch_issues_new/user/1.0.0.jinja",
        "project_activity_fetch_issues_updated/user/1.0.0.jinja",
        "project_activity_fetch_merge_requests_closed/user/1.0.0.jinja",
        "project_activity_fetch_merge_requests_new/user/1.0.0.jinja",
        "project_activity_fetch_merge_requests_updated/user/1.0.0.jinja",
        "project_activity_summarize_project_activity/system/1.0.0.jinja",
        "project_activity_summarize_project_activity/user/1.0.0.jinja",
    }
)

# prompt id -> the `prompts/` directory that owns it
_FEATURE_PROMPT_ROOTS: dict[str, Path] = {}

# feature `prompts/` directory -> the dirs its `subprompts_index.yml` lists
_NESTED_PROMPT_ROOTS: dict[Path, tuple[Path, ...]] = {}


def register_prompt_root(feature_id: str, prompts_dir: Path) -> None:
    """Register a ``prompts/`` dir under a prompt id.

    Args:
        feature_id: The prompt id, for example ``glab_ask_git_command`` or
            ``chat/react``.
        prompts_dir: The ``prompts/`` directory that owns the id.

    Raises:
        ValueError: If ``feature_id`` is already registered to a different path.
            Two directories sharing an id would resolve nondeterministically;
            raising surfaces the collision instead.
    """
    prompts_dir = Path(prompts_dir)
    existing = _FEATURE_PROMPT_ROOTS.get(feature_id)
    if existing is not None and existing != prompts_dir:
        raise ValueError(
            f"Duplicate prompt feature id {feature_id!r}: already registered at "
            f"{existing}, cannot also register {prompts_dir}"
        )
    _FEATURE_PROMPT_ROOTS[feature_id] = prompts_dir


def feature_prompt_root(feature_id: str) -> Path | None:
    """Return the registered ``prompts/`` dir for a prompt id, or ``None``."""
    return _FEATURE_PROMPT_ROOTS.get(feature_id)


def nested_prompt_roots(prompts_dir: Path) -> tuple[Path, ...]:
    """Return the dirs of the prompts that the index in ``prompts_dir`` lists.

    A file in one of these dirs resolves only under that prompt's id.

    Args:
        prompts_dir: A registered ``prompts/`` dir.

    Returns:
        The dirs that ``discover_feature_prompts`` read from the
        ``subprompts_index.yml`` in ``prompts_dir``, or an empty tuple.
    """
    return _NESTED_PROMPT_ROOTS.get(prompts_dir, ())


def discover_feature_prompts(features_dir: Path | None = None) -> None:
    """Register the ``prompts/`` dirs under ``ai/features/`` and ``ai/shared/``.

    ``ai/features/<domain>/<feature>/prompts/`` registers as ``<feature>`` and as
    ``<domain>/<feature>``, so a nested id such as ``chat/react`` keeps its name
    after it moves to ``ai/features/chat/react/``. Each prompt that the
    feature's ``prompts/subprompts_index.yml`` lists registers under its listed
    ID, and its files resolve only under that ID.
    ``ai/shared/<name>/prompts/`` registers as ``<name>``. A directory
    without ``prompts/`` (for example a flow-only feature) is skipped.
    Idempotent and silent when a tree is absent.

    Args:
        features_dir: The ``ai/features`` directory to scan. ``ai/shared`` is its
            sibling. Defaults to ``default_features_dir()`` when not provided.

    Raises:
        ValueError: If two directories register the same prompt id, or if a
            ``subprompts_index.yml`` is malformed or lists a directory without
            ``base/``.
    """
    root = features_dir or default_features_dir()
    shared_root = root.parent / "shared"

    # Each tree is guarded on its own: ai/shared/ can exist without ai/features/.
    if root.is_dir():
        for feature_dir in sorted(root.glob("*/*")):
            prompts_dir = feature_dir / "prompts"
            if prompts_dir.is_dir():
                register_prompt_root(feature_dir.name, prompts_dir)
                register_prompt_root(
                    f"{feature_dir.parent.name}/{feature_dir.name}", prompts_dir
                )
                subprompts = read_subprompts_index(prompts_dir)
                for prompt_id, prompt_dir in subprompts.items():
                    register_prompt_root(prompt_id, prompt_dir)
                if subprompts:
                    _NESTED_PROMPT_ROOTS[prompts_dir] = tuple(subprompts.values())

    if shared_root.is_dir():
        for shared_dir in sorted(shared_root.glob("*")):
            prompts_dir = shared_dir / "prompts"
            if prompts_dir.is_dir():
                register_prompt_root(shared_dir.name, prompts_dir)
