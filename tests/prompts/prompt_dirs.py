"""The prompt directories and prompt version files that the prompt registry loads, for tests that check every prompt."""

from pathlib import Path

from lib.subprompts_index import read_subprompts_index

__all__ = [
    "FEATURE_SETTINGS_THAT_RAILS_SENDS",
    "REPO_ROOT",
    "feature_dir_of",
    "feature_prompt_dirs",
    "prompt_dir",
    "prompt_ids_by_dir",
    "version_files",
]

REPO_ROOT = Path(__file__).parents[2]
_LEGACY_DIR = REPO_ROOT / "ai_gateway" / "prompts" / "definitions"
_FEATURES_DIR = REPO_ROOT / "ai" / "features"
_SHARED_DIR = REPO_ROOT / "ai" / "shared"


# The feature setting that GitLab Rails sends for each prompt that it calls on the prompt routes. Each version of the
# prompt declares it as `models_from`, except the versions that `tests/prompts/test_registry.py` lists.
FEATURE_SETTINGS_THAT_RAILS_SENDS = {
    "code_suggestions/completions": "code_completions",
    "code_suggestions/generations": "code_generations",
    "categorize_question": "duo_chat",
    "chat/build_reader": "duo_chat",
    "chat/commit_reader": "duo_chat",
    "chat/documentation_search": "duo_chat",
    "chat/epic_reader": "duo_chat",
    "chat/issue_reader": "duo_chat",
    "chat/merge_request_reader": "duo_chat",
    "chat/work_item_reader": "duo_chat",
    "chat/explain_code": "duo_chat_explain_code",
    "chat/explain_vulnerability": "duo_chat_explain_vulnerability",
    "chat/fix_code": "duo_chat_fix_code",
    "chat/refactor_code": "duo_chat_refactor_code",
    "chat/summarize_comments": "duo_chat_summarize_comments",
    "chat/troubleshoot_job": "duo_chat_troubleshoot_job",
    "chat/write_tests": "duo_chat_write_tests",
    "classify_code_review_mention_intent": "duo_agent_platform",
    "generate_session_title": "duo_agent_platform",
    "summarize_duo_workflow": "duo_agent_platform",
    "feature_discovery_search": "feature_discovery_search",
    "generate_commit_message": "generate_commit_message",
    "glab_ask_git_command": "glab_ask_git_command",
    "resolve_vulnerability": "resolve_vulnerability",
    "review_merge_request": "review_merge_request",
    "summarize_new_merge_request": "summarize_new_merge_request",
    "summarize_review": "summarize_review",
}


def version_files(prompts_dir: Path) -> list[Path]:
    """Return the ``<model family or base>/<version>.yml`` files in ``prompts_dir``."""
    return sorted(prompts_dir.glob("*/*.yml"))


def feature_prompt_dirs() -> dict[str, Path]:
    """Return the `prompts/` dir of each prompt ID that `discover_feature_prompts` registers and that loads a prompt.

    A flow-only feature keeps only Jinja fragments in its `prompts/` dir. Its ID loads no prompt. Each prompt that a
    `prompts/subprompts_index.yml` lists maps to its own dir.
    """
    dirs: dict[str, Path] = {}
    for feature_dir in _FEATURES_DIR.glob("*/*"):
        if version_files(prompts_dir := feature_dir / "prompts"):
            dirs[feature_dir.name] = prompts_dir
            dirs[f"{feature_dir.parent.name}/{feature_dir.name}"] = prompts_dir
        dirs.update(read_subprompts_index(prompts_dir))
    for shared_dir in _SHARED_DIR.glob("*"):
        if version_files(prompts_dir := shared_dir / "prompts"):
            dirs[shared_dir.name] = prompts_dir
    return dirs


def feature_dir_of(prompts_dir: Path) -> Path:
    """Return the ``ai/features/<domain>/<feature>`` dir that owns ``prompts_dir``.

    Args:
        prompts_dir: A feature's ``prompts/`` dir, or the dir of a prompt that its ``subprompts_index.yml`` lists.

    Raises:
        ValueError: If ``prompts_dir`` is not under ``ai/features/<domain>/<feature>/``.
    """
    parts = prompts_dir.relative_to(_FEATURES_DIR).parts
    if len(parts) < 3:
        raise ValueError(f"{prompts_dir} is not under ai/features/<domain>/<feature>/")
    return _FEATURES_DIR / parts[0] / parts[1]


def prompt_dir(prompt_id: str) -> Path:
    """Return the directory that the registry resolves ``prompt_id`` to."""
    return feature_prompt_dirs().get(prompt_id, _LEGACY_DIR / prompt_id)


def prompt_ids_by_dir() -> dict[Path, set[str]]:
    """Return every prompt directory, with the prompt IDs that resolve to it."""
    prompt_ids: dict[Path, set[str]] = {}
    for version_file in _LEGACY_DIR.rglob("*.yml"):
        # <prompt_id>/<model family or base>/<version>.yml
        parts = version_file.relative_to(_LEGACY_DIR).parts[:-2]
        if parts:
            prompt_ids.setdefault(_LEGACY_DIR.joinpath(*parts), set()).add(
                "/".join(parts)
            )
    for prompt_id, prompts_dir in feature_prompt_dirs().items():
        prompt_ids.setdefault(prompts_dir, set()).add(prompt_id)
    return prompt_ids
