#!/usr/bin/env python
"""Post a prompt version diff comment on a GitLab merge request.

Prompts are versioned as separate files under ``ai_gateway/prompts/definitions``
(e.g. ``system/1.0.0.jinja`` -> ``system/1.1.0.jinja``). Because each version is
a new file, the MR diff shows only an addition and reviewers cannot see what
changed. This script detects new versioned prompt files in an MR, resolves the
immediately preceding version on the target branch, computes a unified diff and
posts it as a single MR comment.

Idempotency: the note body embeds a sentinel HTML comment. On re-runs the
existing note is edited rather than duplicated, and it is deleted when the MR
no longer adds any versioned prompt files.

Required environment variables (available in MR pipelines):
    CI_MERGE_REQUEST_PROJECT_ID
    CI_MERGE_REQUEST_IID
    CI_MERGE_REQUEST_TARGET_BRANCH_NAME
    GITLAB_TOKEN  - token with ``api`` scope able to create MR notes

Optional:
    CI_SERVER_URL - defaults to ``https://gitlab.com``
"""

import difflib
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import gitlab
import gitlab.exceptions
from poetry.core.constraints.version import Version

# Sentinel embedded in the bot comment so re-runs can find and update it.
SENTINEL = "<!-- prompt-diff-bot -->"

PROMPTS_ROOT = "ai_gateway/prompts/definitions"

VERSIONED_EXTENSIONS = frozenset({".jinja", ".yml"})


@dataclass(frozen=True)
class PromptDiff:
    """Diff between two versions of one prompt file."""

    key: str
    old_version: str
    new_version: str
    diff: str


def parse_version(stem: str) -> Version | None:
    """Parse a filename stem such as ``1.1.0`` or ``1.1.0-dev`` into a ``Version``.

    Returns ``None`` for stems that are not valid versions (e.g. ``2.0.0-orbit``).
    Pre-release versions sort before the corresponding stable version, matching
    the resolution rules in ``lib/version.py``.
    """
    try:
        return Version.parse(stem)
    except ValueError:
        return None


def is_versioned_prompt_file(path: str) -> bool:
    """Return True if *path* is a versioned prompt file inside ``PROMPTS_ROOT``."""
    p = PurePosixPath(path)
    if not path.startswith(PROMPTS_ROOT + "/"):
        return False
    if p.suffix not in VERSIONED_EXTENSIONS:
        return False
    return parse_version(p.stem) is not None


def highest_version_below(candidate_stems: list[str], new_stem: str) -> str | None:
    """Return the highest version stem strictly below *new_stem*, or ``None``."""
    new_version = parse_version(new_stem)
    if new_version is None:
        return None

    below: dict[str, Version] = {}
    for stem in candidate_stems:
        version = parse_version(stem)
        if version is not None and version < new_version:
            below[stem] = version
    if not below:
        return None
    return max(below, key=below.__getitem__)


def prompt_key(path: str) -> str:
    """Return a human-readable key like ``chat/explain_code/system`` for a prompt path."""
    return str(PurePosixPath(path).relative_to(PROMPTS_ROOT).parent)


def compute_diff(old_text: str, old_label: str, new_text: str, new_label: str) -> str:
    """Return a unified diff between *old_text* and *new_text*."""
    diff_lines = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=old_label,
        tofile=new_label,
    )
    diff = "".join(diff_lines)
    return diff if diff else "(no textual differences)"


def code_fence(text: str) -> str:
    """Return a backtick fence longer than any backtick run in *text*.

    Prompt bodies often contain ``` fences of their own, which would otherwise close the outer fence early.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def build_comment_body(diffs: list[PromptDiff]) -> str:
    """Build the Markdown comment body."""
    lines = [SENTINEL, "", "### Prompt version diff", ""]
    for entry in diffs:
        summary = f"`{entry.key}: {entry.old_version} → {entry.new_version}`"
        fence = code_fence(entry.diff)
        lines += [
            "<details>",
            f"<summary>{summary}</summary>",
            "",
            f"{fence}diff",
            entry.diff.rstrip("\n"),
            fence,
            "",
            "</details>",
            "",
        ]
    return "\n".join(lines)


def list_mr_diffs(mr: Any) -> list[dict[str, Any]]:
    """Return all file diff entries of *mr* via the paginated ``/diffs`` endpoint.

    ``mr.changes()`` is deprecated and silently drops files on overflow.
    """
    return list(
        mr.manager.gitlab.http_list(
            f"{mr.manager.path}/{mr.encoded_id}/diffs", get_all=True
        )
    )


def detect_new_prompt_files(diffs: list[dict[str, Any]]) -> list[str]:
    """Return paths of added versioned prompt files from MR diff entries."""
    return sorted(
        entry["new_path"]
        for entry in diffs
        if entry.get("new_file") and is_versioned_prompt_file(entry["new_path"])
    )


def list_target_branch_stems(
    project: Any, directory: str, extension: str, ref: str
) -> list[str]:
    """Return version stems of ``*extension`` files in *directory* on *ref*."""
    try:
        items = project.repository_tree(path=directory, ref=ref, get_all=True)
    except gitlab.exceptions.GitlabGetError:
        # Directory does not exist on the target branch yet (brand-new prompt).
        return []
    return [
        PurePosixPath(item["name"]).stem
        for item in items
        if item["type"] == "blob" and PurePosixPath(item["name"]).suffix == extension
    ]


def build_prompt_diff(
    project: Any,
    file_path: str,
    target_branch: str,
    repo_root: Path,
    added_stems: Sequence[str] = (),
) -> PromptDiff | None:
    """Diff *file_path* (read from the checkout) against its predecessor.

    Candidates are the versions on *target_branch* plus *added_stems*, the other versions the MR adds in the same
    directory. A predecessor from *added_stems* is read from the checkout instead of the target branch.
    """
    posix_path = PurePosixPath(file_path)
    directory = str(posix_path.parent)
    extension = posix_path.suffix
    new_stem = posix_path.stem

    try:
        new_text = (repo_root / file_path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: cannot read {file_path}: {exc}", file=sys.stderr)
        return None

    siblings = list_target_branch_stems(project, directory, extension, target_branch)
    prev_stem = highest_version_below([*siblings, *added_stems], new_stem)
    if prev_stem is None:
        print(f"  {file_path}: no previous version on {target_branch!r}; skipping.")
        return None

    prev_path = f"{directory}/{prev_stem}{extension}"
    try:
        if prev_stem in added_stems:
            old_text = (repo_root / prev_path).read_text(encoding="utf-8")
        else:
            old_text = project.files.raw(file_path=prev_path, ref=target_branch).decode(
                "utf-8"
            )
    except (OSError, gitlab.exceptions.GitlabGetError) as exc:
        print(f"Warning: cannot fetch {prev_path}: {exc}", file=sys.stderr)
        return None

    key = prompt_key(file_path)
    print(f"  {key}: {prev_stem} → {new_stem}")
    return PromptDiff(
        key=key,
        old_version=prev_stem,
        new_version=new_stem,
        diff=compute_diff(
            old_text,
            f"{key}/{prev_stem}{extension}",
            new_text,
            f"{key}/{new_stem}{extension}",
        ),
    )


def find_existing_note(mr: Any) -> Any | None:
    """Return the existing bot note on *mr*, or ``None``."""
    for note in mr.notes.list(get_all=True):
        if SENTINEL in note.body:
            return note
    return None


def sync_comment(mr: Any, diffs: list[PromptDiff]) -> None:
    """Create, update or delete the bot comment so it reflects *diffs*."""
    existing = find_existing_note(mr)
    if not diffs:
        if existing:
            print("Deleting stale prompt diff comment.")
            existing.delete()
        return

    body = build_comment_body(diffs)
    if existing:
        print("Updating existing prompt diff comment.")
        existing.body = body
        existing.save()
    else:
        print("Creating prompt diff comment.")
        mr.notes.create({"body": body})


def run(project: Any, mr: Any, target_branch: str, repo_root: Path) -> list[PromptDiff]:
    """Compute prompt diffs for *mr* and sync the MR comment; returns the diffs."""
    new_files = detect_new_prompt_files(list_mr_diffs(mr))
    if new_files:
        print(f"Found {len(new_files)} new versioned prompt file(s):")
    else:
        print("No new versioned prompt files detected.")

    added_by_dir: dict[tuple[str, str], list[str]] = defaultdict(list)
    for file_path in new_files:
        p = PurePosixPath(file_path)
        added_by_dir[(str(p.parent), p.suffix)].append(p.stem)

    diffs = []
    for file_path in new_files:
        p = PurePosixPath(file_path)
        others = [s for s in added_by_dir[(str(p.parent), p.suffix)] if s != p.stem]
        diff = build_prompt_diff(project, file_path, target_branch, repo_root, others)
        if diff:
            diffs.append(diff)
    sync_comment(mr, diffs)
    return diffs


def main() -> int:
    """Entry point; returns an exit code."""
    # pylint: disable=direct-environment-variable-reference
    project_id = os.environ.get("CI_MERGE_REQUEST_PROJECT_ID", "")
    mr_iid = os.environ.get("CI_MERGE_REQUEST_IID", "")
    target_branch = os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", "")
    token = os.environ.get("GITLAB_TOKEN", "")
    server_url = os.environ.get("CI_SERVER_URL", "https://gitlab.com")
    # pylint: enable=direct-environment-variable-reference

    missing = [
        name
        for name, value in (
            ("CI_MERGE_REQUEST_PROJECT_ID", project_id),
            ("CI_MERGE_REQUEST_IID", mr_iid),
            ("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", target_branch),
            ("GITLAB_TOKEN", token),
        )
        if not value
    ]
    if missing:
        print(f"Error: missing environment variables: {missing}", file=sys.stderr)
        return 1

    gl = gitlab.Gitlab(server_url, private_token=token)
    project = gl.projects.get(project_id)
    mr = project.mergerequests.get(int(mr_iid))

    print(f"Scanning MR !{mr_iid} for new versioned prompt files…")
    run(project, mr, target_branch, Path.cwd())
    return 0


if __name__ == "__main__":
    sys.exit(main())
