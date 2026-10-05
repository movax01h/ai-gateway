#!/usr/bin/env python
"""Fail when the legacy prompt or flow-config roots gain a directory.

New features go under ai/features/<domain>/<feature>/, see docs/adding_and_moving_features.md. The legacy roots are
frozen: every directory that holds files under them must be listed in scripts/legacy_roots_allowlist.txt. A new version
inside a listed directory is fine. Delete lines as features move. Never add lines.
"""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path("scripts/legacy_roots_allowlist.txt")
DOC = "docs/adding_and_moving_features.md"

PROMPTS_ROOT = "ai_gateway/prompts/definitions"

# legacy root -> feature subdirectory that receives its files
LEGACY_ROOTS: dict[str, str] = {
    PROMPTS_ROOT: "prompts",
    "duo_workflow_service/agent_platform/v1/flows/configs": "config",
}


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def scan(repo_root: Path) -> set[str]:
    """Return every tracked or untracked, non-ignored file under the legacy roots."""
    # git, not the filesystem: ignored strays (.DS_Store) are skipped and a
    # symlinked directory is reported as one entry instead of being traversed.
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others"]
        + ["--exclude-standard", "--", *LEGACY_ROOTS],
        check=True,
        capture_output=True,
    ).stdout
    paths = {p for p in out.decode("utf-8").split("\0") if p}
    return {
        p for p in paths if (repo_root / p).is_symlink() or (repo_root / p).exists()
    }


def directories(files: set[str]) -> dict[str, list[str]]:
    """Group files by their parent directory."""
    grouped: dict[str, list[str]] = {}
    for path in sorted(files):
        grouped.setdefault(PurePosixPath(path).parent.as_posix(), []).append(path)
    return grouped


def nested_prompt_domains(files: set[str]) -> frozenset[str]:
    """Return the top-level prompt directories that hold two-part prompt IDs.

    A version file at ``<domain>/<feature>/<family>/<version>.yml`` marks
    ``<domain>`` as nested, for example ``chat`` for the prompt ID ``chat/react``.
    """
    domains = set()
    for path in files:
        if _under(path, PROMPTS_ROOT) and path.endswith(".yml"):
            parts = PurePosixPath(path[len(PROMPTS_ROOT) + 1 :]).parts
            if len(parts) == 4:
                domains.add(parts[0])
    return frozenset(domains)


def parse_allowlist(text: str) -> set[str]:
    """Return the directories listed in allowlist text, skipping comments and blanks."""
    lines = (line.strip().rstrip("/") for line in text.splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def load_allowlist(path: Path) -> set[str]:
    """Return the directories listed in the allowlist file at ``path``."""
    return parse_allowlist(path.read_text(encoding="utf-8"))


def load_allowlist_at(repo_root: Path, ref: str) -> set[str] | None:
    """Return the allowlist as committed at ``ref``, or None if it did not exist."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{ALLOWLIST.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return parse_allowlist(result.stdout.decode("utf-8"))


def _split(path: str) -> tuple[str, tuple[str, ...]]:
    for root, target_dir in LEGACY_ROOTS.items():
        if _under(path, root):
            return target_dir, PurePosixPath(path[len(root) + 1 :]).parts
    raise ValueError(f"{path} is not under a legacy root")


def feature_of(directory: str, nested: frozenset[str] = frozenset()) -> str:
    """Return the prompt or flow ID that owns a legacy directory.

    In a ``nested`` prompt domain the ID has two parts, for example ``chat/react``.
    A root itself gets a placeholder.
    """
    target_dir, parts = _split(directory)
    if not parts:
        return "<feature>"
    if target_dir == "prompts" and parts[0] in nested and len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def layout_b_target(directory: str, nested: frozenset[str] = frozenset()) -> str:
    """Map a legacy directory to its location under ai/features/.

    A two-part ID keeps its first part as the domain: ``chat/react`` maps to
    ``ai/features/chat/react/``. Any other ID gets a ``<domain>`` placeholder.
    """
    target_dir, parts = _split(directory)
    feature = feature_of(directory, nested)
    depth = feature.count("/") + 1 if parts else 0
    domain = "" if "/" in feature else "<domain>/"
    rest = "/".join(parts[depth:])
    return f"ai/features/{domain}{feature}/{target_dir}/{rest}".rstrip("/")


def check(
    on_disk: dict[str, list[str]],
    allowed: set[str],
    base: set[str] | None = None,
    nested: frozenset[str] = frozenset(),
) -> list[str]:
    """Return one message per new directory, stale entry, and line added since ``base``."""
    messages = []
    for directory in sorted(set(on_disk) - allowed):
        files = ", ".join(PurePosixPath(f).name for f in on_disk[directory])
        messages.append(
            f"{directory}/: new directories are not allowed under a legacy root "
            f"(added: {files}). Move the whole feature "
            f"{feature_of(directory, nested)!r} out of the legacy root (this "
            f"directory becomes {layout_b_target(directory, nested)}/, or goes to "
            f"ai/shared/ when several features use it) and delete its lines from "
            f"{ALLOWLIST}. See {DOC}."
        )
    for directory in sorted(allowed - set(on_disk)):
        messages.append(
            f"{directory}/: listed in {ALLOWLIST} but holds no files of its own "
            f"(subdirectories are listed separately). Remove that line."
        )
    if base is not None:
        # Only roots the base already froze; a newly added root ships its own lines.
        frozen = [r for r in LEGACY_ROOTS if any(_under(e, r) for e in base)]
        for directory in sorted(allowed - base):
            if any(_under(directory, r) for r in frozen):
                messages.append(
                    f"{directory}/: added to {ALLOWLIST}. The allowlist only "
                    f"shrinks; move the feature to ai/features/ instead. See {DOC}."
                )
    return messages


def main(argv: list[str] | None = None) -> int:
    """Run the lint on the repo and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        metavar="REF",
        help="also fail when the allowlist gained lines since this git ref",
    )
    args = parser.parse_args(argv)
    base = load_allowlist_at(REPO_ROOT, args.base) if args.base else None
    allowed = load_allowlist(REPO_ROOT / ALLOWLIST)
    files = scan(REPO_ROOT)
    messages = check(directories(files), allowed, base, nested_prompt_domains(files))
    for message in messages:
        print(message, file=sys.stderr)
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
