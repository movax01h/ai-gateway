#!/usr/bin/env python
"""Fail when a Jinja include name that the base ref served no longer resolves.

Custom flows in the AI Catalog include prompt fragments by name, so every served name is part of the Flow Registry
public API. A move that changes the prefix of a name must add the old name to LEGACY_INCLUDE_ALIASES in
ai_gateway/prompts/feature_roots.py. A name deleted on purpose goes to RETIRED_INCLUDE_NAMES in the same module. See
docs/adding_and_moving_features.md.
"""

import argparse
import ast
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from lib.subprompts_index import SUBPROMPTS_INDEX, parse_subprompts_index

REPO_ROOT = Path(__file__).resolve().parents[1]
ALIASES_FILE = Path("ai_gateway/prompts/feature_roots.py")
ALIASES_NAME = "LEGACY_INCLUDE_ALIASES"
RETIRED_NAME = "RETIRED_INCLUDE_NAMES"
DOC = "docs/adding_and_moving_features.md"

LEGACY_ROOT = "ai_gateway/prompts/definitions"
FEATURES_ROOT = "ai/features"
SHARED_ROOT = "ai/shared"
ROOTS = (LEGACY_ROOT, FEATURES_ROOT, SHARED_ROOT)


def _under(path: str, root: str) -> bool:
    return path.startswith(root + "/")


def include_names(path: str) -> set[str]:
    """Return every include name that the template file at ``path`` serves.

    Mirrors the loaders in ``ai_gateway/prompts/base.py``: the legacy root serves the
    relative path, a feature serves ``<feature>/<rest>`` and ``<domain>/<feature>/<rest>``,
    and a shared dir serves ``<name>/<rest>``. A non-template file serves nothing.
    """
    if not path.endswith(".jinja"):
        return set()
    if _under(path, LEGACY_ROOT):
        return {path[len(LEGACY_ROOT) + 1 :]}
    if _under(path, FEATURES_ROOT):
        parts = PurePosixPath(path[len(FEATURES_ROOT) + 1 :]).parts
        if len(parts) > 3 and parts[2] == "prompts":
            rest = "/".join(parts[3:])
            return {f"{parts[1]}/{rest}", f"{parts[0]}/{parts[1]}/{rest}"}
    if _under(path, SHARED_ROOT):
        parts = PurePosixPath(path[len(SHARED_ROOT) + 1 :]).parts
        if len(parts) > 2 and parts[1] == "prompts":
            return {f"{parts[0]}/{'/'.join(parts[2:])}"}
    return set()


def _is_subprompts_index(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        _under(path, FEATURES_ROOT)
        and len(parts) == 6
        and parts[4:] == ("prompts", SUBPROMPTS_INDEX)
    )


def listed_prompt_dirs(paths: set[str], read: Callable[[str], str]) -> dict[str, str]:
    """Return the dir prefix of each prompt that a ``subprompts_index.yml`` lists, mapped to its prompt ID.

    Args:
        paths: The file paths to scan.
        read: Returns the text of the file at a path in ``paths``.

    Returns:
        Each ``<prompts_dir>/<directory>/`` prefix mapped to the listed prompt ID.

    Raises:
        ValueError: If an index is malformed.
    """
    prefixes: dict[str, str] = {}
    for index in sorted(filter(_is_subprompts_index, paths)):
        prompts_dir = index.rsplit("/", 1)[0]
        for prompt_id, directory in parse_subprompts_index(read(index), index).items():
            prefixes[f"{prompts_dir}/{directory}/"] = prompt_id
    return prefixes


def served_names(paths: Iterable[str], read: Callable[[str], str]) -> set[str]:
    """Return the include names that a set of file paths serves.

    A template in the dir of a listed prompt serves only ``<prompt_id>/<rest>``, like the loader.

    Args:
        paths: The file paths to scan.
        read: Returns the text of the file at a path in ``paths``.

    Returns:
        The include names that the files serve.

    Raises:
        ValueError: If a ``subprompts_index.yml`` in ``paths`` is malformed.
    """
    paths = set(paths)
    listed = listed_prompt_dirs(paths, read)
    names: set[str] = set()
    for path in paths:
        prefix = next((p for p in listed if path.startswith(p)), None)
        if prefix is None:
            names |= include_names(path)
        elif path.endswith(".jinja"):
            names.add(f"{listed[prefix]}/{path[len(prefix) :]}")
    return names


def scan(repo_root: Path) -> set[str]:
    """Return every tracked or untracked, non-ignored file under the prompt roots."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others"]
        + ["--exclude-standard", "--", *ROOTS],
        check=True,
        capture_output=True,
    ).stdout
    paths = {p for p in out.decode("utf-8").split("\0") if p}
    return {p for p in paths if (repo_root / p).exists()}


def scan_at(repo_root: Path, ref: str) -> set[str] | None:
    """Return every file under the prompt roots at ``ref``, or None when ``ref`` is unknown."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", "--name-only", ref]
        + ["--", *ROOTS],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return {p for p in result.stdout.decode("utf-8").split("\0") if p}


def show_at(repo_root: Path, ref: str, path: str) -> str:
    """Return the text of the file at ``path`` in ``ref``."""
    return subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{path}"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


def _literal(node: ast.expr) -> object:
    """Evaluate a literal, accepting ``frozenset(...)`` around one."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and not node.keywords
        and len(node.args) <= 1
    ):
        return frozenset(ast.literal_eval(node.args[0])) if node.args else frozenset()
    return ast.literal_eval(node)


def parse_constant(source: str, name: str) -> object:
    """Return the literal assigned to module constant ``name`` in Python ``source``.

    The lint reads the source instead of importing the module, so it needs none of
    the application dependencies.

    Raises:
        ValueError: If ``name`` is not assigned a literal at module level.
    """
    for node in ast.parse(source).body:
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(
            isinstance(t, ast.Name) and t.id == name for t in targets
        ):
            return _literal(value)
    raise ValueError(f"{name}: no literal assignment found")


def parse_aliases(source: str) -> dict[str, str]:
    """Return ``LEGACY_INCLUDE_ALIASES`` from Python ``source``."""
    value = parse_constant(source, ALIASES_NAME)
    if not isinstance(value, dict):
        raise ValueError(f"{ALIASES_NAME}: expected a dict literal")
    return value


def parse_retired(source: str) -> frozenset[str]:
    """Return ``RETIRED_INCLUDE_NAMES`` from Python ``source``."""
    value = parse_constant(source, RETIRED_NAME)
    if not isinstance(value, (frozenset, set)):
        raise ValueError(f"{RETIRED_NAME}: expected a frozenset literal")
    return frozenset(value)


def check(
    current: set[str],
    aliases: dict[str, str],
    retired: frozenset[str],
    base: set[str] | None = None,
) -> list[str]:
    """Return one message per broken include name and per stale alias or retired entry."""
    messages = []
    for name, target in sorted(aliases.items()):
        if name in current:
            messages.append(
                f"{name}: listed in {ALIASES_NAME} but a template still serves it, "
                f"so the alias is dead. Remove the entry from {ALIASES_FILE}."
            )
        if target not in current:
            messages.append(
                f"{name}: {ALIASES_NAME} maps it to {target!r}, which no template "
                f"serves. Point the entry in {ALIASES_FILE} at the current name."
            )
        if name in retired:
            messages.append(
                f"{name}: listed in both {ALIASES_NAME} and {RETIRED_NAME}. Keep one."
            )
    for name in sorted(retired & current):
        messages.append(
            f"{name}: listed in {RETIRED_NAME} but a template still serves it. "
            f"Remove the entry from {ALIASES_FILE}."
        )
    if base is not None:
        for name in sorted(base - current - set(aliases) - retired):
            messages.append(
                f"{name}: no template serves this include name any more. Custom "
                f"flows in the AI Catalog include it by this name. Map it to the "
                f"new name in {ALIASES_NAME}, or add it to {RETIRED_NAME} when the "
                f"fragment is gone on purpose. Both live in {ALIASES_FILE}. See {DOC}."
            )
    return messages


def main(argv: list[str] | None = None) -> int:
    """Run the lint on the repo and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        metavar="REF",
        help="also fail when an include name served at this git ref no longer resolves",
    )
    args = parser.parse_args(argv)

    base = None
    if args.base:
        base_files = scan_at(REPO_ROOT, args.base)
        if base_files is None:
            print(
                f"{args.base}: unknown git ref, skipped the removal check",
                file=sys.stderr,
            )
        else:
            base = served_names(
                base_files, lambda path: show_at(REPO_ROOT, args.base, path)
            )

    source = (REPO_ROOT / ALIASES_FILE).read_text(encoding="utf-8")
    aliases, retired = parse_aliases(source), parse_retired(source)
    current = served_names(
        scan(REPO_ROOT), lambda path: (REPO_ROOT / path).read_text(encoding="utf-8")
    )
    messages = check(current, aliases, retired, base)
    for message in messages:
        print(message, file=sys.stderr)
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
