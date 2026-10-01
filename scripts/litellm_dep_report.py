#!/usr/bin/env python
"""Print which packages changed between two poetry.lock files, and flag major version bumps.

A litellm bump can move other packages. For example, litellm 1.103.2 moved `mcp` from 1.29.0 to 2.2.0.
This report is informational. It always exits 0.

Usage: litellm_dep_report.py BASE_LOCK HEAD_LOCK
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def load_versions(path: Path) -> dict[str, str]:
    """Map package name to locked version."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in data.get("package", [])}


def major(version: str) -> str:
    """Return the leading numeric component of a version."""
    return version.split(".", 1)[0]


def changes(
    base: dict[str, str], head: dict[str, str]
) -> list[tuple[str, str | None, str | None]]:
    """List (name, old, new) for every package that was added, removed or changed."""
    return [
        (name, base.get(name), head.get(name))
        for name in sorted(set(base) | set(head))
        if base.get(name) != head.get(name)
    ]


def render(rows: list[tuple[str, str | None, str | None]]) -> str:
    """Format the changes as a readable list."""
    if not rows:
        return "No packages changed in poetry.lock."

    lines = []
    for name, old, new in rows:
        if old is None:
            lines.append(f"added    {name} {new}")
        elif new is None:
            lines.append(f"removed  {name} {old}")
        elif major(old) != major(new):
            lines.append(f"MAJOR    {name} {old} -> {new}")
        else:
            lines.append(f"changed  {name} {old} -> {new}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Entry point for the CLI."""
    if len(argv) != 3:
        print(__doc__)
        return 0

    base_path, head_path = Path(argv[1]), Path(argv[2])
    if not base_path.exists() or base_path.stat().st_size == 0:
        print(f"No base lock file at {base_path}. Skipping the dependency report.")
        return 0

    print(render(changes(load_versions(base_path), load_versions(head_path))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
