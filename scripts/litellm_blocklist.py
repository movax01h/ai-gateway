#!/usr/bin/env python
"""Fail if the installed litellm is a version we already know is broken.

Twice now a known-bad litellm release has been identified, worked around, and
then had the knowledge evaporate when the excluding merge request was closed.
``scripts/litellm_blocklist.yml`` is the permanent record, and this script is
what enforces it in CI.
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

BLOCKLIST_PATH = Path(__file__).resolve().parent / "litellm_blocklist.yml"


def load_blocklist(path: Path = BLOCKLIST_PATH) -> list[dict[str, Any]]:
    """Read the blocklist file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("blocked") or [])


def installed_version() -> str:
    """Return the installed litellm version."""
    return metadata.version("litellm")


def find_violations(
    version: str, blocked: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return every blocklist entry matching the given version."""
    parsed = Version(version)
    return [
        entry
        for entry in blocked
        if parsed in SpecifierSet(str(entry["specifier"]), prereleases=True)
    ]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="version to check instead of the installed one (for testing)",
    )
    args = parser.parse_args(argv)

    version = args.version or installed_version()
    violations = find_violations(version, load_blocklist())

    if not violations:
        print(f"litellm {version} is not blocklisted.")
        return 0

    print(f"litellm {version} is blocklisted:\n")
    for entry in violations:
        print(f"  {entry['specifier']}")
        print(f"    {' '.join(str(entry['reason']).split())}")
        if entry.get("upstream"):
            print(f"    upstream: {entry['upstream']}")
        print()
    print(
        "Pick a version outside these ranges. If the upstream issue is fixed, "
        "remove the entry from scripts/litellm_blocklist.yml in the same merge "
        "request and say why."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
