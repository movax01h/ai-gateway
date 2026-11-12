"""Read the ``prompts/subprompts_index.yml`` file that declares the file-based prompts of a feature.

A feature with several file-based prompts keeps each one in ``prompts/<directory>/`` and lists it in the index:

.. code-block:: yaml

    project_activity_fetch_issues_new: fetch_issues_new

The index is a flat YAML mapping with one ``<prompt_id>: <directory>`` entry on each line. The AI Gateway, the scripts
under ``scripts/``, and the tests read it without a YAML library, so this module accepts only that line format, plus
comments, blank lines, and a ``---`` before the first entry.
"""

import re
from pathlib import Path

__all__ = [
    "SUBPROMPTS_INDEX",
    "parse_subprompts_index",
    "read_subprompts_index",
]

SUBPROMPTS_INDEX = "subprompts_index.yml"

# YAML needs a space, not a tab, after the colon and before a comment: `a:b` and `b#c` are plain strings.
_ENTRY = re.compile(r"(\w+): +(\w+)(?: +#.*)?", re.ASCII)


def parse_subprompts_index(text: str, source: str) -> dict[str, str]:
    """Return the prompt ID to directory name mapping in the text of an index.

    Args:
        text: The content of a ``subprompts_index.yml`` file.
        source: The file name to show in error messages.

    Returns:
        Each prompt ID mapped to its directory name, in file order.

    Raises:
        ValueError: If a line is not blank, a comment, a ``<prompt_id>: <directory>`` entry, or one ``---`` before
            the first entry, or if a prompt ID occurs twice.
    """
    entries: dict[str, str] = {}
    document_started = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "---" and not document_started:
            document_started = True
            continue
        document_started = True
        match = _ENTRY.fullmatch(line.rstrip(" "))
        if match is None:
            raise ValueError(
                f"{source}:{number}: expected '<prompt_id>: <directory>' with letters, digits, and underscores only, "
                f"got {line!r}"
            )
        prompt_id, directory = match.groups()
        if prompt_id in entries:
            raise ValueError(
                f"{source}:{number}: prompt ID {prompt_id!r} occurs more than once"
            )
        entries[prompt_id] = directory
    return entries


def read_subprompts_index(prompts_dir: Path) -> dict[str, Path]:
    """Return the prompts that the index in a feature's ``prompts/`` directory declares.

    Args:
        prompts_dir: The ``prompts/`` directory of a feature.

    Returns:
        Each declared prompt ID mapped to its directory, or an empty dict when ``prompts_dir`` has no index.

    Raises:
        ValueError: If the index is malformed (see ``parse_subprompts_index``), or if a listed directory has no
            ``base/<version>.yml`` file.
    """
    index = prompts_dir / SUBPROMPTS_INDEX
    if not index.is_file():
        return {}

    prompts: dict[str, Path] = {}
    for prompt_id, directory in parse_subprompts_index(
        index.read_text(encoding="utf-8"), str(index)
    ).items():
        prompt_dir = prompts_dir / directory
        if not any((prompt_dir / "base").glob("*.yml")):
            raise ValueError(
                f"{index}: prompt ID {prompt_id!r} lists {prompt_dir}, which has no base/<version>.yml file"
            )
        prompts[prompt_id] = prompt_dir
    return prompts
