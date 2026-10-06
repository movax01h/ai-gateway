"""Parsing, matching and rendering of code review custom instructions.

The instructions reach the service from the GitLab API for merge-request review. Everything downstream of that fetch is
independent of how they were obtained and lives here, so a caller that reads `.gitlab/duo/mr-review-instructions.yaml`
from a working tree instead shares the parsing and rendering unchanged.
"""

import fnmatch
from typing import Any, Dict, List, Optional

import yaml

INSTRUCTIONS_FILE_PATH = ".gitlab/duo/mr-review-instructions.yaml"

# Flows whose terminal step renders the attribution itself from a finding's
# `custom_instruction_ref` pass `include_format_hint=False`, so the model is not told to
# write it too and comments are not attributed twice.
CUSTOM_INSTRUCTION_FORMAT_HINT = """

When commenting based on custom instructions, format as:
"According to custom instructions in '[instruction_name]' ([brief paraphrase of relevant instruction]): [your specific comment about the code]"

Example: "According to custom instructions in 'Security Best Practices' (validate all API input): This endpoint should validate input parameters to prevent SQL injection."

This formatting is only required for custom instruction comments. Regular review comments based on standard review criteria should NOT include this prefix."""


def parse_instructions(content: Optional[str]) -> List[Dict[str, Any]]:
    """Parse YAML custom instructions content into the normalized instruction shape."""
    if not content:
        return []

    # Malformed instruction files degrade to "no instructions" rather than failing the
    # review; the shapes yaml.safe_load can return are too open to enumerate.
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict) or not isinstance(data.get("instructions"), list):
            return []

        normalized = (
            _normalize_instruction(item)
            for item in data["instructions"]
            if isinstance(item, dict)
        )

        return [instruction for instruction in normalized if instruction is not None]
    except Exception:
        return []


def filter_matching_instructions(
    all_instructions: List[Dict[str, Any]], diff_file_paths: List[str]
) -> List[Dict[str, Any]]:
    """Keep only instructions matching at least one of the changed files."""
    if not all_instructions:
        return []

    return [
        instruction
        for instruction in all_instructions
        if any(_matches_pattern(path, instruction) for path in diff_file_paths)
    ]


def format_instructions(
    custom_instructions: List[Dict[str, Any]],
    include_format_hint: bool = True,
) -> str:
    """Render the `<custom_instructions>` block, or an empty string when nothing renders."""
    instruction_items = []
    for instruction in custom_instructions:
        body = instruction.get("instructions")
        # GitLab 19.0+ resolves these in Rails, whose `valid?` is `name.present? &&
        # instructions.present?`, so a list or numeric body never passes through the parser.
        if not isinstance(body, str) or not body.strip():
            continue

        include_patterns = ", ".join(instruction["include_patterns"]) or "all files"
        exclude_patterns = ", ".join(instruction["exclude_patterns"]) or "none"

        instruction_items.append(
            f'For files matching "{include_patterns}" '
            f"(excluding: {exclude_patterns}) - {str(instruction['name']).strip()}:\n"
            f"{body.strip()}\n"
        )

    # An empty block would still tell the model to apply instructions that are not there.
    if not instruction_items:
        return ""

    instructions_text = "\n".join(instruction_items)
    format_hint = CUSTOM_INSTRUCTION_FORMAT_HINT if include_format_hint else ""

    return f"""<custom_instructions>
Apply these additional review instructions to matching files:

{instructions_text}
IMPORTANT: Only apply each custom instruction to files that match its specified pattern. If a file doesn't match any custom instruction pattern, only apply the standard review criteria.{format_hint}
</custom_instructions>"""


def _normalize_instruction(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one instruction, or None when it cannot scope anything.

    Usability is decided here rather than in a separate predicate, because whether an instruction is usable is the same
    question as whether it normalizes.
    """
    name = item.get("name")
    instructions = item.get("instructions")
    # A scalar here would otherwise iterate per character, and the resulting `*` pattern
    # turns a scoped instruction into one that matches every file.
    file_filters = _as_patterns(item.get("fileFilters"))

    if not (
        isinstance(name, str)
        and name.strip()
        and isinstance(instructions, str)
        and instructions.strip()
        and file_filters is not None
    ):
        return None

    return {
        # Stripped on the way in, not at render: the reviewer copies this into a finding's
        # `custom_instruction_ref`, which is quoted verbatim in the published comment.
        "name": name.strip(),
        "instructions": instructions,
        "include_patterns": [f for f in file_filters if not f.startswith("!")],
        "exclude_patterns": [f[1:] for f in file_filters if f.startswith("!")],
    }


def _as_patterns(value: Any) -> Optional[List[str]]:
    """The patterns scoping an instruction, empty when it is unscoped, None when it cannot scope.

    An absent `fileFilters` means every file, which is what the documentation promises and what
    Rails produces from `Array(nil)`. Empty is therefore distinct from unusable, which rejects
    the instruction.
    """
    if value is None:
        return []

    patterns = [value] if isinstance(value, str) else value
    if not isinstance(patterns, list):
        return None

    if not all(isinstance(item, str) for item in patterns):
        return None

    normalized = [_strip_pattern(item) for item in patterns]

    # Rejected whole rather than filtered: dropping an unusable entry can leave an
    # exclude-only instruction, which matches every file outside those excludes.
    if not all(item.removeprefix("!") for item in normalized):
        return None

    return normalized


def _strip_pattern(item: str) -> str:
    """Strip around any leading `!`, so a padded exclusion still names what it excludes.

    Stripping only the outside leaves `"! vendor/**"` excluding `" vendor/**"`, which matches nothing, so the
    instruction would apply to the very files the author excluded.
    """
    item = item.strip()

    return "!" + item[1:].strip() if item.startswith("!") else item


def _matches_pattern(path: str, instruction: Dict[str, Any]) -> bool:
    includes = instruction.get("include_patterns", [])
    excludes = instruction.get("exclude_patterns", [])

    # With include patterns: match only files matching includes (minus exclusions).
    # Without include patterns: match all files (minus exclusions).
    matches_include = not includes or any(
        fnmatch.fnmatch(path, pattern) for pattern in includes
    )
    matches_exclude = any(fnmatch.fnmatch(path, pattern) for pattern in excludes)

    return matches_include and not matches_exclude
