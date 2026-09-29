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
        if not isinstance(data, dict) or "instructions" not in data:
            return []

        return [
            _normalize_instruction(item)
            for item in data["instructions"]
            if isinstance(item, dict) and _is_valid_instruction(item)
        ]
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
    """Render the `<custom_instructions>` block, or an empty string when there are none."""
    if not custom_instructions:
        return ""

    instruction_items = []
    for instruction in custom_instructions:
        include_patterns = ", ".join(instruction["include_patterns"]) or "all files"
        exclude_patterns = ", ".join(instruction["exclude_patterns"]) or "none"

        instruction_items.append(
            f'For files matching "{include_patterns}" '
            f"(excluding: {exclude_patterns}) - {instruction['name']}:\n"
            f"{instruction['instructions'].strip()}\n"
        )

    instructions_text = "\n".join(instruction_items)
    format_hint = CUSTOM_INSTRUCTION_FORMAT_HINT if include_format_hint else ""

    return f"""<custom_instructions>
Apply these additional review instructions to matching files:

{instructions_text}
IMPORTANT: Only apply each custom instruction to files that match its specified pattern. If a file doesn't match any custom instruction pattern, only apply the standard review criteria.{format_hint}
</custom_instructions>"""


def _is_valid_instruction(item: Dict[str, Any]) -> bool:
    return bool(
        item.get("name") and item.get("instructions") and item.get("fileFilters")
    )


def _normalize_instruction(item: Dict[str, Any]) -> Dict[str, Any]:
    file_filters = item.get("fileFilters", [])

    return {
        "name": item.get("name"),
        "instructions": item.get("instructions"),
        "include_patterns": [f for f in file_filters if not f.startswith("!")],
        "exclude_patterns": [f[1:] for f in file_filters if f.startswith("!")],
    }


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
