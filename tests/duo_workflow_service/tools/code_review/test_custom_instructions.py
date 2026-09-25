import pytest

from duo_workflow_service.tools.code_review.custom_instructions import (
    filter_matching_instructions,
    format_instructions,
    parse_instructions,
)

VALID_YAML = """
instructions:
  - name: Ruby Style
    fileFilters:
      - "**/*.rb"
      - "!**/spec/**"
    instructions: |
      Follow Ruby naming conventions
  - name: Everything
    fileFilters:
      - "!**/*.md"
    instructions: Keep it simple
"""


def test_parse_instructions_normalizes_file_filters():
    result = parse_instructions(VALID_YAML)

    assert result == [
        {
            "name": "Ruby Style",
            "instructions": "Follow Ruby naming conventions\n",
            "include_patterns": ["**/*.rb"],
            "exclude_patterns": ["**/spec/**"],
        },
        {
            "name": "Everything",
            "instructions": "Keep it simple",
            "include_patterns": [],
            "exclude_patterns": ["**/*.md"],
        },
    ]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "not a mapping",
        "instructions: not-a-list",
        "instructions:\n  - name: no filters\n    instructions: x",
        "instructions:\n  - fileFilters: ['*.rb']\n    instructions: x",
        "instructions: [\n",
    ],
    ids=[
        "none",
        "empty",
        "scalar",
        "instructions-not-a-list",
        "missing-file-filters",
        "missing-name",
        "invalid-yaml",
    ],
)
def test_parse_instructions_degrades_to_empty(content):
    assert parse_instructions(content) == []


def test_filter_matching_instructions_uses_include_and_exclude_patterns():
    instructions = parse_instructions(VALID_YAML)

    assert [
        i["name"] for i in filter_matching_instructions(instructions, ["app/a.rb"])
    ] == [
        "Ruby Style",
        "Everything",
    ]
    assert [
        i["name"]
        for i in filter_matching_instructions(instructions, ["app/spec/a_spec.rb"])
    ] == ["Everything"]
    assert filter_matching_instructions(instructions, ["docs/README.md"]) == []
    assert filter_matching_instructions([], ["app/a.rb"]) == []


def test_format_instructions():
    assert format_instructions([]) == ""

    result = format_instructions(parse_instructions(VALID_YAML))

    assert result.startswith("<custom_instructions>")
    assert result.endswith("</custom_instructions>")
    assert (
        'For files matching "**/*.rb" (excluding: **/spec/**) - Ruby Style:' in result
    )
    assert 'For files matching "all files" (excluding: **/*.md) - Everything:' in result
