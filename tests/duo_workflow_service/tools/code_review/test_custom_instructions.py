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
        "instructions:\n  - fileFilters: ['*.rb']\n    instructions: x",
        "instructions: [\n",
    ],
    ids=[
        "none",
        "empty",
        "scalar",
        "instructions-not-a-list",
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


def test_scalar_file_filters_do_not_become_a_match_all():
    """A string is iterable, so a scalar must be wrapped rather than iterated per character."""
    parsed = parse_instructions(
        'instructions:\n  - name: N\n    fileFilters: "*.rb"\n    instructions: Do it\n'
    )

    assert parsed[0]["include_patterns"] == ["*.rb"]
    assert filter_matching_instructions(parsed, ["app/a.rb"])
    assert not filter_matching_instructions(parsed, ["docs/README.md"])


USABLE_INSTRUCTION = (
    "  - name: Good\n    fileFilters: ['*.rb']\n    instructions: Do it\n"
)


@pytest.mark.parametrize(
    "unusable",
    [
        "  - name: N\n    fileFilters: ['*.rb']\n    instructions:\n      - a\n      - b\n",
        "  - name: N\n    fileFilters: ['*.rb']\n    instructions: 42\n",
        "  - name: {a: 1}\n    fileFilters: ['*.rb']\n    instructions: Do it\n",
        "  - name: N\n    fileFilters: [1, 2]\n    instructions: Do it\n",
        "  - name: N\n    fileFilters: 42\n    instructions: Do it\n",
        "  - name: N\n    fileFilters: ['*.rb']\n    instructions: '   '\n",
        "  - name: ''\n    fileFilters: ['*.rb']\n    instructions: Do it\n",
        "  - name: '   '\n    fileFilters: ['*.rb']\n    instructions: Do it\n",
    ],
    ids=[
        "list-body",
        "int-body",
        "non-string-name",
        "non-string-filter-entry",
        "non-list-filters",
        "blank-body",
        "empty-name",
        "whitespace-name",
    ],
)
def test_an_unusable_instruction_is_skipped_rather_than_emptying_the_file(unusable):
    """A usable sibling is what makes this test mean anything.

    Asserting an empty result would pass just as well when the blanket `except` in
    `parse_instructions` abandons the whole file, which is the failure being guarded against.
    """
    parsed = parse_instructions("instructions:\n" + unusable + USABLE_INSTRUCTION)

    assert [instruction["name"] for instruction in parsed] == ["Good"]


@pytest.mark.parametrize(
    "filters",
    ["", "\n    fileFilters:", "\n    fileFilters: []"],
    ids=["absent", "null", "empty-list"],
)
def test_an_unscoped_instruction_applies_to_every_file(filters):
    """Documented as applying everywhere, and what Rails produces from `Array(nil)`.

    `_matches_pattern` and `format_instructions` already treat an empty include list that way,
    so only the normalizer stood between an author and the documented behaviour.
    """
    parsed = parse_instructions(
        f"instructions:\n  - name: Everywhere{filters}\n    instructions: Be terse\n"
    )

    assert parsed[0]["include_patterns"] == []
    assert filter_matching_instructions(parsed, ["app/a.rb"])
    assert filter_matching_instructions(parsed, ["docs/README.md"])
    assert 'For files matching "all files"' in format_instructions(parsed)


def test_an_unscoped_instruction_still_honours_its_exclusions():
    """An exclude-only list is the one shape that already reached this behaviour."""
    parsed = parse_instructions(
        "instructions:\n  - name: Everywhere\n"
        '    fileFilters: ["!vendor/**"]\n    instructions: Be terse\n'
    )

    assert filter_matching_instructions(parsed, ["app/a.rb"])
    assert not filter_matching_instructions(parsed, ["vendor/a.rb"])


def test_a_filter_list_with_a_non_string_entry_is_rejected_whole():
    """An exclude-only instruction matches everything outside those excludes.

    Keeping only the usable entries can leave exactly that, so the whole list is rejected.
    """
    parsed = parse_instructions(
        "instructions:\n"
        "  - name: N\n"
        '    fileFilters: [{path: "*.rb"}, "!vendor/**"]\n'
        "    instructions: Do it\n"
    )

    assert parsed == []


def test_a_valid_instruction_alongside_a_malformed_one_still_applies():
    """Only the malformed instruction is dropped; it does not take the file with it."""
    parsed = parse_instructions(
        "instructions:\n"
        "  - name: Bad\n"
        "    fileFilters: [1, 2]\n"
        "    instructions: Do it\n"
        "  - name: Good\n"
        '    fileFilters: ["*.rb"]\n'
        "    instructions: Do it\n"
    )

    assert [i["name"] for i in parsed] == ["Good"]
    assert filter_matching_instructions(parsed, ["app/a.rb"])
    assert not filter_matching_instructions(parsed, ["docs/README.md"])


@pytest.mark.parametrize(
    "filters",
    ['""', '["", "!vendor/**"]', '["*.rb", ""]', '["!"]', '["*.rb", "!  "]'],
    ids=[
        "scalar-blank",
        "blank-with-exclude",
        "blank-alongside-include",
        "bare-exclusion",
        "blank-exclusion",
    ],
)
def test_a_blank_pattern_rejects_the_instruction(filters):
    """A blank pattern scopes nothing, and dropping it can leave an exclude-only instruction."""
    parsed = parse_instructions(
        f"instructions:\n  - name: N\n    fileFilters: {filters}\n    instructions: Do it\n"
    )

    assert parsed == []


def test_a_name_is_stored_stripped():
    """The reviewer copies the name into `custom_instruction_ref`, which the published comment quotes verbatim, so the
    padding must not survive parsing."""
    parsed = parse_instructions(
        'instructions:\n  - name: "  My Instruction  "\n'
        '    fileFilters: ["*.rb"]\n    instructions: Do it\n'
    )

    assert parsed[0]["name"] == "My Instruction"
    assert "- My Instruction:" in format_instructions(parsed)


def test_a_padded_include_pattern_still_matches():
    """Fnmatch compares the pattern literally, so stray whitespace would match no real path."""
    parsed = parse_instructions(
        'instructions:\n  - name: N\n    fileFilters: ["*.rb "]\n    instructions: Do it\n'
    )

    assert parsed[0]["include_patterns"] == ["*.rb"]
    assert filter_matching_instructions(parsed, ["app/a.rb"])


@pytest.mark.parametrize(
    "exclusion",
    ["!vendor/** ", "! vendor/**", " ! vendor/** "],
    ids=["after", "inside", "both"],
)
def test_a_padded_exclude_pattern_still_excludes(exclusion):
    """On an exclusion the failure widens rather than narrows: the instruction would apply to
    the very files the author excluded."""
    parsed = parse_instructions(
        "instructions:\n  - name: N\n"
        f'    fileFilters: ["*.rb", "{exclusion}"]\n    instructions: Do it\n'
    )

    assert parsed[0]["exclude_patterns"] == ["vendor/**"]
    assert filter_matching_instructions(parsed, ["app/a.rb"])
    assert not filter_matching_instructions(parsed, ["vendor/a.rb"])


def rails_resolved(**overrides):
    """One instruction in the shape the Rails resolver returns, which skips `parse_instructions`."""
    return {
        "name": "Ruby Style",
        "instructions": "Follow Ruby naming conventions",
        "include_patterns": ["*.rb"],
        "exclude_patterns": [],
        **overrides,
    }


@pytest.mark.parametrize(
    "body",
    [["a", "b"], 42, None, "   "],
    ids=["list", "number", "null", "blank"],
)
def test_a_rails_resolved_body_that_is_not_text_is_skipped(body):
    """Rails checks only `present?`, so a list or number body arrives unvalidated.

    Rendering it called `.strip()` on a non-string and took the whole review down with an
    `AttributeError`.
    """
    good = rails_resolved(name="Good")
    rendered = format_instructions(
        [rails_resolved(instructions=body), good], include_format_hint=False
    )

    assert "- Good:" in rendered
    assert "- Ruby Style:" not in rendered


def test_a_rails_resolved_name_renders_stripped():
    """`_normalize_instruction` strips the name, but nothing on the Rails path does."""
    rendered = format_instructions(
        [rails_resolved(name="  Padded Name  ")], include_format_hint=False
    )

    assert "- Padded Name:" in rendered


def test_no_renderable_instruction_emits_no_block():
    """An empty block still instructs the model to apply instructions that are not there."""
    assert (
        format_instructions(
            [rails_resolved(instructions=42)], include_format_hint=False
        )
        == ""
    )
