"""Tests for the title and description text of BL findings."""

import pytest

from duo_workflow_service.bl_security import finding_text as ft
from duo_workflow_service.bl_security.finding_text import IMPACTS, SHORT_NAMES
from duo_workflow_service.bl_security.model_text import TITLE_MAX
from duo_workflow_service.tools.bl_report import IN_SCOPE_CWES


def test_every_in_scope_cwe_has_a_short_name_and_an_impact():
    assert set(SHORT_NAMES) == set(IN_SCOPE_CWES)
    assert set(IMPACTS) == set(IN_SCOPE_CWES)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Missing check, e.g. on refund, vs. cancel. Next one.",
            ["Missing check, e.g. on refund, vs. cancel.", "Next one."],
        ),
        (
            "See routes/basket.ts:18-22 and server.ts ~500. It is open.",
            ["See routes/basket.ts:18-22 and server.ts ~500.", "It is open."],
        ),
        (
            "It never checks. security.appendUserId() runs, i.e. too late! Why? Because.",
            [
                "It never checks.",
                "security.appendUserId() runs, i.e. too late!",
                "Why?",
                "Because.",
            ],
        ),
        (
            "Calls `foo. Bar()` here. Done.",
            ["Calls `foo. Bar()` here.", "Done."],
        ),
        (
            "It takes approx. 5 ms. See cf. Smith. Roles, teams etc. are open. Then more.",
            [
                "It takes approx. 5 ms.",
                "See cf. Smith.",
                "Roles, teams etc. are open.",
                "Then more.",
            ],
        ),
        (
            "It waits... then fails… and stops. Next.",
            ["It waits... then fails… and stops.", "Next."],
        ),
        (
            "Is `?id=1` checked? no, never. Is it logged? Yes.",
            ["Is `?id=1` checked? no, never.", "Is it logged?", "Yes."],
        ),
        ("Signed by J. Smith today.", ["Signed by J. Smith today."]),
        ("  multi\n line\ttext  ", ["multi line text"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert ft.split_sentences(text) == expected


def test_clip_title_drops_punctuation_before_the_ellipsis():
    assert ft.clip_title("one two, three four", limit=10) == "one two…"


def test_clip_title_never_cuts_mid_word():
    title = ft.clip_title("word " * 40, limit=30)
    assert len(title) <= 30
    assert title == "word word word word word word…"
    assert ft.clip_title("x" * 40, limit=10) == "x" * 9 + "…"
    assert ft.clip_title("short") == "short"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a.ts", "`a.ts`"),
        ("a`b", "``a`b``"),
        ("`a", "`` `a ``"),
    ],
)
def test_inline_code_escapes_backticks(text, expected):
    assert ft.inline_code(text) == expected


def test_code_block_fence_cannot_be_closed_from_inside():
    assert ft.code_block("x = ```y```\n") == "````\nx = ```y```\n````"
    assert ft.code_block("x") == "```\nx\n```"


class TestFallbackTitle:
    def test_file(self):
        assert ft.fallback_title("863", "routes/basket.ts") == (
            "Incorrect authorization in routes/basket.ts"
        )

    def test_unknown_cwe(self):
        assert ft.fallback_title("79", "a.ts") == "CWE-79 in a.ts"

    def test_no_file(self):
        assert ft.fallback_title("862", "") == "Missing authorization"

    def test_nothing_known(self):
        assert ft.fallback_title("", "") == "Business-logic finding"

    def test_no_backticks(self):
        assert ft.fallback_title("862", "a`b.ts") == "Missing authorization in ab.ts"

    def test_a_long_path_keeps_its_end(self):
        title = ft.fallback_title("915", "saleor/" * 20 + "graphql/account/types.py")
        assert len(title) <= TITLE_MAX
        # Whole directories are dropped from the left, never part of one.
        assert title.endswith(" in …saleor/graphql/account/types.py")

    def test_a_long_path_with_one_long_last_part_is_cut_on_the_left(self):
        title = ft.fallback_title("915", "d/" + "x" * 100 + ".py")
        assert len(title) == TITLE_MAX
        assert title.endswith("x.py")
        assert " in \u2026x" in title

    def test_a_long_path_never_ends_in_a_bare_ellipsis(self):
        title = ft.fallback_title("915", "x" * 100 + "/")
        assert len(title) == TITLE_MAX
        assert title.endswith("x/")

    @pytest.mark.parametrize("room", [-1, 0, 1])
    def test_a_name_with_no_room_for_the_path_stands_alone(self, monkeypatch, room):
        name = "n" * (TITLE_MAX - len(" in ") - room)
        monkeypatch.setitem(ft.SHORT_NAMES, "915", name)
        assert ft.fallback_title("915", "routes/basket.ts") == name

    def test_a_name_with_room_for_one_path_character(self, monkeypatch):
        name = "n" * (TITLE_MAX - len(" in ") - 2)
        monkeypatch.setitem(ft.SHORT_NAMES, "915", name)
        assert ft.fallback_title("915", "routes/basket.ts") == f"{name} in \u2026s"

    def test_a_long_cwe_number_stays_within_the_limit(self):
        title = ft.fallback_title("9" * 200, "routes/basket.ts")
        assert len(title) == TITLE_MAX
        assert title.startswith("CWE-999") and title.endswith("\u2026")


class TestDescription:
    def test_all_parts_in_order(self):
        text = ft.markdown_description(
            cwe="862",
            file="a.ts",
            line=3,
            excerpt="    if (x) {\n      run()\n    }\n",
            body="First. # not a heading. 1. not a list",
        )
        assert text == (
            f"**What**\n\n{IMPACTS['862']}\n\n"
            "**Where**\n\n`a.ts:3`\n\n```\nif (x) {\n  run()\n}\n```\n\n"
            "**Details**\n\n- First.\n- \\# not a heading.\n- 1\\. not a list"
        )

    def test_parts_with_nothing_to_say_are_left_out(self):
        assert (
            ft.markdown_description(cwe="79", file="", line=0, excerpt="", body="")
            == ""
        )

    def test_where_without_a_line_or_excerpt(self):
        assert ft.markdown_description(
            cwe="79", file="a.ts", line=0, excerpt=" ", body=""
        ) == ("**Where**\n\n`a.ts`")

    def test_details_escape_tags_and_mentions_outside_code(self):
        text = ft.markdown_description(
            cwe="79",
            file="",
            line=0,
            excerpt="",
            body="Send <token> as @admin, see `<b>@x` and a@b.com.",
        )
        assert text == (
            "**Details**\n\n- Send \\<token> as `@admin`, see `<b>@x` and a@b.com."
        )

    def test_details_escape_gitlab_references(self):
        text = ft.markdown_description(
            cwe="79", file="", line=0, excerpt="", body="See !3 and grp/proj#12."
        )
        assert text == "**Details**\n\n- See \\!3 and grp/proj\\#12."


def test_description_prefers_the_given_impact():
    text = ft.markdown_description(
        cwe="862", file="", line=0, excerpt="", body="", impact="Mine."
    )
    assert text == "**What**\n\nMine."


@pytest.mark.parametrize(
    "impact,expected",
    [
        ("@admin see <token> # head", "`@admin` see \\<token> # head"),
        ("# Heading takes the account", "\\# Heading takes the account"),
        ("Fixed in #12 and ~bug", "Fixed in \\#12 and \\~bug"),
    ],
)
def test_description_escapes_the_model_impact_like_details(impact, expected):
    text = ft.markdown_description(
        cwe="862", file="", line=0, excerpt="", body="", impact=impact
    )
    assert text == f"**What**\n\n{expected}"
