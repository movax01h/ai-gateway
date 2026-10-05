"""Tests for the checks on the title and impact the model writes for a BL finding."""

import pytest

from duo_workflow_service.bl_security import model_text as mt


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "Checkout accepts another user's basket",
            "Checkout accepts another user's basket",
        ),
        ("CWE-639: Basket read by id.", "Basket read by id"),
        ("cwe639 - Basket read by id", "Basket read by id"),
        ("CWE 639: Basket read by id", "Basket read by id"),
        ("CWE639: Basket read by id", "Basket read by id"),
        ("Basket read by any user!", "Basket read by any user"),
        ("Basket read by any user\u2026", "Basket read by any user"),
        ("`GET /basket/:id` reads any basket", "GET /basket/:id reads any basket"),
        ("  Basket  read\tby id \\'x\\' ", "Basket read by id 'x'"),
        ("Two\nlines here ok", None),
        ("Short", None),
        ("x" * mt.TITLE_MAX, "x" * mt.TITLE_MAX),
        ("x" * (mt.TITLE_MAX + 1), None),
        (None, None),
        (7, None),
    ],
)
def test_model_title(raw, expected):
    assert mt.model_title(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Any user can read\n any basket.", "Any user can read any basket."),
        ("  ", None),
        ("x" * mt.MODEL_IMPACT_MAX, "x" * mt.MODEL_IMPACT_MAX),
        ("x" * (mt.MODEL_IMPACT_MAX + 1), None),
        (None, None),
    ],
)
def test_model_impact(raw, expected):
    assert mt.model_impact(raw) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("@admin see <token> # head", "`@admin` see \\<token> # head"),
        ("# Heading takes the account", "\\# Heading takes the account"),
        ("1. Numbered", "1\\. Numbered"),
        ("> Quoted", "\\> Quoted"),
        (
            "Send <token> as @admin, see `<b>@x` and a@b.com.",
            "Send \\<token> as `@admin`, see `<b>@x` and a@b.com.",
        ),
        ("Plain text stays.", "Plain text stays."),
        (
            'Closes #12, !3, %4, &5, ~bug and ~"two words".',
            'Closes \\#12, \\!3, \\%4, \\&5, \\~bug and \\~"two words".',
        ),
        ("#7 at the start", "\\#7 at the start"),
        ("See grp/proj#12 and grp/proj!3", "See grp/proj\\#12 and grp/proj\\!3"),
        (
            "C# code, a #hash, 100%, R&D, a~b and ~ alone.",
            "C# code, a #hash, 100%, R&D, a\\~b and ~ alone.",
        ),
        (
            "Kept: `#12 ~bug` and https://x.test/a#12?b=!3",
            "Kept: `#12 ~bug` and https://x.test/a#12?b=!3",
        ),
        (
            "A URL stops at a tag: https://x.test/<b>",
            "A URL stops at a tag: https://x.test/\\<b>",
        ),
    ],
)
def test_escape_markdown(text, expected):
    assert mt.escape_markdown(text) == expected
