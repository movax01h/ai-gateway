# pylint: disable=file-naming-for-tests
"""Tests for the window cut before redaction in ``DuoBaseTool._arun``.

A ``str`` result much longer than ``max_bytes`` is cut before redaction, so that redacting megabytes does not stall the
event loop. The margin is shrunk here so that short texts exercise the cut.

These tests live in the security test directory for the reason given in ``test_duo_base_tool_secret_redaction.py``: the
tool test directory replaces ``_arun``.
"""

import random
import re
import string
from typing import Any
from unittest.mock import patch

import pytest

import duo_workflow_service.tools.duo_base_tool as duo_base_tool_module
from duo_workflow_service.security.secret_redaction import REDACTED_PLACEHOLDER
from duo_workflow_service.tools import tool_output_manager
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.tool_output_manager import (
    TruncationConfig,
    TruncationDirection,
)

MAX_BYTES = 4000
TRUNCATED_SIZE = 3000
SLACK = MAX_BYTES - TRUNCATED_SIZE
MARGIN = 500
WINDOW = MAX_BYTES + MARGIN

# "glpat-" and 50 characters: the longest GitLab token the detector matches, 56 bytes, replaced by 10.
TOKEN = "glpat-" + "A" * 50

# A value cut from its closing quote, or from the keyword that follows it. The patterns need the rest of the match, so
# the window alone would leave ``hunter2`` raw.
SHAPES = {
    "unclosed": "password = {q}hunter2" + " ok" * 3000 + "{q}",
    "comparison": "if {q}hunter2{q}" + " " * 6000 + "== password:",
    "comparison_word_run": "if {q}hunter2{q} == " + "a" * 6000 + "password:",
}


def shaped(shape: str, quote: str, position: int) -> str:
    """``SHAPES[shape]`` with its value starting at about ``position``, and text after it."""
    return "x" * position + SHAPES[shape].format(q=quote) + "y" * 5000


class TextTool(DuoBaseTool):
    """Minimal tool whose ``_execute`` returns a caller-controlled string."""

    name: str = "text_tool"
    description: str = "Tool that returns a pre-set string for testing"
    _response: Any = None

    def set_response(self, response: Any) -> None:
        self._response = response

    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._response


@pytest.fixture(autouse=True)
def small_margin(monkeypatch):
    monkeypatch.setattr(tool_output_manager, "_REDACTION_WINDOW_MARGIN", MARGIN)


def run_config(direction: TruncationDirection):
    return TruncationConfig(
        max_bytes=MAX_BYTES, truncated_size=TRUNCATED_SIZE, direction=direction
    )


async def arun(
    text: str, direction: TruncationDirection, config: TruncationConfig | None = None
) -> str:
    tool = TextTool(metadata={}, truncation_config=config or run_config(direction))
    tool.set_response(text)
    return await tool._arun()


async def windowed_and_whole(
    monkeypatch,
    text: str,
    direction: TruncationDirection = TruncationDirection.FROM_START,
) -> tuple[str, str]:
    """The output with the window, then the output of redacting the whole result as before the window existed."""
    windowed = await arun(text, direction)
    monkeypatch.setattr(tool_output_manager, "_REDACTION_WINDOW_MARGIN", 10**9)
    return windowed, await arun(text, direction)


def kept(output: str) -> str:
    """The text the model sees between the tags of the truncation notice."""
    match = re.search(
        r"<truncated_tool_output>\n(.*)\n\s*</truncated_tool_output>",
        output,
        re.DOTALL,
    )
    assert match, output
    return match.group(1).strip()


def original_tokens(output: str) -> int:
    match = re.search(r"Original size: (\d+) tokens", output)
    assert match, output
    return int(match.group(1))


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(TruncationDirection))
@pytest.mark.parametrize("seed", range(5))
async def test_window_gives_the_output_of_redacting_everything(
    monkeypatch, direction, seed
):
    """Cutting before redaction leaves what the model sees unchanged.

    A few tokens at random places, some of them possibly split by the cut. Each is far shorter than the slack, and
    together they shrink the window by much less than the margin.
    """
    rng = random.Random(seed)
    lines = [f"line {i} ok" for i in range(3000)]
    for _ in range(8):
        secret = "".join(rng.choices(string.ascii_letters, k=20))
        lines[rng.randrange(len(lines))] += f" glpat-{secret}"
    text = "\n".join(lines)
    assert len(text) > WINDOW

    windowed, whole = await windowed_and_whole(monkeypatch, text, direction)

    assert kept(windowed) == kept(whole)
    assert "glpat" not in windowed


@pytest.mark.asyncio
async def test_notice_reports_the_size_of_the_whole_result(monkeypatch):
    text = "\n".join(f"line {i} ok" for i in range(30000))

    windowed, whole = await windowed_and_whole(monkeypatch, text)

    # Both are estimates sampled from the text, so allow some difference.
    assert 0.75 < original_tokens(windowed) / original_tokens(whole) < 1.25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction, text",
    [
        # The window is 2500 filler bytes, 35 tokens and 5 bytes of a 36th token.
        (
            TruncationDirection.FROM_START,
            "x" * 2500 + (TOKEN + " ") * 36 + "y" * 5000,
        ),
        # The window is 5 bytes of the first token, 35 tokens and 2500 filler bytes. A token must end at a space to match.
        (
            TruncationDirection.FROM_END,
            "y" * 5000 + TOKEN + (" " + TOKEN) * 35 + " " + "x" * 2499,
        ),
    ],
    ids=["from_start", "from_end"],
)
async def test_fragment_of_a_split_token_is_dropped(direction, text):
    """A shrunk window keeps less, so the unredacted fragment of the split token is not kept.

    Redacted, the window is 2890 bytes, under ``max_bytes``, so keeping all of it would keep the fragment too.
    """
    output = await arun(text, direction)

    assert kept(output) == "x" * (2890 - SLACK)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(TruncationDirection))
async def test_window_redacted_down_to_the_slack_keeps_only_the_notice(direction):
    text = (TOKEN + " ") * 100

    output = await arun(text, direction)

    assert "truncation_notice" in output
    assert kept(output) == ""
    assert "glpat" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, config",
    [
        # Up to the window: redacted whole, as before the window existed.
        ("x" * (WINDOW - 100) + " " + TOKEN, None),
        # No slack between max_bytes and truncated_size: the guarantee does not hold, so nothing is cut.
        (
            "x" * (WINDOW + 1000),
            TruncationConfig(max_bytes=TRUNCATED_SIZE, truncated_size=TRUNCATED_SIZE),
        ),
    ],
    ids=["up_to_the_window", "no_slack"],
)
async def test_result_is_redacted_whole(text, config):
    seen = []
    original = duo_base_tool_module.redact_secrets

    def spy(response, tool_name):
        seen.append(response)
        return original(response, tool_name)

    with patch.object(duo_base_tool_module, "redact_secrets", side_effect=spy):
        await arun(text, TruncationDirection.FROM_START, config)

    assert seen == [text]


@pytest.mark.asyncio
@pytest.mark.parametrize("repeats", [300, 400], ids=["whole", "windowed"])
async def test_special_token_in_the_result_does_not_stop_the_cut(repeats):
    """The token counter raised on text such as ``<|endoftext|>``, so a result was not cut or the call failed."""
    output = await arun("<|endoftext|> " * repeats, TruncationDirection.FROM_START)

    assert "truncation_notice" in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, cut",
    [
        ("ab\ud800cd", False),
        ("x" * (WINDOW - 100) + "\ud800", False),
        ("x" * (WINDOW + 3000) + "\ud800", True),
    ],
    ids=["short", "up_to_the_window", "past_the_window"],
)
async def test_lone_surrogate_in_the_result_does_not_fail_the_call(text, cut):
    """A lone surrogate cannot be encoded as UTF-8.

    Before the window the call passed, so a short result must too.
    """
    output = await arun(text, TruncationDirection.FROM_START)

    assert ("truncation_notice" in output) == cut


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("quote", ["'", '"', "`"])
@pytest.mark.parametrize("position", [0, 1500, 2900])
async def test_value_cut_from_what_completes_its_match_is_masked(
    monkeypatch, shape, quote, position
):
    """Redacting everything masks ``hunter2``, since what completes the match is in the text.

    The window lacks it, so the window is masked from the value to its end. The model sees the same text up to there.
    """
    text = shaped(shape, quote, position)

    windowed, whole = await windowed_and_whole(monkeypatch, text)

    assert "hunter2" not in windowed
    assert kept(windowed).endswith(f"{quote}{REDACTED_PLACEHOLDER}")
    assert kept(whole).startswith(kept(windowed))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, direction, masked",
    [
        # Nothing after the kept part is shown, so there is nothing to mask.
        (
            shaped("unclosed", "'", TRUNCATED_SIZE + 100),
            TruncationDirection.FROM_START,
            False,
        ),
        # A " ends a ' value, so a quote past the window cannot close it, as in the whole text.
        (
            "password = 'hunter2" + ' "' + " ok" * 3000 + "'" + "y" * 5000,
            TruncationDirection.FROM_START,
            False,
        ),
        (
            "password = `hunter2` " + "ok " * 3000,
            TruncationDirection.FROM_START,
            True,
        ),
        # The window is the end of the result, so no quote closes the value anywhere.
        (
            "z" * 6000 + "password = 'hunter2" + " ok" * 100,
            TruncationDirection.FROM_END,
            False,
        ),
        # The same, with the value far enough from the end to be in the part that is kept.
        (
            "z" * 6000 + "password = 'hunter2" + " ok" * 1000,
            TruncationDirection.FROM_END,
            False,
        ),
    ],
    ids=[
        "starts_past_the_kept_part",
        "ended_by_the_other_quote",
        "closed_in_the_window",
        "no_closing_quote_kept_from_the_end",
        "no_closing_quote_in_the_kept_part_from_the_end",
    ],
)
async def test_window_shows_what_redacting_everything_shows(
    monkeypatch, text, direction, masked
):
    windowed, whole = await windowed_and_whole(monkeypatch, text, direction)

    assert kept(windowed) == kept(whole)
    assert (REDACTED_PLACEHOLDER in windowed) == masked
