# flake8: noqa: W605
"""Security functions for emoji detection and removal."""

import re
from typing import Any, Dict, List, Union

from duo_workflow_service.security.markdown_content_security import _apply_recursively

_EMOJI_SURROGATE_PATTERN = re.compile(
    r"\\ud([89a-fA-F][0-9a-fA-F]{2})\s*\\ud([c-fC-F][0-9a-fA-F]{2})"
)
_UNICODE_4_DIGIT_PATTERN = re.compile(r"\\u([0-9A-Fa-f]{4})")
_UNICODE_8_DIGIT_PATTERN = re.compile(r"\\U([0-9A-Fa-f]{8})")
_UNICODE_EMOJI_ESCAPE_PATTERN = re.compile(r"\s*\\+u1[fF][3-6][0-9a-fA-F]{2}\s*")
_EMOJI_MAIN_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e6-\U0001f1ff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\U0001f004\U0001f0cf"
    "\U0001f170-\U0001f251"
    "\U00002190-\U000021ff"
    "\U000024c2-\U0001f251"
    "\U0001f000-\U0001f02f"
    "\U0001f030-\U0001f093"
    "\U0000fe00-\U0000fe0f"
    "\U0000e000-\U0000f8ff"
    "]+",
    flags=re.UNICODE,
)
_SKIN_TONE_PATTERN = re.compile(r"[\U0001F3FB-\U0001F3FF]+")
_WHITESPACE_CLEANUP_PATTERN = re.compile(r"\s{2,}")
_NEWLINE_CLEANUP_PATTERN = re.compile(r"\n\s*\n")

_COMMON_EMOJI_PATTERNS = [
    re.compile(r"\\ud83d\\ude[0-4][0-9a-f]", re.IGNORECASE),
    re.compile(r"\\ud83c\\udf[0-9a-f]{2}", re.IGNORECASE),
    re.compile(r"\\ud83d\\udc[0-9a-f]{2}", re.IGNORECASE),
    re.compile(r"\\ud83d\\udd[0-9a-f]{2}", re.IGNORECASE),
    re.compile(r"\\ud83c\\udd[0-9a-f]{2}", re.IGNORECASE),
    re.compile(r"\\ud83d\\udea[0-9a-f]", re.IGNORECASE),
    re.compile(r"\\ud83c\\udff[c-f]", re.IGNORECASE),
]

_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
_ZERO_WIDTH_ESCAPED = {
    char: char.encode("unicode_escape").decode("ascii") for char in _ZERO_WIDTH_CHARS
}
_ASCII_ONLY_PATTERN = re.compile(r"[^\x00-\x7F]+")


def strip_emojis(
    response: Union[str, Dict[str, Any], List[Any]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Strip emoji characters from response data.

    Args:
        response: The response data to process

    Returns:
        Response with emojis removed, compatible with ToolMessage.content
    """

    def _strip_emojis_from_string(text: str) -> str:
        if not text:
            return text

        # First, remove JSON-escaped emojis with optional whitespace between surrogate pairs
        text = _EMOJI_SURROGATE_PATTERN.sub("", text)

        # Remove common emoji patterns that are JSON-escaped (no space between pairs)
        for pattern in _COMMON_EMOJI_PATTERNS:
            text = pattern.sub("", text)

        # Remove JSON-escaped emoji patterns (e.g., \u1f600)
        text = _UNICODE_EMOJI_ESCAPE_PATTERN.sub("", text)

        if any(char in text for char in _ZERO_WIDTH_CHARS):
            for char in _ZERO_WIDTH_CHARS:
                text = text.replace(char, "")
                text = text.replace(_ZERO_WIDTH_ESCAPED[char], "")

        # Remove actual emoji characters from the text
        text = _EMOJI_MAIN_PATTERN.sub("", text)
        text = _SKIN_TONE_PATTERN.sub("", text)

        # Clean up excessive newlines and whitespace
        text = _NEWLINE_CLEANUP_PATTERN.sub("\n", text)
        text = _WHITESPACE_CLEANUP_PATTERN.sub(" ", text)
        text = text.strip()

        try:
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            text = "".join(char for char in text if ord(char) < 128)

        return text

    processed = _apply_recursively(response, _strip_emojis_from_string)

    # Wrap dict in list for ToolMessage compatibility
    if isinstance(processed, dict):
        return [processed]

    return processed
