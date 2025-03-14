import re
from collections import Counter
from typing import Any, Optional

import structlog

from ai_gateway.code_suggestions.processing.ops import (
    find_common_lines,
    find_cursor_position,
    find_newline_position,
    find_non_whitespace_point,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser

__all__ = [
    "clean_model_reflection",
    "trim_by_min_allowed_context",
    "fix_end_block_errors",
    "fix_end_block_errors_legacy",
    "strip_code_block_markdown",
    "prepend_new_line",
    "strip_asterisks",
]

log = structlog.stdlib.get_logger("codesuggestions")

SCORE_THRESHOLD_DISABLED = -9999.0

_COMMENT_IDENTIFIERS = ["/*", "//", "#"]
_SPECIAL_CHARS = "()[];.,$%&^*@#!{}/"
_RE_MARKDOWN_CODE_BLOCK_BEGIN = re.compile(r"^`{3}\S*\n", flags=re.MULTILINE)
_RE_LEADING_ASTERISKS = r"^\s*\*{5,}"
_IRRELEVANT_KEYWORDS = ["<|cursor|>"]


async def clean_model_reflection(context: str, completion: str, **kwargs: Any) -> str:
    def _is_single_line_comment(lines: list[str]):
        return len(lines) == 1 and lines[0].lstrip().startswith(
            tuple(_COMMENT_IDENTIFIERS)
        )

    def _with_special_characters(counter: Counter, min_p: float):
        special_characters_count = sum(counter.get(c, 0) for c in _SPECIAL_CHARS)
        total_count = sum(counter.values())

        return (special_characters_count / total_count) >= min_p

    def _with_low_diversity(counter: Counter, min_p: float):
        unique_count = len(counter)
        total_count = sum(counter.values())

        return (unique_count / total_count) >= min_p

    def _is_large_group(
        group: tuple,
        lines: list[str],
        min_block_size: int = 5,
        min_special_chars: float = 0.25,
        min_diversity_chars: float = 0.35,
    ):
        counter = Counter("".join(line.strip() for line in lines))
        total_count = sum(counter.values())

        return (
            len(group) >= min_block_size
            and total_count > 0
            and not _with_special_characters(counter, min_special_chars)
            and not _with_low_diversity(counter, min_diversity_chars)
        )

    text = f"{context}{completion}"

    br_pos = find_newline_position(text, start_index=len(context))
    if br_pos == -1:
        # Only the current line was completed, no need to dedupe completion
        return completion

    lines_before = _split_code_lines(text[:br_pos])
    lines_after = _split_code_lines(text[br_pos:])

    common_lines = find_common_lines(
        source=[line.strip() for line in lines_before],
        target=[line.strip() for line in lines_after],
    )

    prev_line = 0
    lines_completion = []
    for group in common_lines:
        start_line, end_line = group[0], group[-1]
        target_lines = lines_after[start_line : end_line + 1]
        lines_completion.extend(lines_after[prev_line:start_line])

        if not (
            _is_single_line_comment(target_lines)
            or _is_large_group(group, target_lines, **kwargs)
        ):
            # Add appropriate lines to the final completion
            # and ignore other lines
            lines_completion.extend(target_lines)

        prev_line = end_line + 1

    # Add remaining lines to the completion list
    lines_completion.extend(lines_after[prev_line:])

    # Get the completion of the current line + processed lines
    completion = text[len(context) : br_pos]
    completion = "".join([completion, *lines_completion])

    return completion


# This trims the suggestion to the minimum allowed block, i.e.: the smallest block surrounding the cursor
# Introduced in https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/308
async def trim_by_min_allowed_context(
    prefix: str,
    completion: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    code_sample = f"{prefix}{completion}"
    len_prefix = len(prefix)
    target_point = find_non_whitespace_point(code_sample, start_index=len_prefix)
    if target_point == (-1, -1):
        return completion

    try:
        parser = await CodeParser.from_language_id(
            code_sample,
            lang_id,
        )
        context = parser.min_allowed_context(target_point)
        end_pos = find_cursor_position(code_sample, context.end)
        if end_pos == -1:
            return completion

        out = code_sample[len_prefix:end_pos]
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")
        out = completion

    return out


async def fix_end_block_errors_legacy(
    prefix: str,
    completion: str,
    suffix: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    """
    Strips suffix from completion only if the resulting code has zero parsing errors.

    This processor is more conservative in its approach. It will only strip the suffix
    if the resulting code has absolutely no parsing errors. If the original code had
    any parsing errors to begin with, the suffix will not be stripped.

    Args:
        prefix: The code context before the completion
        completion: The code completion to process
        suffix: The code context after the completion
        lang_id: Optional language identifier for the code

    Returns:
        str: The processed completion with suffix potentially stripped if no errors exist.
    """
    # Hypothesis 1: the suffix contains only one line.
    suffix_first_line = suffix.strip()
    if len(suffix_first_line) == 0:
        return completion

    # Hypothesis 2: the suffix contains more than only one line.
    idx_suffix_new_line = suffix_first_line.find("\n")
    if idx_suffix_new_line != -1:
        # Hypothesis confirmed: keep only the first line within the variable.
        suffix_first_line = suffix_first_line[:idx_suffix_new_line]

    completion_lookup = completion.rstrip()
    if not completion_lookup.endswith(suffix_first_line):
        # Return the original copy of the completion.
        return completion

    try:
        # Remove the suffix from the completion.
        completion_lookup = completion_lookup[: -len(suffix_first_line)]
        # Check if any errors exists when joining the original suffix
        # and the updated version of the completion.
        code_sample = f"{prefix}{completion_lookup}{suffix}"
        parser = await CodeParser.from_language_id(code_sample, lang_id)
        if len(parser.errors()) == 0:
            completion = completion_lookup
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion


async def fix_end_block_errors(
    prefix: str,
    completion: str,
    suffix: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    """
    Strips suffix from completion if it doesn't introduce new parsing errors.

    This processor takes a more lenient approach compared to fix_end_block_errors_legacy.
    It will strip the suffix even if the original code had parsing errors, as long as
    stripping the suffix doesn't introduce additional errors compared to the original code.
    This makes it more effective at handling code that may have pre-existing issues.

    This processor will search through the completion for the suffix, if found it will stop
    at each instance of the suffix to check if the completion before here reduces the amount of errors.

    Args:
        prefix: The code context before the completion
        completion: The code completion to process
        suffix: The code context after the completion
        lang_id: Optional language identifier for the code

    Returns:
        str: The processed completion with suffix potentially stripped if no new errors are introduced.
    """
    stripped_suffix = suffix.rstrip()
    if len(stripped_suffix) == 0:
        return completion

    # Hypothesis 1: the suffix contains only one line.
    suffix_first_line = stripped_suffix

    # Hypothesis 2: the suffix contains more than one line; this overrides Hypothesis 1
    idx_suffix_new_line = suffix_first_line.strip().find("\n")
    if idx_suffix_new_line != -1:
        # Hypothesis confirmed: keep only the first line within the variable.
        suffix_first_line = suffix_first_line[:idx_suffix_new_line]

    completion_lookup = completion

    # See if suffix exists in completion
    if completion_lookup.find(suffix_first_line) == -1:
        # Return the original copy of the completion.
        return completion

    try:
        # Check for errors in the original code
        code_sample_before_suggestion = f"{prefix}{suffix}"
        parser_before_suggestion = await CodeParser.from_language_id(
            code_sample_before_suggestion, lang_id
        )
        before_errors = list(
            filter(
                lambda e: find_cursor_position(code_sample_before_suggestion, e.start)
                < len(code_sample_before_suggestion),
                parser_before_suggestion.errors(),
            )
        )
        errors_before_suggestion = len(before_errors)

        # Start at last suffix existing in completion, trim everything after
        # and see if it improves errors
        least_error_count = 9999
        while (last_suffix_pos := completion_lookup.rfind(suffix_first_line)) != -1:
            completion_lookup = completion_lookup[:last_suffix_pos].rstrip()

            # Check if there are any new errors when inserting the code suggestion
            code_sample_after_suggestion = f"{prefix}{completion_lookup}{suffix}"
            parser_after_suggestion = await CodeParser.from_language_id(
                code_sample_after_suggestion, lang_id
            )
            after_errors = list(
                filter(
                    lambda e: find_cursor_position(
                        code_sample_after_suggestion, e.start
                    )
                    < len(prefix),
                    parser_after_suggestion.errors(),
                )
            )
            errors_after_suggestion = len(after_errors)

            if (
                errors_after_suggestion <= errors_before_suggestion
                and errors_after_suggestion <= least_error_count
            ):
                least_error_count = errors_after_suggestion
                completion = completion_lookup
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion


async def fix_truncation(
    prefix: str,
    completion: str,
    suffix: str,
    max_output_tokens_used: bool,
    raw_completion: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    """
    Trims back a truncated completion to a more sensible stopping point if it does not
    introduce new parsing errors.

    This process trims back to the last space of the last line. If the last line does
    not contain any non-leading and non-trailing spaces, it trims off the entire last
    line unless doing so would remove all the code content.

    Args:
        prefix: The code context before the completion
        completion: The code completion to process
        suffix: The code context after the completion
        max_output_tokens_used: True if the completion consists of the
                                max number of output tokens permitted.
        raw_completion: The completion before any post processing
        lang_id: Optional language identifier for the code

    Returns:
        str: The processed completion
    """

    # We assume the completion is likely truncated if it:
    # 1. uses exactly max output tokens,
    # 2. is unmodified by previous post processors, and
    # 3. does not end with space or newline characters.
    def _is_likely_truncated() -> bool:
        return (
            max_output_tokens_used
            and completion == raw_completion
            and completion == completion.rstrip()
        )

    if not _is_likely_truncated():
        return completion

    last_line = completion.splitlines()[-1:][0]
    last_space_index = last_line.rfind(" ")
    string_to_remove = (
        last_line[last_space_index:] if last_space_index != -1 else last_line
    )

    trimmed_completion = completion.removesuffix(string_to_remove).rstrip()

    # Return the unmodified completion if trimming it would remove all code
    # content (e.g. when the completion is only one line without spaces).
    if not trimmed_completion.strip():
        return completion

    code_before_trim = f"{prefix}{completion}{suffix}"
    code_after_trim = f"{prefix}{trimmed_completion}{suffix}"

    try:
        parser_before_trim = await CodeParser.from_language_id(
            code_before_trim, lang_id
        )
        parser_after_trim = await CodeParser.from_language_id(code_after_trim, lang_id)

        if len(parser_after_trim.errors()) <= len(parser_before_trim.errors()):
            return trimmed_completion
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion


def strip_code_block_markdown(text: str) -> str:
    text = _RE_MARKDOWN_CODE_BLOCK_BEGIN.sub("", text, count=0)
    text = text.rstrip("`")

    return text


def prepend_new_line(code_context: str, completion: str) -> str:
    if (
        len(completion)
        and not code_context.endswith("\n")
        and not completion.startswith("\n")
    ):
        completion = "\n" + completion

    return completion


def _split_code_lines(s: str) -> list[str]:
    lines_split = s.splitlines(keepends=True)
    lines_processed = []

    for i, line in enumerate(lines_split):
        line = line.rstrip("\n")
        if i > 0:
            line = "\n" + line

        lines_processed.append(line)

    if len(lines_split) and lines_split[-1].endswith("\n"):
        lines_processed.append("\n")

    return lines_processed


# If the completion contains only comments, we should not return anything
async def remove_comment_only_completion(
    completion: str,
    lang_id: Optional[LanguageId] = None,
) -> str:
    if not completion:
        return completion
    try:
        parser = await CodeParser.from_language_id(
            completion,
            lang_id,
        )
        if parser.comments_only():
            log.info("removing comments-only completion")
            return ""
    except ValueError as e:
        log.warning(f"Failed to parse code: {e}")

    return completion


# This trims leading asterisks in the suggestion
# https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1470
def strip_asterisks(completion: str) -> str:
    # search first part of completion for a string of asterisks
    # if there is a match, return an empty completion
    if re.search(_RE_LEADING_ASTERISKS, completion):
        return ""

    # else, return the original completion
    # if there is no match for a string of asterisks, no need to clean the completion
    return completion


# This function removes irrelevant keywords from completions
# https://gitlab.com/gitlab-org/gitlab/-/issues/517027
def clean_irrelevant_keywords(completions: str) -> str:
    pattern = "|".join(map(re.escape, _IRRELEVANT_KEYWORDS))
    return re.sub(pattern, "", completions)


# Very simple filtering based on score
def filter_score(completion: str, score: float, threshold: float) -> str:
    if (
        threshold != SCORE_THRESHOLD_DISABLED
        and isinstance(score, (int, float))
        and score < threshold
    ):
        return ""
    return completion
