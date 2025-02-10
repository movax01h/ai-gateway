from enum import StrEnum
from functools import partial
from inspect import iscoroutinefunction
from typing import Any, Callable, NewType, Optional

from ai_gateway.code_suggestions.processing.ops import strip_whitespaces
from ai_gateway.code_suggestions.processing.post.base import PostProcessorBase
from ai_gateway.code_suggestions.processing.post.ops import (
    SCORE_THRESHOLD_DISABLED,
    clean_model_reflection,
    filter_score,
    fix_end_block_errors,
    fix_end_block_errors_legacy,
    remove_comment_only_completion,
    strip_asterisks,
    trim_by_min_allowed_context,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId
from ai_gateway.structured_logging import get_request_logger

__all__ = [
    "PostProcessorOperation",
    "PostProcessor",
]

request_log = get_request_logger("suggestion_post_processing")

AliasOpsRecord = NewType("AliasOpsRecord", tuple[str, Callable[[str], str]])


class PostProcessorOperation(StrEnum):
    REMOVE_COMMENTS = "remove_comment_only_completion"
    TRIM_BY_MINIMUM_CONTEXT = "trim_by_min_allowed_context"
    FIX_END_BLOCK_ERRORS = "fix_end_block_errors"
    FIX_END_BLOCK_ERRORS_LEGACY = "fix_end_block_errors_legacy"
    CLEAN_MODEL_REFLECTION = "clean_model_reflection"
    STRIP_WHITESPACES = "strip_whitespaces"
    STRIP_ASTERISKS = "strip_asterisks"
    FILTER_SCORE = "filter_score"


# This is the ordered list of prost-processing functions
# Please do not change the order unless you have determined that it is acceptable
ORDERED_POST_PROCESSORS = [
    PostProcessorOperation.REMOVE_COMMENTS,
    PostProcessorOperation.TRIM_BY_MINIMUM_CONTEXT,
    PostProcessorOperation.FIX_END_BLOCK_ERRORS,
    PostProcessorOperation.CLEAN_MODEL_REFLECTION,
    PostProcessorOperation.STRIP_WHITESPACES,
]


class PostProcessor(PostProcessorBase):
    def __init__(
        self,
        code_context: str,
        lang_id: Optional[LanguageId] = None,
        suffix: Optional[str] = None,
        overrides: Optional[
            dict[PostProcessorOperation, PostProcessorOperation]
        ] = None,
        exclude: Optional[list] = None,
        extras: Optional[list] = None,
        score_threshold: Optional[float] = None,
    ):
        self.code_context = code_context
        self.lang_id = lang_id
        self.suffix = suffix if suffix else ""
        self.overrides = overrides if overrides else {}
        self.exclude = set(exclude) if exclude else []
        self.extras = extras if extras else []
        self.score_threshold = (
            score_threshold if score_threshold else SCORE_THRESHOLD_DISABLED
        )

    @property
    def ops(self) -> list[AliasOpsRecord]:
        return {
            PostProcessorOperation.FILTER_SCORE: partial(
                filter_score, threshold=self.score_threshold
            ),
            PostProcessorOperation.REMOVE_COMMENTS: partial(
                remove_comment_only_completion, lang_id=self.lang_id
            ),
            PostProcessorOperation.TRIM_BY_MINIMUM_CONTEXT: partial(
                trim_by_min_allowed_context, self.code_context, lang_id=self.lang_id
            ),
            PostProcessorOperation.FIX_END_BLOCK_ERRORS: partial(
                fix_end_block_errors,
                self.code_context,
                suffix=self.suffix,
                lang_id=self.lang_id,
            ),
            PostProcessorOperation.FIX_END_BLOCK_ERRORS_LEGACY: partial(
                fix_end_block_errors_legacy,
                self.code_context,
                suffix=self.suffix,
                lang_id=self.lang_id,
            ),
            PostProcessorOperation.CLEAN_MODEL_REFLECTION: partial(
                clean_model_reflection, self.code_context
            ),
            PostProcessorOperation.STRIP_WHITESPACES: strip_whitespaces,
            PostProcessorOperation.STRIP_ASTERISKS: strip_asterisks,
        }

    async def process(self, completion: str, **kwargs: Any) -> str:
        score = kwargs.get("score")

        for processor in self._ordered_post_processors():
            if str(processor) in self.exclude:
                continue

            completion = await self._apply_post_processor(
                processor, completion, score=score
            )

            if completion == "":
                return ""

        return completion

    def _ordered_post_processors(self):
        return ORDERED_POST_PROCESSORS + self.extras

    async def _apply_post_processor(self, processor_key, completion, score=None):
        # Override post-processor if present in `overrides`, else use the given processor
        actual_processor_key = self.overrides.get(processor_key, processor_key)
        func = self.ops[actual_processor_key]

        if actual_processor_key == PostProcessorOperation.FILTER_SCORE:
            func = partial(func, score=score)

        if self._is_async(func):
            processed_completion = await func(completion)
        else:
            processed_completion = func(completion)

        if processed_completion != completion:
            request_log.info(
                f"Post processor {actual_processor_key} modified completion with result {processed_completion}"
            )

        return processed_completion

    def _is_async(self, func):
        return iscoroutinefunction(func)
