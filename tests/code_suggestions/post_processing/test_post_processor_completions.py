from unittest.mock import Mock, patch

import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor as PostProcessorCompletions,
)
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessorOperation,
)


@pytest.fixture
def mock_remove_comment_only_completion():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.remove_comment_only_completion"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_trim_by_min_allowed_context():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.trim_by_min_allowed_context"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_fix_end_block_errors():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_end_block_errors"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_fix_end_block_errors_legacy():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_end_block_errors_legacy"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_clean_model_reflection():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.clean_model_reflection"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_strip_whitespaces():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.strip_whitespaces"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_strip_asterisks():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.strip_asterisks"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_filter_score():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.filter_score"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture
def mock_fix_truncation():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_truncation"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


class TestPostProcessorCompletions:
    @pytest.mark.asyncio
    async def test_process(
        self,
        mock_remove_comment_only_completion: Mock,
        mock_trim_by_min_allowed_context: Mock,
        mock_fix_end_block_errors: Mock,
        mock_fix_end_block_errors_legacy: Mock,
        mock_clean_model_reflection: Mock,
        mock_strip_whitespaces: Mock,
        mock_filter_score: Mock,
        mock_fix_truncation: Mock,
    ):
        code_context = "test code context"
        lang_id = LanguageId.RUBY
        suffix = "suffix"
        completion = "test completion"

        post_processor = PostProcessorCompletions(code_context, lang_id, suffix)
        await post_processor.process(completion)

        mock_remove_comment_only_completion.assert_called_once()
        mock_trim_by_min_allowed_context.assert_called_once()

        mock_fix_end_block_errors.assert_called_once()
        mock_fix_end_block_errors_legacy.assert_not_called()

        mock_clean_model_reflection.assert_called_once()
        mock_strip_whitespaces.assert_called_once()
        mock_filter_score.assert_not_called()
        mock_fix_truncation.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_with_custom_operations(
        self,
        mock_remove_comment_only_completion: Mock,
        mock_trim_by_min_allowed_context: Mock,
        mock_fix_end_block_errors: Mock,
        mock_fix_end_block_errors_legacy: Mock,
        mock_clean_model_reflection: Mock,
        mock_strip_whitespaces: Mock,
        mock_strip_asterisks: Mock,
        mock_filter_score: Mock,
        mock_fix_truncation: Mock,
    ):
        code_context = "test code context"
        lang_id = LanguageId.RUBY
        suffix = "suffix"
        completion = "test completion"

        post_processor = PostProcessorCompletions(
            code_context,
            lang_id,
            suffix,
            overrides={
                PostProcessorOperation.FIX_END_BLOCK_ERRORS: PostProcessorOperation.FIX_END_BLOCK_ERRORS_LEGACY,
            },
            extras=[
                PostProcessorOperation.STRIP_ASTERISKS,
                PostProcessorOperation.FILTER_SCORE,
                PostProcessorOperation.FIX_TRUNCATION,
            ],
        )
        await post_processor.process(completion)

        mock_remove_comment_only_completion.assert_called_once()
        mock_trim_by_min_allowed_context.assert_called_once()

        mock_fix_end_block_errors.assert_not_called()
        mock_fix_end_block_errors_legacy.assert_called_once()

        mock_clean_model_reflection.assert_called_once()
        mock_strip_whitespaces.assert_called_once()
        mock_strip_asterisks.assert_called_once()
        mock_filter_score.assert_called_once()
        mock_fix_truncation.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_exclusions(
        self,
        mock_remove_comment_only_completion: Mock,
        mock_trim_by_min_allowed_context: Mock,
        mock_fix_end_block_errors: Mock,
        mock_fix_end_block_errors_legacy: Mock,
        mock_clean_model_reflection: Mock,
        mock_strip_whitespaces: Mock,
        mock_filter_score: Mock,
        mock_fix_truncation: Mock,
    ):
        code_context = "test code context"
        lang_id = LanguageId.RUBY
        suffix = "suffix"
        completion = "test completion"

        post_processor = PostProcessorCompletions(
            code_context, lang_id, suffix, exclude=["strip_whitespaces"]
        )
        await post_processor.process(completion)

        mock_remove_comment_only_completion.assert_called_once()
        mock_trim_by_min_allowed_context.assert_called_once()

        mock_fix_end_block_errors.assert_called_once()
        mock_fix_end_block_errors_legacy.assert_not_called()

        mock_clean_model_reflection.assert_called_once()
        mock_strip_whitespaces.assert_not_called()
        mock_filter_score.assert_not_called()
        mock_fix_truncation.assert_not_called()
