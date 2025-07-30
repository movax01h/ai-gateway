from unittest.mock import Mock, patch

import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor as PostProcessorCompletions,
)
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessorOperation,
)


@pytest.fixture(name="mock_remove_comment_only_completion")
def mock_remove_comment_only_completion_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.remove_comment_only_completion"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_trim_by_min_allowed_context")
def mock_trim_by_min_allowed_context_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.trim_by_min_allowed_context"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_fix_end_block_errors")
def mock_fix_end_block_errors_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_end_block_errors"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_fix_end_block_errors_legacy")
def mock_fix_end_block_errors_legacy_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_end_block_errors_legacy"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_clean_model_reflection")
def mock_clean_model_reflection_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.clean_model_reflection"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_strip_whitespaces")
def mock_strip_whitespaces_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.strip_whitespaces"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_strip_asterisks")
def mock_strip_asterisks_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.strip_asterisks"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_filter_score")
def mock_filter_score_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.filter_score"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_fix_truncation")
def mock_fix_truncation_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.fix_truncation"
    ) as mock:
        mock.return_value = "processed completion"

        yield mock


@pytest.fixture(name="mock_clean_irrelevant_keywords")
def mock_clean_irrelevant_keywords_fixture():
    with patch(
        "ai_gateway.code_suggestions.processing.post.completions.clean_irrelevant_keywords"
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
        mock_clean_irrelevant_keywords: Mock,
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
        mock_clean_irrelevant_keywords.assert_called_once()

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
        mock_clean_irrelevant_keywords: Mock,
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
            score_threshold={"test-model": -1.0},
        )
        await post_processor.process(completion, model_name="test/test-model")

        mock_remove_comment_only_completion.assert_called_once()
        mock_trim_by_min_allowed_context.assert_called_once()
        mock_clean_irrelevant_keywords.assert_called_once()

        mock_fix_end_block_errors.assert_not_called()
        mock_fix_end_block_errors_legacy.assert_called_once()

        mock_clean_model_reflection.assert_called_once()
        mock_strip_whitespaces.assert_called_once()
        mock_strip_asterisks.assert_called_once()
        mock_filter_score.assert_called_once_with(
            "processed completion", score=None, threshold=-1.0
        )
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
        mock_clean_irrelevant_keywords: Mock,
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
        mock_clean_irrelevant_keywords.assert_called_once()

        mock_fix_end_block_errors.assert_called_once()
        mock_fix_end_block_errors_legacy.assert_not_called()

        mock_clean_model_reflection.assert_called_once()
        mock_strip_whitespaces.assert_not_called()
        mock_filter_score.assert_not_called()
        mock_fix_truncation.assert_not_called()
