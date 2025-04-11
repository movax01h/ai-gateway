import pytest

from ai_gateway.code_suggestions.processing import CodeContent
from ai_gateway.code_suggestions.processing.completions import (
    MetadataPromptBuilder,
    _PromptBuilder,
)
from ai_gateway.code_suggestions.processing.ops import LanguageId


@pytest.mark.parametrize(
    "lang_id,file_name,prefix,suffix,expected_prefix,expected_suffix",
    [
        (
            LanguageId.C,
            "test.c",
            "int main()",
            "}",
            "/* This code has a filename of test.c and is written in C. */\nint main()",
            "}",
        ),
        (
            LanguageId.PYTHON,
            "test.py",
            "def sort(arr):",
            "\n",
            "# This code has a filename of test.py and is written in Python.\ndef sort(arr):",
            "\n",
        ),
        (
            None,
            "App.vue",
            "<script setup>",
            "\n",
            "This code has a filename of App.vue\n<script setup>",
            "\n",
        ),
    ],
)
def test_completions_prompt_builder(
    lang_id: LanguageId,
    file_name: str,
    prefix: str,
    suffix: str,
    expected_prefix: str,
    expected_suffix: str,
):
    prompt_builder = _PromptBuilder(
        CodeContent(prefix, length_tokens=1),
        CodeContent(suffix, length_tokens=1),
        file_name,
        lang_id=lang_id,
    )

    prompt = prompt_builder.build()

    assert prompt.prefix == expected_prefix
    assert prompt.suffix == expected_suffix
    assert isinstance(prompt.metadata, MetadataPromptBuilder)
