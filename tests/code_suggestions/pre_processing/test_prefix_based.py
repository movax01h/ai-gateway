from typing import Optional, Union

import pytest
from transformers import AutoTokenizer

from ai_gateway.code_suggestions.processing import (
    MetadataCodeContent,
    MetadataExtraInfo,
    MetadataPromptBuilder,
    Prompt,
)
from ai_gateway.code_suggestions.processing.pre import (
    PromptBuilderPrefixBased,
    TokenizerTokenStrategy,
)
from ai_gateway.code_suggestions.prompts import PromptTemplate
from ai_gateway.models.base_chat import Message, Role

# This template takes 4 tokens (ignore placeholders)
# with the Salesforce/codegen2-16B tokenizer
_TEST_TEMPLATE_1 = """
start
{prefix}
end
""".strip(
    "\n"
)

# This template takes 4 tokens (ignore placeholders)
# with the Salesforce/codegen2-16B tokenizer
_TEST_TEMPLATE_2 = """
start {lang}
{prefix}
end
""".strip(
    "\n"
)


class TestPromptBuilderPrefixBased:
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-16B")

    @pytest.mark.parametrize(
        ("prefix", "total_max_len", "expected_prompt"),
        [
            (
                "random_text",
                2048,
                Prompt(
                    prefix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                1,
                Prompt(
                    prefix="text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
            (
                ["random_text", "random_another_text"],
                4,
                Prompt(
                    prefix="random_text\ntext",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=16, length_tokens=4)
                        }
                    ),
                ),
            ),
        ],
    )
    def test_without_template(
        self, prefix: Union[str, list[str]], total_max_len: int, expected_prompt: Prompt
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(prefix)
        else:
            builder.add_content(*prefix)

        actual = builder.build()

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        ("prefix", "total_max_len", "tpl", "tpl_args", "expected_prompt"),
        [
            (
                "random_text",
                2048,
                _TEST_TEMPLATE_1,
                dict(),
                Prompt(
                    prefix="start\nrandom_text\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                5,
                _TEST_TEMPLATE_1,
                dict(),
                Prompt(
                    prefix="start\ntext\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                2048,
                _TEST_TEMPLATE_2,
                dict(lang="python"),
                Prompt(
                    prefix="start python\nrandom_text\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                "random_text",
                6,
                _TEST_TEMPLATE_2,
                dict(lang="python"),
                Prompt(
                    prefix="start python\ntext\nend",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=4, length_tokens=1)
                        }
                    ),
                ),
            ),
        ],
    )
    def test_with_template(
        self,
        prefix: Union[str, list[str]],
        total_max_len: int,
        tpl: str,
        tpl_args: dict,
        expected_prompt: Prompt,
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(prefix)
        else:
            builder.add_content(*prefix)

        builder.add_template(PromptTemplate(tpl), **tpl_args)
        actual = builder.build()

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        ("prompt", "total_max_len", "ignore_exception", "expected_prompt"),
        [
            (
                "random_text",
                2048,
                False,
                Prompt(
                    prefix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prompt": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            ("random_text", 1, False, None),  # catch the exception raised
            (
                "random_text",
                1,
                True,
                Prompt(
                    prefix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prompt": MetadataCodeContent(length=11, length_tokens=3)
                        }
                    ),
                ),
            ),
            (
                [
                    Message(role=Role.SYSTEM, content="random_text"),
                    Message(role=Role.USER, content="random_another_text"),
                ],
                1,
                True,
                Prompt(
                    prefix=[
                        Message(role=Role.SYSTEM, content="random_text"),
                        Message(role=Role.USER, content="random_another_text"),
                    ],
                    metadata=MetadataPromptBuilder(
                        components={
                            "prompt": MetadataCodeContent(length=30, length_tokens=8)
                        }
                    ),
                ),
            ),
        ],
    )
    def test_with_prompt_wrapped(
        self,
        prompt: str,
        total_max_len: int,
        ignore_exception: bool,
        expected_prompt: Optional[Prompt],
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if expected_prompt or ignore_exception:
            actual = builder.wrap(prompt, ignore_exception=True)
            assert actual == expected_prompt
        else:
            with pytest.raises(ValueError):
                builder.wrap(prompt)

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "suffix_reserved_percent",
            "total_max_len",
            "expected_prompt",
        ),
        [
            (
                "random_text",
                "random_text",
                0.5,
                2048,
                Prompt(
                    prefix="random_text",
                    suffix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=11, length_tokens=3),
                        }
                    ),
                ),
            ),
            (
                "random_text",
                "random_text",
                0.25,
                4,
                Prompt(
                    prefix="random_text",
                    suffix="random",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=6, length_tokens=1),
                        }
                    ),
                ),
            ),
            (
                ["random_text", "random_another_text"],
                "random_text",
                0.25,
                4,
                Prompt(
                    prefix="random_text",
                    suffix="random",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=6, length_tokens=1),
                        }
                    ),
                ),
            ),
        ],
    )
    def test_with_suffix(
        self,
        prefix: Union[str, list[str]],
        suffix: str,
        suffix_reserved_percent: float,
        total_max_len: int,
        expected_prompt: Prompt,
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(
                prefix, suffix=suffix, suffix_reserved_percent=suffix_reserved_percent
            )
        else:
            builder.add_content(
                *prefix, suffix=suffix, suffix_reserved_percent=suffix_reserved_percent
            )

        actual = builder.build()

        assert actual == expected_prompt

    @pytest.mark.parametrize(
        (
            "prefix",
            "suffix",
            "code_context",
            "suffix_reserved_percent",
            "context_max_percent",
            "total_max_len",
            "expected_prompt",
        ),
        [
            (
                "prefix_text",
                "random_text",
                ["context_text"],
                0.5,
                1.0,
                2048,
                Prompt(
                    prefix="context_text\nprefix_text",
                    suffix="random_text",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=11, length_tokens=3),
                        },
                        code_context=MetadataExtraInfo(
                            name="code_context",
                            pre=MetadataCodeContent(length=12, length_tokens=3),
                            post=MetadataCodeContent(length=12, length_tokens=3),
                        ),
                    ),
                ),
            ),
            (
                "prefix_text",
                "random_text",
                ["context_text"],
                0.25,
                1.0,
                4,
                Prompt(
                    prefix="prefix_text",
                    suffix="random",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=6, length_tokens=1),
                        },
                        code_context=MetadataExtraInfo(
                            name="code_context",
                            pre=MetadataCodeContent(length=12, length_tokens=3),
                            post=MetadataCodeContent(length=0, length_tokens=0),
                        ),
                    ),
                ),
            ),
            (
                "prefix_text",
                "random_text",
                ["context_text"],
                0.25,
                0.175,
                6,
                Prompt(
                    prefix="context\nprefix_text",
                    suffix="random",
                    metadata=MetadataPromptBuilder(
                        components={
                            "prefix": MetadataCodeContent(length=11, length_tokens=3),
                            "suffix": MetadataCodeContent(length=6, length_tokens=1),
                        },
                        code_context=MetadataExtraInfo(
                            name="code_context",
                            pre=MetadataCodeContent(length=12, length_tokens=3),
                            post=MetadataCodeContent(length=7, length_tokens=1),
                        ),
                    ),
                ),
            ),
        ],
    )
    def test_with_code_context(
        self,
        prefix: Union[str, list[str]],
        suffix: str,
        code_context: list[str],
        suffix_reserved_percent: float,
        context_max_percent: float,
        total_max_len: int,
        expected_prompt: Prompt,
    ):
        builder = PromptBuilderPrefixBased(
            total_max_len, TokenizerTokenStrategy(self.tokenizer)
        )

        if isinstance(prefix, str):
            builder.add_content(
                prefix,
                suffix=suffix,
                suffix_reserved_percent=suffix_reserved_percent,
                context_max_percent=context_max_percent,
                code_context=code_context,
            )
        else:
            builder.add_content(
                *prefix,
                suffix=suffix,
                suffix_reserved_percent=suffix_reserved_percent,
                context_max_percent=context_max_percent,
                code_context=code_context,
            )

        actual = builder.build()

        assert actual == expected_prompt
