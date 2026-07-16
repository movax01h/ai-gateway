from typing import cast, override

from transformers import PreTrainedTokenizerBase

from ai_gateway.code_suggestions.processing.typing import CodeContent, TokenStrategyBase

__all__ = [
    "TokenizerTokenStrategy",
]


class TokenizerTokenStrategy(TokenStrategyBase):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    @override
    def truncate_content(
        self, text: str, max_length: int, truncation_side: str = "left"
    ) -> CodeContent:
        self.tokenizer.truncation_side = truncation_side

        tokens = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        # Decoding a single sequence returns a `str`; `decode` is typed as
        # `str | list[str]` to also cover batch decoding.
        decoded = cast(str, self.tokenizer.decode(tokens["input_ids"]))

        return CodeContent(
            text=decoded,
            length_tokens=len(tokens["input_ids"]),
        )

    @override
    def estimate_length(self, text: str | list[str]) -> list[int]:
        return self.tokenizer(
            text,
            return_length=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["length"]
