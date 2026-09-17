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
        self,
        text: str,
        max_length: int,
        truncation_side: str = "left",
        line_boundary: bool = False,
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
        length_tokens = len(tokens["input_ids"])

        if line_boundary and length_tokens == max_length:
            if truncation_side == "left":
                decoded = decoded.partition("\n")[2] or decoded
            else:
                decoded = decoded.rpartition("\n")[0] or decoded
            length_tokens = self.estimate_length(decoded)[0]

        return CodeContent(text=decoded, length_tokens=length_tokens)

    @override
    def estimate_length(self, text: str | list[str]) -> list[int]:
        return self.tokenizer(
            text,
            return_length=True,
            return_attention_mask=False,
            add_special_tokens=False,
        )["length"]
