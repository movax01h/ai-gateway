"""Patch AutoTokenizer.from_pretrained before test modules are imported.

test_prefix_based.py and test_tokens.py instantiate the tokenizer as class-level
attributes, which executes at import/collection time. HF_HUB_OFFLINE=1 prevents
network calls, but fails when the tokenizer is not in the local cache.

This module-level patch intercepts from_pretrained before those class bodies are
evaluated and returns a mock that replicates Salesforce/codegen2-16B behavior for
the exact strings used in these tests.

Token vocabulary (matches the real tokenizer for all test inputs):
    "random"  -> [1]  decode -> "random"
    "_"       -> [2]  decode -> "_"
    "text"    -> [3]  decode -> "text"
    "another" -> [4]  decode -> "another"
    "context" -> [5]  decode -> "context"
    "prefix"  -> [6]  decode -> "prefix"
    "start"   -> [7]  decode -> "start"
    "\\n"     -> [8]  decode -> "\\n"
    "end"     -> [9]  decode -> "end"
    " python" -> [10] decode -> " python"

Resulting token counts that match test assertions:
    "random_text"                    -> 3 tokens
    "random_another_text"            -> 5 tokens
    "random_textrandom_another_text" -> 8 tokens (concatenated messages)
    "context_text"                   -> 3 tokens (first token -> "context")
    "prefix_text"                    -> 3 tokens
    "start\\n\\nend"                 -> 4 tokens
    "start python\\n\\nend"          -> 5 tokens
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Token vocabulary - sorted longest-first so greedy matching is correct.
# ---------------------------------------------------------------------------
_VOCAB: dict[str, int] = {
    " python": 10,
    "another": 4,
    "context": 5,
    "prefix": 6,
    "random": 1,
    "start": 7,
    "text": 3,
    "end": 9,
    "_": 2,
    "\n": 8,
}
_VOCAB_SORTED = sorted(_VOCAB.keys(), key=len, reverse=True)
_DECODE: dict[int, str] = {v: k for k, v in _VOCAB.items()}

_UNKNOWN_BASE = 1000  # unknown chars get id = _UNKNOWN_BASE + ord(char)


def _encode(text: str) -> list[int]:
    """Greedy longest-match tokenisation for test inputs."""
    ids: list[int] = []
    i = 0
    while i < len(text):
        for word in _VOCAB_SORTED:
            if text[i : i + len(word)] == word:
                ids.append(_VOCAB[word])
                i += len(word)
                break
        else:
            ids.append(_UNKNOWN_BASE + ord(text[i]))
            i += 1
    return ids


def _decode(ids: list[int]) -> str:
    return "".join(_DECODE[i] if i in _DECODE else chr(i - _UNKNOWN_BASE) for i in ids)


def _create_mock_tokenizer() -> MagicMock:
    mock = MagicMock()
    mock.truncation_side = "left"

    def _call(
        text,
        max_length=None,
        truncation=False,
        return_length=False,
        **_kwargs,
    ):
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        all_ids: list[list[int]] = []
        for t in texts:
            ids = _encode(t)
            if truncation and max_length is not None:
                if max_length == 0:
                    ids = []
                elif mock.truncation_side == "left":
                    ids = ids[-max_length:]
                else:
                    ids = ids[:max_length]
            all_ids.append(ids)

        result: dict = {
            "input_ids": all_ids if is_batch else all_ids[0],
        }
        if return_length:
            result["length"] = [len(ids) for ids in all_ids]
        return result

    mock.side_effect = _call
    mock.decode.side_effect = lambda ids, **_kw: _decode(ids)
    return mock


# Patch before any test module in this directory is imported.
# patch.start() without a corresponding stop() is intentional here: the patch
# must be active during collection (class-body execution) and must remain active
# for the full test run. pytest tears down the process afterwards anyway.
_mock_tokenizer = _create_mock_tokenizer()
_hf_patch = patch(
    "transformers.AutoTokenizer.from_pretrained", return_value=_mock_tokenizer
)
_hf_patch.start()
