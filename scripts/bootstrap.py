#!/usr/bin/env python

import tiktoken

from ai_gateway.tokenizer import init_tokenizer

if __name__ == "__main__":
    init_tokenizer()
    tiktoken.encoding_for_model("gpt-4o")
