# pylint: disable=file-naming-for-tests
from typing import Optional
from unittest.mock import patch

import pytest
from tree_sitter_language_pack import SupportedLanguage, get_parser

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser
from ai_gateway.code_suggestions.prompts.parsers.base import BaseVisitor
from ai_gateway.code_suggestions.prompts.parsers.treesitter import SUPPORTED_GRAMMARS


class _StubVisitor(BaseVisitor):
    def _visit_node(self, node):
        pass


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, ""),
        (b"", ""),
        (b"hello", "hello"),
        (b"\xc3\x28", "("),
    ],
)
def test_bytes_to_str(data: Optional[bytes], expected: str):
    assert _StubVisitor()._bytes_to_str(data) == expected


@pytest.mark.parametrize("lang_id", [None, LanguageId.COBOL])
@pytest.mark.asyncio
async def test_unsupported_languages(lang_id: LanguageId):
    with pytest.raises(ValueError):
        await CodeParser.from_language_id("import Foundation", lang_id)


@pytest.mark.asyncio
async def test_missing_grammar_raises_value_error():
    with patch(
        "ai_gateway.code_suggestions.prompts.parsers.treesitter.get_parser",
        side_effect=LookupError("missing"),
    ):
        with pytest.raises(ValueError):
            await CodeParser.from_language_id("x", LanguageId.PYTHON)


@pytest.mark.parametrize("grammar_name", sorted(SUPPORTED_GRAMMARS))
def test_supported_grammars_are_bundled(grammar_name: SupportedLanguage):
    assert get_parser(grammar_name).parse(b"x").root_node is not None


@pytest.mark.parametrize("lang_id", list(LanguageId))
@pytest.mark.asyncio
async def test_every_language_parses_or_raises_value_error(lang_id: LanguageId):
    try:
        parser = await CodeParser.from_language_id("x", lang_id)
    except ValueError:
        return

    assert isinstance(parser, CodeParser)


@pytest.mark.asyncio
async def test_non_utf8():
    value = b"\xc3\x28"  # Invalid UTF-8 byte sequence

    with pytest.raises(ValueError):
        await CodeParser.from_language_id(value, LanguageId.JS)
