# pylint: disable=file-naming-for-tests
from typing import Optional

import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser
from ai_gateway.code_suggestions.prompts.parsers.base import BaseVisitor


class _StubVisitor(BaseVisitor):
    def _visit_node(self, node):
        pass


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, ""),
        (b"", ""),
        (b"hello", "hello"),
        (b"\xc3\x28", "("),  # invalid UTF-8 is decoded with errors="ignore"
    ],
)
def test_bytes_to_str(data: Optional[bytes], expected: str):
    assert _StubVisitor()._bytes_to_str(data) == expected


@pytest.mark.parametrize("lang_id", [None])
@pytest.mark.asyncio
async def test_unsupported_languages(lang_id: LanguageId):
    with pytest.raises(ValueError):
        await CodeParser.from_language_id("import Foundation", lang_id)


@pytest.mark.asyncio
async def test_non_utf8():
    value = b"\xc3\x28"  # Invalid UTF-8 byte sequence

    with pytest.raises(ValueError):
        await CodeParser.from_language_id(value, LanguageId.JS)
