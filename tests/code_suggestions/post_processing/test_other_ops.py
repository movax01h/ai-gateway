# pylint: disable=file-naming-for-tests
import pytest

from ai_gateway.code_suggestions.processing.post import ops
from ai_gateway.code_suggestions.processing.post.ops import (
    remove_comment_only_completion,
)
from ai_gateway.code_suggestions.processing.typing import LanguageId


@pytest.mark.parametrize(
    ("text", "expected_value"),
    [
        ("first line\nsecond line", "first line\nsecond line"),
        ("```\nfirst line\nsecond line```", "first line\nsecond line"),
        ("```unk\nfirst line\nsecond line```", "first line\nsecond line"),
        ("```unk\nfirst line\nsecond line", "first line\nsecond line"),
        ("\nfirst line\nsecond line```", "\nfirst line\nsecond line"),
        ("```python\none line```", "one line"),
        ("```TypeScript\none line```", "one line"),
        ("```java_script\none line```", "one line"),
        ("```unknown_lang_123\none line```", "one line"),
    ],
)
def test_strip_code_block_markdown(text: str, expected_value: str):
    actual_value = ops.strip_code_block_markdown(text)

    assert actual_value == expected_value


@pytest.mark.parametrize(
    ("code_context", "completion", "expected_value"),
    [
        ("", "", ""),
        ("code context", "", ""),
        ("code context", "\ncompletion", "\ncompletion"),
        ("code context", "completion", "\ncompletion"),
        ("code context\n", "completion", "completion"),
    ],
)
def test_prepend_new_line(code_context: str, completion: str, expected_value: str):
    actual_value = ops.prepend_new_line(code_context, completion)

    assert actual_value == expected_value


@pytest.mark.parametrize(
    ("completion", "lang_id", "expected"),
    [
        (
            'if __name__=="__main__":\n\tprint(f"Hello world!")',
            LanguageId.PYTHON,
            'if __name__=="__main__":\n\tprint(f"Hello world!")',
        ),
        (
            "# This function prints 'hello'\ndef hello():\n\tprint('hello')\n",
            LanguageId.PYTHON,
            "# This function prints 'hello'\ndef hello():\n\tprint('hello')\n",
        ),
        (
            "# This is just a comment\n# followed by another comment",
            LanguageId.PYTHON,
            "",
        ),
        (
            "this is a line being completed')\n\n",
            LanguageId.PYTHON,
            "this is a line being completed')\n\n",
        ),
    ],
)
@pytest.mark.asyncio
async def test_remove_comment_only_completion(
    completion: str, lang_id: LanguageId, expected: str
):
    actual = await remove_comment_only_completion(completion, lang_id)

    assert actual == expected


@pytest.mark.parametrize(
    ("text", "expected_value"),
    [
        ("first line\nsecond line", "first line\nsecond line"),
        ("```python\nreturn a + b\n```", "return a + b"),
        (
            "Here is the code:\n```python\nreturn a + b\n```\nThis adds numbers.",
            "return a + b",
        ),
        ("```python\nreturn a + b", "return a + b"),
        ("```python\nreturn x\n``", "return x"),
        ("Here it is:\r\n```python\r\nreturn a\r\n```", "return a"),
        ("```\nreturn a + b\n```", "return a + b"),
        ("prose only, no code", "prose only, no code"),
        ("const s = ```\nnot a fence`;\nmore", "const s = ```\nnot a fence`;\nmore"),
        (
            "l1\nl2\nl3\n```python\nlate fence\n```",
            "l1\nl2\nl3\n```python\nlate fence\n```",
        ),
        ("```python\nfirst\n```\n```python\nsecond\n```", "first"),
        ("```c#\nvar x = 1;\n```", "var x = 1;"),
        ("```{lang}\ncode\n```", "```{lang}\ncode\n```"),
        ("```python \nreturn a\n```", "return a"),
        ("```js\nconst t = `a`;\n`b` + c\n```", "const t = `a`;\n`b` + c"),
    ],
)
def test_extract_fenced_code(text: str, expected_value: str):
    actual_value = ops.extract_fenced_code(text)

    assert actual_value == expected_value
