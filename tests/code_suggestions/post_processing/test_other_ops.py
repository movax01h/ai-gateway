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


@pytest.mark.parametrize(
    ("completion", "suffix", "expected_value"),
    [
        (
            '            retry_after = original_error.response.headers.get("retry-after")',
            '\n\n            retry_after = original_error.response.headers.get("retry-after")\n\n    return retry_after\n',
            "",
        ),
        (
            "provider = request.app.state.cloud_connector_auth_provider\n    return cloud_connector_ready(provider)",
            "\n    \n\n    provider = request.app.state.cloud_connector_auth_provider\n    return cloud_connector_ready(provider)\n",
            "",
        ),
        (
            "provider = request.app.state.cloud_connector_auth_provider\n    return cloud_connector_ready(provider",
            ")\n    \n\n    provider = request.app.state.cloud_connector_auth_provider\n    return cloud_connector_ready(provider)\n",
            "",
        ),
        (
            "res = await self.model.generate(\n    prompt.prefix,\n    stream,\n)\nelse:\n    res = other(\n        a, b\n    )\n\nif res",
            "\n    res = await self.model.generate(\n        prompt.prefix,\n        prompt.suffix,\n        stream,\n    )\nelif x:\n    pass\nelse:\n    res = other(\n        a, b\n    )\n\nif res:\n    return res\n",
            "",
        ),
        (
            "    return retry_after\n\n\ndef next_function():\n    pass",
            "\n    return retry_after\n\n\ndef unrelated():\n    pass\n",
            "    return retry_after\n\n\ndef next_function():\n    pass",
        ),
        (
            "    result = a + b\n    return result",
            "\nprint(add(1, 2))\n",
            "    result = a + b\n    return result",
        ),
        (")", ")\n    return x\n", ")"),
        ("}\n}", "}\n}\n}\n", "}\n}"),
        ("    return x", "\n    return x\n", "    return x"),
        (
            "    return None\n    else:",
            "\n    else:\n        return None\n",
            "    return None\n    else:",
        ),
        (
            "    return result",
            "def other():\n    pass\n    return result\n",
            "    return result",
        ),
        (
            "    result = a + b\n    return result",
            "",
            "    result = a + b\n    return result",
        ),
        ("", "\n    return x\n", ""),
        (
            "repeated_statement = compute_value(16)",
            "\n".join(f"line_{i} = compute_value({i})" for i in range(1, 16))
            + "\nrepeated_statement = compute_value(16)\n",
            "",
        ),
        (
            "repeated_statement = compute_value(17)",
            "\n".join(f"line_{i} = compute_value({i})" for i in range(1, 17))
            + "\nrepeated_statement = compute_value(17)\n",
            "repeated_statement = compute_value(17)",
        ),
    ],
)
def test_drop_suffix_repeat(completion: str, suffix: str, expected_value: str):
    assert ops.drop_suffix_repeat(completion, suffix) == expected_value
