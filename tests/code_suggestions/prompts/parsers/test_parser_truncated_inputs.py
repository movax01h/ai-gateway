# pylint: disable=file-naming-for-tests
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser

PARSE_TIMEOUT_SECONDS = 60

TRUNCATED_INPUTS = [
    ("TS", "[n=(=a}"),
    ("TS", "(st f=<T>()=>(e}"),
    ("TS", "r,{z:x=[=T)"),
    ("TS", "{:const=(=a]"),
    (
        "TS",
        "try {\n"
        '  console.log("foo");\n'
        "} catch (err) {\n"
        "  if (true) {\n"
        '    console.log("bar");\n'
        "  } else {\n"
        "    throw new MyError(a, b",
    ),
    ("LUA", "end"),
    ("LUA", "end\n"),
    ("LUA", "  end"),
    ("LUA", "x = foo(bar)\n"),
    ("LUA", "t[#t + 1] = v\n"),
    ("LUA", "foo:bar()\n"),
    ("LUA", "x"),
    ("LUA", "true"),
    ("LUA", "nil"),
    ("LUA", "lo"),
    ("LUA", "x = "),
    ("LUA", "{a"),
    ("LUA", "0x"),
]

PARSE_SCRIPT = """
import asyncio, json, sys
from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser

for lang, content in json.load(sys.stdin):
    print(f"PARSING {lang} {content!r}", flush=True)
    asyncio.run(CodeParser.from_language_id(content, LanguageId[lang]))
    print(f"PARSED {lang} {content!r}", flush=True)
"""


def test_truncated_inputs_parse_without_hanging_or_a_grammar_cache(tmp_path: Path):
    home_that_is_a_file = tmp_path / "home"
    home_that_is_a_file.write_text("")
    env = {"HOME": str(home_that_is_a_file), "XDG_CACHE_HOME": str(home_that_is_a_file)}
    try:
        with patch.dict("os.environ", env):
            result = subprocess.run(
                [sys.executable, "-c", PARSE_SCRIPT],
                input=json.dumps(TRUNCATED_INPUTS),
                capture_output=True,
                text=True,
                timeout=PARSE_TIMEOUT_SECONDS,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        last_line = (
            (exc.stdout or b"").decode(errors="ignore").strip().splitlines()[-1:]
        )
        pytest.fail(f"parser hung on {last_line}")

    assert result.returncode == 0, result.stderr
    parsed = [line for line in result.stdout.splitlines() if line.startswith("PARSED ")]
    assert len(parsed) == len(TRUNCATED_INPUTS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lang_id", "content"),
    [
        (LanguageId.CSHARP, "class A { void F() { } }\n"),
        (LanguageId.LUA, "local x = 1\n"),
        (LanguageId.TS, "const x: number = 1;\n"),
    ],
)
async def test_renamed_and_bundled_grammars_parse(lang_id: LanguageId, content: str):
    parser = await CodeParser.from_language_id(content, lang_id)
    assert parser.errors() == []
