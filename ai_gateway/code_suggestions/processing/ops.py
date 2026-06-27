import re
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
from tree_sitter import Node

from ai_gateway.code_suggestions.processing.typing import LanguageId

__all__ = [
    "compare_exact",
    "find_common_lines",
    "find_cursor_position",
    "find_newline_position",
    "find_non_whitespace_point",
    "prepend_lang_id",
    "remove_incomplete_block",
    "remove_incomplete_lines",
    "strip_whitespaces",
    "trim_by_max_len",
    "trim_by_sep",
]


class _LanguageDef(NamedTuple):
    lang_id: LanguageId
    grammar_name: str
    human_name: str
    extensions: frozenset[str]
    editor_names: frozenset[str]


_ALL_LANGS = {
    _LanguageDef(LanguageId.C, "c", "C", frozenset({"c", "h"}), frozenset({"c"})),
    _LanguageDef(
        LanguageId.CPP,
        "cpp",
        "C++",
        frozenset({"cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H"}),
        frozenset({"cpp"}),
    ),
    _LanguageDef(
        LanguageId.CSHARP, "csharp", "C#", frozenset({"cs"}), frozenset({"csharp"})
    ),
    _LanguageDef(LanguageId.GO, "go", "Go", frozenset({"go"}), frozenset({"go"})),
    _LanguageDef(
        LanguageId.JAVA,
        "java",
        "Java",
        frozenset({"java"}),
        frozenset({"java"}),
    ),
    _LanguageDef(
        LanguageId.JS,
        "javascript",
        "JavaScript",
        frozenset({"js", "jsx"}),
        frozenset({"javascript", "javascriptreact"}),
    ),
    _LanguageDef(
        LanguageId.PHP,
        "php",
        "PHP",
        frozenset({"php", "php3", "php4", "php5", "phps", "phpt"}),
        frozenset({"php"}),
    ),
    _LanguageDef(
        LanguageId.PYTHON, "python", "Python", frozenset({"py"}), frozenset({"python"})
    ),
    _LanguageDef(
        LanguageId.RUBY, "ruby", "Ruby", frozenset({"rb"}), frozenset({"ruby"})
    ),
    _LanguageDef(
        LanguageId.RUST, "rust", "Rust", frozenset({"rs"}), frozenset({"rust"})
    ),
    _LanguageDef(
        LanguageId.SCALA, "scala", "Scala", frozenset({"scala"}), frozenset({"scala"})
    ),
    _LanguageDef(
        LanguageId.TS,
        "typescript",
        "TypeScript",
        frozenset({"ts", "tsx"}),
        frozenset({"typescript", "typescriptreact"}),
    ),
    _LanguageDef(
        LanguageId.KOTLIN,
        "kotlin",
        "Kotlin",
        frozenset({"kts", "kt"}),
        frozenset({"kotlin"}),
    ),
    _LanguageDef(
        LanguageId.ADA,
        "ada",
        "Ada",
        frozenset({"adb", "ads", "ada"}),
        frozenset({"ada"}),
    ),
    _LanguageDef(
        LanguageId.APEX,
        "apex",
        "Apex",
        frozenset({"cls", "trigger"}),
        frozenset({"apex"}),
    ),
    _LanguageDef(
        LanguageId.ARDUINO,
        "arduino",
        "Arduino",
        frozenset({"ino"}),
        frozenset({"arduino"}),
    ),
    _LanguageDef(
        LanguageId.ASM,
        "asm",
        "Assembly",
        frozenset({"asm", "s", "S"}),
        frozenset({"asm", "x86asm"}),
    ),
    _LanguageDef(LanguageId.AWK, "awk", "AWK", frozenset({"awk"}), frozenset({"awk"})),
    _LanguageDef(
        LanguageId.BASH,
        "bash",
        "Bash",
        frozenset({"sh", "bash"}),
        frozenset({"shellscript", "bash"}),
    ),
    _LanguageDef(
        LanguageId.BATCH,
        "batch",
        "Batch",
        frozenset({"bat", "cmd"}),
        frozenset({"bat"}),
    ),
    _LanguageDef(
        LanguageId.BICEP, "bicep", "Bicep", frozenset({"bicep"}), frozenset({"bicep"})
    ),
    _LanguageDef(LanguageId.C3, "c3", "C3", frozenset({"c3"}), frozenset({"c3"})),
    _LanguageDef(
        LanguageId.CAIRO, "cairo", "Cairo", frozenset({"cairo"}), frozenset({"cairo"})
    ),
    _LanguageDef(
        LanguageId.CLOJURE,
        "clojure",
        "Clojure",
        frozenset({"clj", "cljs", "cljc"}),
        frozenset({"clojure"}),
    ),
    _LanguageDef(
        LanguageId.COBOL,
        "cobol",
        "COBOL",
        frozenset({"cbl", "cob", "cobol"}),
        frozenset({"cobol"}),
    ),
    _LanguageDef(
        LanguageId.COMMONLISP,
        "commonlisp",
        "Common Lisp",
        frozenset({"lisp", "cl", "lsp"}),
        frozenset({"commonlisp", "lisp"}),
    ),
    _LanguageDef(
        LanguageId.CRYSTAL,
        "crystal",
        "Crystal",
        frozenset({"cr"}),
        frozenset({"crystal"}),
    ),
    _LanguageDef(
        LanguageId.CUDA,
        "cuda",
        "CUDA",
        frozenset({"cu", "cuh"}),
        frozenset({"cuda", "cuda-cpp"}),
    ),
    _LanguageDef(LanguageId.D, "d", "D", frozenset({"d"}), frozenset({"d"})),
    _LanguageDef(
        LanguageId.DART, "dart", "Dart", frozenset({"dart"}), frozenset({"dart"})
    ),
    _LanguageDef(
        LanguageId.DEVICETREE,
        "devicetree",
        "DeviceTree",
        frozenset({"dts", "dtsi"}),
        frozenset({"dts"}),
    ),
    _LanguageDef(
        LanguageId.DOCKERFILE,
        "dockerfile",
        "Dockerfile",
        frozenset({"dockerfile"}),
        frozenset({"dockerfile"}),
    ),
    _LanguageDef(
        LanguageId.ELISP,
        "elisp",
        "Emacs Lisp",
        frozenset({"el"}),
        frozenset({"emacs-lisp"}),
    ),
    _LanguageDef(
        LanguageId.ELIXIR,
        "elixir",
        "Elixir",
        frozenset({"ex", "exs"}),
        frozenset({"elixir"}),
    ),
    _LanguageDef(LanguageId.ELM, "elm", "Elm", frozenset({"elm"}), frozenset({"elm"})),
    _LanguageDef(
        LanguageId.ERLANG,
        "erlang",
        "Erlang",
        frozenset({"erl", "hrl"}),
        frozenset({"erlang"}),
    ),
    _LanguageDef(
        LanguageId.FENNEL, "fennel", "Fennel", frozenset({"fnl"}), frozenset({"fennel"})
    ),
    _LanguageDef(
        LanguageId.FISH, "fish", "Fish", frozenset({"fish"}), frozenset({"fish"})
    ),
    _LanguageDef(
        LanguageId.FORTRAN,
        "fortran",
        "Fortran",
        frozenset({"f", "f90", "f95", "f03", "f08", "f77", "for"}),
        frozenset({"fortran", "FortranFreeForm"}),
    ),
    _LanguageDef(
        LanguageId.FSHARP,
        "fsharp",
        "F#",
        frozenset({"fs", "fsi", "fsx"}),
        frozenset({"fsharp"}),
    ),
    _LanguageDef(
        LanguageId.GDSCRIPT,
        "gdscript",
        "GDScript",
        frozenset({"gd"}),
        frozenset({"gdscript"}),
    ),
    _LanguageDef(
        LanguageId.GLEAM, "gleam", "Gleam", frozenset({"gleam"}), frozenset({"gleam"})
    ),
    _LanguageDef(
        LanguageId.GLSL,
        "glsl",
        "GLSL",
        frozenset({"glsl", "vert", "frag"}),
        frozenset({"glsl"}),
    ),
    _LanguageDef(
        LanguageId.GROOVY,
        "groovy",
        "Groovy",
        frozenset({"groovy", "gvy", "gy", "gsh"}),
        frozenset({"groovy"}),
    ),
    # `.hh` is already claimed by C++ headers; keep Hack to its modern `.hack` extension.
    _LanguageDef(
        LanguageId.HACK, "hack", "Hack", frozenset({"hack"}), frozenset({"hack"})
    ),
    _LanguageDef(
        LanguageId.HARE, "hare", "Hare", frozenset({"ha"}), frozenset({"hare"})
    ),
    _LanguageDef(
        LanguageId.HASKELL,
        "haskell",
        "Haskell",
        frozenset({"hs", "lhs"}),
        frozenset({"haskell"}),
    ),
    _LanguageDef(
        LanguageId.HAXE, "haxe", "Haxe", frozenset({"hx"}), frozenset({"haxe"})
    ),
    _LanguageDef(LanguageId.HCL, "hcl", "HCL", frozenset({"hcl"}), frozenset({"hcl"})),
    _LanguageDef(
        LanguageId.HLSL, "hlsl", "HLSL", frozenset({"hlsl"}), frozenset({"hlsl"})
    ),
    _LanguageDef(
        LanguageId.IDRIS, "idris", "Idris", frozenset({"idr"}), frozenset({"idris"})
    ),
    _LanguageDef(LanguageId.JAI, "jai", "Jai", frozenset({"jai"}), frozenset({"jai"})),
    _LanguageDef(
        LanguageId.JANET, "janet", "Janet", frozenset({"janet"}), frozenset({"janet"})
    ),
    _LanguageDef(LanguageId.JQ, "jq", "jq", frozenset({"jq"}), frozenset({"jq"})),
    _LanguageDef(
        LanguageId.JULIA, "julia", "Julia", frozenset({"jl"}), frozenset({"julia"})
    ),
    _LanguageDef(LanguageId.LUA, "lua", "Lua", frozenset({"lua"}), frozenset({"lua"})),
    _LanguageDef(
        LanguageId.LUAU, "luau", "Luau", frozenset({"luau"}), frozenset({"luau"})
    ),
    _LanguageDef(
        LanguageId.MAGIK, "magik", "Magik", frozenset({"magik"}), frozenset({"magik"})
    ),
    _LanguageDef(
        LanguageId.MATLAB,
        "matlab",
        "MATLAB",
        frozenset({"mat"}),
        frozenset({"matlab"}),
    ),
    _LanguageDef(
        LanguageId.MOJO, "mojo", "Mojo", frozenset({"mojo"}), frozenset({"mojo"})
    ),
    _LanguageDef(
        LanguageId.MOVE, "move", "Move", frozenset({"move"}), frozenset({"move"})
    ),
    _LanguageDef(LanguageId.NIM, "nim", "Nim", frozenset({"nim"}), frozenset({"nim"})),
    _LanguageDef(LanguageId.NIX, "nix", "Nix", frozenset({"nix"}), frozenset({"nix"})),
    _LanguageDef(
        LanguageId.OBJC,
        "objc",
        "Objective-C",
        frozenset({"m", "mm"}),
        frozenset({"objective-c", "objective-cpp"}),
    ),
    _LanguageDef(
        LanguageId.OCAML,
        "ocaml",
        "OCaml",
        frozenset({"ml", "mli"}),
        frozenset({"ocaml"}),
    ),
    _LanguageDef(
        LanguageId.ODIN, "odin", "Odin", frozenset({"odin"}), frozenset({"odin"})
    ),
    _LanguageDef(
        LanguageId.PASCAL,
        "pascal",
        "Pascal",
        frozenset({"pas", "pp"}),
        frozenset({"pascal", "objectpascal"}),
    ),
    _LanguageDef(
        LanguageId.PERL,
        "perl",
        "Perl",
        frozenset({"pl", "pm", "pod", "t"}),
        frozenset({"perl"}),
    ),
    _LanguageDef(
        LanguageId.PONY, "pony", "Pony", frozenset({"pony"}), frozenset({"pony"})
    ),
    _LanguageDef(
        LanguageId.POWERSHELL,
        "powershell",
        "PowerShell",
        frozenset({"ps1", "psm1", "psd1"}),
        frozenset({"powershell"}),
    ),
    _LanguageDef(
        LanguageId.PROLOG,
        "prolog",
        "Prolog",
        frozenset({"pro", "prolog"}),
        frozenset({"prolog"}),
    ),
    _LanguageDef(
        LanguageId.PURESCRIPT,
        "purescript",
        "PureScript",
        frozenset({"purs"}),
        frozenset({"purescript"}),
    ),
    _LanguageDef(
        LanguageId.QL,
        "ql",
        "CodeQL",
        frozenset({"ql", "qll"}),
        frozenset({"ql", "codeql"}),
    ),
    _LanguageDef(LanguageId.R, "r", "R", frozenset({"r", "R"}), frozenset({"r"})),
    _LanguageDef(
        LanguageId.RACKET, "racket", "Racket", frozenset({"rkt"}), frozenset({"racket"})
    ),
    _LanguageDef(
        LanguageId.RESCRIPT,
        "rescript",
        "ReScript",
        frozenset({"res", "resi"}),
        frozenset({"rescript"}),
    ),
    _LanguageDef(
        LanguageId.ROBOT,
        "robot",
        "Robot Framework",
        frozenset({"robot"}),
        frozenset({"robot"}),
    ),
    _LanguageDef(LanguageId.ROC, "roc", "Roc", frozenset({"roc"}), frozenset({"roc"})),
    _LanguageDef(
        LanguageId.SCHEME,
        "scheme",
        "Scheme",
        frozenset({"scm", "ss"}),
        frozenset({"scheme"}),
    ),
    _LanguageDef(
        LanguageId.SMALLTALK,
        "smalltalk",
        "Smalltalk",
        frozenset({"st"}),
        frozenset({"smalltalk"}),
    ),
    _LanguageDef(
        LanguageId.SOLIDITY,
        "solidity",
        "Solidity",
        frozenset({"sol"}),
        frozenset({"solidity"}),
    ),
    _LanguageDef(
        LanguageId.SOURCEPAWN,
        "sourcepawn",
        "SourcePawn",
        frozenset({"sp"}),
        frozenset({"sourcepawn"}),
    ),
    _LanguageDef(LanguageId.SQL, "sql", "SQL", frozenset({"sql"}), frozenset({"sql"})),
    _LanguageDef(
        LanguageId.STARLARK,
        "starlark",
        "Starlark",
        frozenset({"star", "bzl"}),
        frozenset({"starlark", "bazel"}),
    ),
    _LanguageDef(
        LanguageId.SVELTE,
        "svelte",
        "Svelte",
        frozenset({"svelte"}),
        frozenset({"svelte"}),
    ),
    _LanguageDef(
        LanguageId.SWIFT, "swift", "Swift", frozenset({"swift"}), frozenset({"swift"})
    ),
    _LanguageDef(
        LanguageId.SYSTEMVERILOG,
        "systemverilog",
        "SystemVerilog",
        frozenset({"sv", "svh"}),
        frozenset({"systemverilog"}),
    ),
    _LanguageDef(
        LanguageId.TACT, "tact", "Tact", frozenset({"tact"}), frozenset({"tact"})
    ),
    _LanguageDef(
        LanguageId.TCL, "tcl", "Tcl", frozenset({"tcl", "tk"}), frozenset({"tcl"})
    ),
    _LanguageDef(
        LanguageId.TERRAFORM,
        "terraform",
        "Terraform",
        frozenset({"tf", "tfvars"}),
        frozenset({"terraform"}),
    ),
    _LanguageDef(
        LanguageId.TLAPLUS,
        "tlaplus",
        "TLA+",
        frozenset({"tla"}),
        frozenset({"tlaplus"}),
    ),
    _LanguageDef(LanguageId.V, "v", "V", frozenset({"vsh", "vv"}), frozenset({"v"})),
    _LanguageDef(
        LanguageId.VB,
        "vb",
        "Visual Basic",
        frozenset({"vb", "bas", "vbs"}),
        frozenset({"vb", "vbscript"}),
    ),
    _LanguageDef(
        LanguageId.VERILOG,
        "verilog",
        "Verilog",
        frozenset({"v"}),
        frozenset({"verilog"}),
    ),
    _LanguageDef(
        LanguageId.VHDL, "vhdl", "VHDL", frozenset({"vhd", "vhdl"}), frozenset({"vhdl"})
    ),
    _LanguageDef(
        LanguageId.VIM,
        "vim",
        "Vimscript",
        frozenset({"vim"}),
        frozenset({"vim", "viml"}),
    ),
    _LanguageDef(LanguageId.VUE, "vue", "Vue", frozenset({"vue"}), frozenset({"vue"})),
    _LanguageDef(
        LanguageId.WGSL, "wgsl", "WGSL", frozenset({"wgsl"}), frozenset({"wgsl"})
    ),
    _LanguageDef(LanguageId.ZIG, "zig", "Zig", frozenset({"zig"}), frozenset({"zig"})),
    _LanguageDef(LanguageId.ZSH, "zsh", "Zsh", frozenset({"zsh"}), frozenset({"zsh"})),
}

_LANG_ID_TO_LANG_DEF = {value.lang_id: value for value in _ALL_LANGS}

_EXTENSION_TO_LANG_ID = {
    ext: language.lang_id for language in _ALL_LANGS for ext in language.extensions
}

_EDITOR_LANG_TO_LANG_ID = {
    name: language.lang_id for language in _ALL_LANGS for name in language.editor_names
}

_EXTENSION_TO_LANG_NAME = {
    ext: language.grammar_name for language in _ALL_LANGS for ext in language.extensions
}

# A new line with a non-indented letter or comment (/*, #, //)
_END_OF_CODE_BLOCK_REGEX = re.compile(r"\n([a-zA-Z]|(\/\*)|(#)|(\/\/))")

# The maximum percentage of the text that can be trimmed to remove an incomplete code block
_MAX_CODE_BLOCK_TRIM_PERCENT = 0.1


class ProgramLanguage:
    def __init__(self, lang_id: LanguageId):
        self._lang_id = lang_id
        self._lang_def = _LANG_ID_TO_LANG_DEF.get(lang_id)

    def __getattr__(self, name):
        return getattr(self._lang_def, name)

    @classmethod
    def from_language_id(cls, lang_id: LanguageId):
        return ProgramLanguage(lang_id)


def prepend_lang_id(s: str, lang_id: Optional[LanguageId]):
    if lang_id:
        lang = lang_id.name.lower()
        s = f"<{lang}>{s}"

    return s


def remove_incomplete_lines(s: str, sep: str = "\n") -> str:
    if (index := s.rfind(sep)) > 0:
        return s[:index]

    return s


def remove_incomplete_block(
    s: str, max_trim_percent: float = _MAX_CODE_BLOCK_TRIM_PERCENT
) -> str:
    end_of_block = _END_OF_CODE_BLOCK_REGEX.search(
        s, endpos=int(len(s) * max_trim_percent)
    )
    if end_of_block:
        index = end_of_block.start()
        return s[index + 1 :]

    return s


def trim_by_max_len(s: str, max_context_size: int) -> str:
    if max_context_size < 1:
        raise ValueError("expected `max_context_size` greater or equal to 1")
    return s[-max_context_size:]


def trim_by_sep(s: str, sep: str = "```") -> str:
    if (index := s.find(sep)) != -1:
        return s[:index]

    return s


def lang_from_filename(file_name: Union[str, Path]) -> Optional[LanguageId]:
    ext = Path(file_name).suffix.replace(".", "")
    return _EXTENSION_TO_LANG_ID.get(ext, None)


def lang_name_from_filename(file_name: Union[str, Path]) -> Optional[str]:
    ext = Path(file_name).suffix.replace(".", "")
    return _EXTENSION_TO_LANG_NAME.get(ext, None)


def lang_from_editor_lang(editor_lang: str) -> Optional[LanguageId]:
    return _EDITOR_LANG_TO_LANG_ID.get(editor_lang, None)


def find_non_whitespace_point(value: str, start_index: int = 0) -> tuple[int, int]:
    row = 0
    col = 0

    found_row = -1
    found_col = -1

    for idx, c in enumerate(value):
        if c == "\n":
            # increase the row counter and reset the column one
            row += 1
            col = 0
            continue

        if idx >= start_index and not c.isspace():
            found_row = row
            found_col = col
            break

        col += 1

    return found_row, found_col


def find_newline_position(value: str, start_index: int = 0) -> int:
    """Finds the nearest newline position close to `start_index`"""
    substring = value[:start_index]
    substring_rstrip = substring.rstrip(" \t")

    if substring_rstrip.endswith("\n"):
        return len(substring)

    for idx, c in enumerate(value[start_index:]):
        if c == "\n":
            return len(substring) + idx + 1

    return -1


def compare_exact(a: str, b: str) -> bool:
    return a == b


def find_common_lines(
    source: list[str],
    target: list[str],
    comparison_func: Callable[[str, str], bool] = compare_exact,
) -> list[tuple[Any, ...]]:
    # editorconfig-checker-disable
    """Finds the common strings between two lists, keeping track of repeated ranges.

    Example:
    ----------
    >>> source = ["abc", "def", "g"]
    >>> target = ["abc", "def", "c", "abc"]
    >>> find_common_lines(source, target)
        [(0,1), (3,)]

    The return indicates that target[0] and target[1] match some lines in source and are
    consecutive, target[3] matches but is not consecutive.

    Method:
    ----------
    1. Construct a matrix of size len(source) x len(target) to store the longest common
       subsequence (LCS) lengths.
    2. Fill this matrix by iterating through source and target and checking if the strings at each
       index match using the `comparison_func`.
    3. If they match, update the LCS length by taking the diagonal value and adding 1.
    4. After the matrix is filled, take the max value along each column to find the matching
       indices in target.
    5. Filter out 0s and collect the actual indices in target that match source.
    6. To group consecutive matches, computes the diff between indices and splits them into groups
       wherever the diff is not 1 (i.e. not consecutive).

    :param source: A list of strings to which we compare the target
    :param target: A list of strings we compare against the source
    :param comparison_func: The function used to compare two strings. Defaults to an exact match.

    :return: A list of indices of common strings grouped if they are consecutive lines
    """
    # editorconfig-checker-enable

    len_source = len(source)
    len_target = len(target)

    # The 0th row and column always contain zero values to simplify
    # the LCS algorithm implementation
    l_matrix: npt.NDArray[np.int_] = np.zeros(
        (len_source + 1, len_target + 1), dtype=int
    )

    # Tabulated implementation for the LCS problem.
    # Complexity: O(len_source*len_target)
    # Goal: find all common lines and their sequences to collect them into groups later
    for i in range(len_source + 1):
        for j in range(len_target + 1):
            if i == 0 or j == 0:
                l_matrix[i, j] = 0
            elif comparison_func(source[i - 1], target[j - 1]):
                # Optimization: start groups of size larger than `1` with `2`, otherwise start with `1`
                # Goal: when getting the maximum over the rows, we need to take larger groups into account first
                prev_match = l_matrix[i - 1, j - 1]
                l_matrix[i - 1, j - 1] = (
                    prev_match + 1 if prev_match == 1 else prev_match
                )

                # The LCS step according to the tabulated implementation
                l_matrix[i, j] = l_matrix[i - 1, j - 1] + 1
            else:
                l_matrix[i, j] = 0

    # Get the line numbers with the max value, the length of the 1D array equals to `len_source+1`
    target_max = l_matrix.argmax(axis=0)

    # Collect only those lines that match `source`.
    # Note: since we padded the L matrix with zeros, we need to trim the array when getting the indices
    target_lines = target_max > 0
    target_matches = target_max[target_lines]
    target_lines_idx = np.where(target_lines[1:])[0]

    if len(target_lines_idx) == 0:
        return []

    # Group common lines
    # Groups of size larger than `1` always contain consecutive lines
    # E.g.:
    # Input: [0,4,5,6,7]
    # Output: [(0,), (4,5,6), (7,)]
    diff_matches = np.diff(target_matches)
    splits = np.split(target_lines_idx, np.where(diff_matches != 1)[0] + 1)

    return [tuple(g) for g in splits]


def split_on_point(
    source_code: str, point: tuple[int, int]
) -> tuple[Optional[str], Optional[str]]:
    """Splits the source_code into a prefix and a suffix.

    Returns (None,None) if the splitting point is invalid.
    """
    pos = find_cursor_position(source_code, point)
    if pos == -1:
        return (None, None)

    prefix = source_code[:pos]
    suffix = source_code[pos:]
    return (prefix, suffix)


def find_cursor_position(source_code: str, point: tuple[int, int]) -> int:
    """Converts a 2D point to its 1D position in the source_code."""
    if not source_code:
        return -1

    row, col = point
    lines = source_code.splitlines()

    if row >= len(lines) or col > len(lines[row]):
        return -1

    pos = 0
    for line in lines[:row]:
        pos += len(line) + 1
    pos += col
    return pos


def convert_point_to_relative_point_in_node(
    node: Node, point: tuple[int, int]
) -> tuple[int, int]:
    """Converts the global point to the relative point within the node."""
    row = point[0] - node.start_point[0]
    col = point[1] - node.start_point[1]
    return (row, col)


async def strip_whitespaces(text: str) -> str:
    return "" if text.isspace() else text
