from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Mapping, NamedTuple, Optional

__all__ = [
    "CodeContent",
    "LanguageId",
    "MetadataCodeContent",
    "MetadataExtraInfo",
    "MetadataPromptBuilder",
    "Prompt",
    "TokenStrategyBase",
]


class LanguageId(IntEnum):
    C = 1
    CPP = 2
    CSHARP = 3
    GO = 4
    JAVA = 5
    JS = 6
    PHP = 7
    PYTHON = 8
    RUBY = 9
    RUST = 10
    SCALA = 11
    TS = 12
    KOTLIN = 13
    ADA = 14
    APEX = 15
    ARDUINO = 16
    ASM = 17
    AWK = 18
    BASH = 19
    BATCH = 20
    BICEP = 21
    C3 = 22
    CAIRO = 23
    CLOJURE = 24
    COBOL = 25
    COMMONLISP = 26
    CRYSTAL = 27
    CUDA = 28
    D = 29
    DART = 30
    DEVICETREE = 31
    DOCKERFILE = 32
    ELISP = 33
    ELIXIR = 34
    ELM = 35
    ERLANG = 36
    FENNEL = 37
    FISH = 38
    FORTRAN = 39
    FSHARP = 40
    GDSCRIPT = 41
    GLEAM = 42
    GLSL = 43
    GROOVY = 44
    HACK = 45
    HARE = 46
    HASKELL = 47
    HAXE = 48
    HCL = 49
    HLSL = 50
    IDRIS = 51
    JAI = 52
    JANET = 53
    JQ = 54
    JULIA = 55
    LUA = 56
    LUAU = 57
    MAGIK = 58
    MATLAB = 59
    MOJO = 60
    MOVE = 61
    NIM = 62
    NIX = 63
    OBJC = 64
    OCAML = 65
    ODIN = 66
    PASCAL = 67
    PERL = 68
    PONY = 69
    POWERSHELL = 70
    PROLOG = 71
    PURESCRIPT = 72
    QL = 73
    R = 74
    RACKET = 75
    RESCRIPT = 76
    ROBOT = 77
    ROC = 78
    SCHEME = 79
    SMALLTALK = 80
    SOLIDITY = 81
    SOURCEPAWN = 82
    SQL = 83
    STARLARK = 84
    SVELTE = 85
    SWIFT = 86
    SYSTEMVERILOG = 87
    TACT = 88
    TCL = 89
    TERRAFORM = 90
    TLAPLUS = 91
    V = 92
    VB = 93
    VERILOG = 94
    VHDL = 95
    VIM = 96
    VUE = 97
    WGSL = 98
    ZIG = 99
    ZSH = 100


class MetadataCodeContent(NamedTuple):
    length: int
    length_tokens: int


class MetadataExtraInfo(NamedTuple):
    name: str
    pre: MetadataCodeContent
    post: MetadataCodeContent


class MetadataPromptBuilder(NamedTuple):
    components: Mapping[str, MetadataCodeContent]
    imports: Optional[MetadataExtraInfo] = None
    function_signatures: Optional[MetadataExtraInfo] = None
    code_context: Optional[MetadataExtraInfo] = None


class CodeContent(NamedTuple):
    text: str
    length_tokens: int


class Prompt(NamedTuple):
    prefix: str | list
    metadata: MetadataPromptBuilder
    suffix: Optional[str] = None

    def get_normalized_prefix(self) -> str:
        if isinstance(self.prefix, list):
            return "".join(map(str, self.prefix))
        return self.prefix


class TokenStrategyBase(ABC):
    @abstractmethod
    def truncate_content(
        self, text: str, max_length: int, truncation_side: str = "left"
    ) -> CodeContent:
        pass

    @abstractmethod
    def estimate_length(self, text: str | list[str]) -> list[int]:
        pass
