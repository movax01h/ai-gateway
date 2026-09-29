#!/usr/bin/env python
"""Pin the litellm API surface that AI Gateway depends on.

AI Gateway imports a number of litellm internals that carry no deprecation
policy and move between minor versions. A bump that renames or relocates one of
them breaks at import time in production while every unit test still passes,
because the test suite patches litellm at the object level.

This module keeps an explicit manifest of that surface and checks it against the
installed litellm. Run it with ``--check`` in CI, ``--snapshot`` to re-record
signatures after a deliberate change, ``--check-complete`` to prove the manifest
still covers every litellm import in the tree, and ``--diff`` to compare two
released versions while preparing a bump.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import subprocess  # nosec B404 - only used for --diff, with a fixed argv
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = Path(__file__).resolve().parent / "litellm_surface.json"

# Directories scanned by --check-complete. vendor/ and scripts/ are deliberately
# included: ai_gateway/vendor/langchain_litellm/litellm.py is one of the most
# fragile consumers in the tree, and it calls get_valid_models at import time.
SOURCE_DIRS = (
    "ai_gateway",
    "lib",
    "duo_workflow_service",
)

# Modules that must import cleanly. A move or removal here is the failure mode
# behind !5183 (ModuleNotFoundError on litellm.proxy.management_endpoints).
MODULES: tuple[str, ...] = (
    "litellm",
    "litellm.exceptions",
    "litellm.utils",
    "litellm.types.utils",
    "litellm.responses.utils",
    "litellm.llms.openai.openai",
    "litellm.llms.custom_httpx.http_handler",
    "litellm.proxy._types",
    "litellm.proxy.pass_through_endpoints.pass_through_endpoints",
    "litellm.litellm_core_utils.prompt_templates.factory",
)

# Symbols the repo imports by name, as {module: (name, ...)}.
SYMBOLS: dict[str, tuple[str, ...]] = {
    "litellm": (
        "AnthropicConfig",
        "CustomStreamWrapper",
        "ModelResponse",
        "OpenAIGPT5Config",
        "acompletion",
        "completion",
        "register_model",
    ),
    "litellm.exceptions": (
        "APIConnectionError",
        "BadRequestError",
        "ContextWindowExceededError",
        "InternalServerError",
        "ServiceUnavailableError",
    ),
    "litellm.utils": ("get_valid_models",),
    "litellm.types.utils": ("Delta", "ModelResponseStream"),
    "litellm.responses.utils": ("ResponseAPILoggingUtils",),
    "litellm.llms.openai.openai": ("OpenAIChatCompletionResponseIterator",),
    "litellm.llms.custom_httpx.http_handler": ("AsyncHTTPHandler",),
    "litellm.proxy._types": ("ProxyException", "UserAPIKeyAuth"),
    "litellm.proxy.pass_through_endpoints.pass_through_endpoints": (
        "create_pass_through_route",
    ),
}

# Attributes the repo replaces or reaches into. These are the silent failures:
# if litellm renames chunk_parser, the assignment in
# ai_gateway/proxy/clients/__init__.py creates a dead attribute and the
# override never runs, with no error anywhere.
PATCH_TARGETS: tuple[tuple[str, str | None, str], ...] = (
    (
        "litellm.llms.openai.openai",
        "OpenAIChatCompletionResponseIterator",
        "chunk_parser",
    ),
    (
        "litellm.responses.utils",
        "ResponseAPILoggingUtils",
        "_transform_response_api_usage_to_chat_usage",
    ),
    (
        "litellm",
        "OpenAIGPT5Config",
        "is_model_gpt_5_model",
    ),
    # The permanent fix for #2335 replaces this module-level function. If
    # litellm renames it the import in litellm_empty_text_patch.py raises, but
    # if litellm changes what it does the override silently stops matching, so
    # the wire fixtures in Stage C are what actually cover the behaviour.
    (
        "litellm.litellm_core_utils.prompt_templates.factory",
        None,
        "_sanitize_empty_text_content",
    ),
)

# Callables whose full signature is pinned to the snapshot. Reserved for narrow,
# stable functions and for anything the repo wraps or monkey-patches, where an
# added parameter really does mean our call site needs review. !5453 had to
# update a monkey-patch after create_pass_through_route changed shape.
PINNED_SIGNATURES: tuple[tuple[str, str | None, str], ...] = (
    ("litellm", None, "register_model"),
    ("litellm.utils", None, "get_valid_models"),
    (
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints",
        None,
        "create_pass_through_route",
    ),
    ("litellm", "OpenAIGPT5Config", "is_model_gpt_5_model"),
)

# Callables that take a wide, growing keyword surface. Pinning the whole
# signature here would fire on every upstream release that adds a parameter,
# which is noise. What actually matters is that the parameters our call sites
# pass by name still exist, so only those are asserted.
#
# Keys not listed here are passed through litellm's **kwargs by design
# (for example "client" and "prompt_cache_max_len" in
# ai_gateway/models/litellm.py) and cannot be checked this way.
REQUIRED_PARAMS: tuple[tuple[str, str | None, str, tuple[str, ...]], ...] = (
    (
        "litellm",
        None,
        "acompletion",
        (
            "model",
            "messages",
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "timeout",
            "stop",
            "tools",
            "tool_choice",
            "logprobs",
            "api_key",
            "api_version",
            "extra_headers",
        ),
    ),
)


class SurfaceError(Exception):
    """Raised when the installed litellm no longer matches the manifest."""


@dataclass(frozen=True)
class Finding:
    """A single mismatch between the manifest and the installed litellm."""

    kind: str
    target: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.target}: {self.detail}"


def installed_version() -> str:
    """Return the installed litellm version, or 'unknown'."""
    try:
        return metadata.version("litellm")
    except metadata.PackageNotFoundError:
        return "unknown"


def _qualname(module: str, holder: str | None, name: str) -> str:
    return f"{module}.{holder}.{name}" if holder else f"{module}.{name}"


def _resolve(module: str, holder: str | None, name: str) -> Any:
    obj = importlib.import_module(module)
    if holder:
        obj = getattr(obj, holder)
    return getattr(obj, name)


def check_modules() -> list[Finding]:
    """Import every manifest module and report the ones that fail."""
    findings = []
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            findings.append(Finding("module", module, f"import failed: {exc!r}"))
    return findings


def check_symbols() -> list[Finding]:
    """Check that every symbol the repo imports by name still exists."""
    findings = []
    for module, names in SYMBOLS.items():
        try:
            mod = importlib.import_module(module)
        except Exception as exc:
            findings.append(Finding("symbol", module, f"import failed: {exc!r}"))
            continue
        for name in names:
            if not hasattr(mod, name):
                findings.append(
                    Finding("symbol", f"{module}.{name}", "no longer exported")
                )
    return findings


def check_patch_targets() -> list[Finding]:
    """Check that every attribute the repo overrides still exists and is callable.

    Existence alone is not enough. If ``chunk_parser`` became a plain attribute
    rather than a method, assigning a function over it would still "work" and
    still be wrong, so the kind is checked too.
    """
    findings = []
    for module, holder, name in PATCH_TARGETS:
        target = _qualname(module, holder, name)
        try:
            attr = _resolve(module, holder, name)
        except Exception as exc:
            findings.append(Finding("patch-target", target, f"missing: {exc!r}"))
            continue
        if not callable(attr):
            findings.append(
                Finding("patch-target", target, f"not callable, got {type(attr)!r}")
            )
    return findings


def current_signatures() -> dict[str, str]:
    """Return the current signature of every pinned callable."""
    signatures = {}
    for module, holder, name in PINNED_SIGNATURES:
        target = _qualname(module, holder, name)
        try:
            signatures[target] = str(inspect.signature(_resolve(module, holder, name)))
        except Exception as exc:
            signatures[target] = f"<unresolvable: {exc!r}>"
    return signatures


def check_signatures(snapshot: dict[str, str]) -> list[Finding]:
    """Compare pinned signatures against the recorded snapshot."""
    findings = []
    for target, current in current_signatures().items():
        recorded = snapshot.get(target)
        if recorded is None:
            findings.append(
                Finding("signature", target, "not in snapshot; run --snapshot")
            )
        elif recorded != current:
            findings.append(
                Finding("signature", target, f"was {recorded}, now {current}")
            )
    return findings


def check_required_params() -> list[Finding]:
    """Check that every parameter our call sites pass by name still exists."""
    findings = []
    for module, holder, name, required in REQUIRED_PARAMS:
        target = _qualname(module, holder, name)
        try:
            params = inspect.signature(_resolve(module, holder, name)).parameters
        except Exception as exc:
            findings.append(Finding("params", target, f"unresolvable: {exc!r}"))
            continue
        for param in required:
            if param not in params:
                findings.append(
                    Finding("params", f"{target}({param}=)", "parameter removed")
                )
    return findings


def _iter_python_files(dirs: Iterable[str]) -> Iterator[Path]:
    for directory in dirs:
        root = REPO_ROOT / directory
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _iter_litellm_imports(path: Path) -> Iterator[tuple[int, str, str | None]]:
    """Yield (lineno, module, symbol) for every litellm import in a file.

    Uses ``ast`` rather than a regex so aliased, parenthesised and multi-line
    imports are all seen.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "litellm" or alias.name.startswith("litellm."):
                    yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "litellm" or module.startswith("litellm."):
                for alias in node.names:
                    yield node.lineno, module, alias.name


def check_manifest_complete() -> list[Finding]:
    """Fail if any litellm import in the tree is missing from the manifest.

    The manifest is hand-maintained, so without this an engineer can add a new
    private-API dependency and get no protection from it. Adding a litellm
    internal should be a deliberate, reviewed act, and a one-line manifest edit
    is what makes it one.
    """
    findings = []
    for path in _iter_python_files(SOURCE_DIRS):
        rel = path.relative_to(REPO_ROOT)
        for lineno, module, symbol in _iter_litellm_imports(path):
            where = f"{rel}:{lineno}"
            if symbol is None:
                if module not in MODULES:
                    findings.append(
                        Finding(
                            "manifest",
                            module,
                            f"imported at {where} but not in MODULES",
                        )
                    )
                continue
            known = SYMBOLS.get(module, ())
            submodule = f"{module}.{symbol}"
            if symbol not in known and submodule not in MODULES:
                findings.append(
                    Finding(
                        "manifest",
                        submodule,
                        f"imported at {where} but not in SYMBOLS",
                    )
                )
    return findings


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, str]:
    """Read the recorded signature snapshot."""
    if not path.exists():
        raise SurfaceError(f"snapshot not found at {path}; run --snapshot")
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data["signatures"])


def write_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Record the installed litellm's signatures to disk."""
    payload = {
        "_comment": (
            "Recorded litellm API surface. Regenerate with "
            "'poetry run python scripts/litellm_surface.py --snapshot' when a "
            "signature change is deliberate, and explain it in the MR."
        ),
        "litellm_version": installed_version(),
        "signatures": current_signatures(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_checks() -> list[Finding]:
    """Run every manifest check and return all findings."""
    findings = check_modules()
    findings += check_symbols()
    findings += check_patch_targets()
    findings += check_signatures(load_snapshot())
    findings += check_required_params()
    findings += check_manifest_complete()
    return findings


def _pip_surface(version: str) -> dict[str, Any]:
    """Install one litellm version in a throwaway venv and dump its surface."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(  # nosec B603 - fixed argv, no shell
            [sys.executable, "-m", "venv", tmp], check=True, capture_output=True
        )
        python = Path(tmp) / "bin" / "python"
        subprocess.run(  # nosec B603
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                f"litellm[proxy]=={version}",
            ],
            check=True,
            capture_output=True,
        )
        probe = (
            "import json,sys;"
            f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r});"
            "import litellm_surface as s;"
            "print(json.dumps({'signatures': s.current_signatures(),"
            "'modules': [f.target for f in s.check_modules()],"
            "'symbols': [f.target for f in s.check_symbols()],"
            "'patch_targets': [f.target for f in s.check_patch_targets()]}))"
        )
        out = subprocess.run(  # nosec B603
            [str(python), "-c", probe], check=True, capture_output=True, text=True
        )
        return json.loads(out.stdout)


def diff_versions(old: str, new: str) -> int:
    """Print the surface delta between two litellm releases."""
    before = _pip_surface(old)
    after = _pip_surface(new)

    print(f"litellm {old} -> {new}\n")

    for key in ("modules", "symbols", "patch_targets"):
        broke = sorted(set(after[key]) - set(before[key]))
        fixed = sorted(set(before[key]) - set(after[key]))
        for target in broke:
            print(f"  BROKEN  {key}: {target}")
        for target in fixed:
            print(f"  FIXED   {key}: {target}")

    for target, old_sig in sorted(before["signatures"].items()):
        new_sig = after["signatures"].get(target)
        if new_sig != old_sig:
            print(
                f"  CHANGED signature: {target}\n            {old_sig}\n         -> {new_sig}"
            )

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="verify the installed litellm"
    )
    group.add_argument(
        "--check-complete",
        action="store_true",
        help="verify the manifest covers every litellm import in the tree",
    )
    group.add_argument(
        "--snapshot", action="store_true", help="re-record the signature snapshot"
    )
    group.add_argument(
        "--diff", nargs=2, metavar=("OLD", "NEW"), help="compare two litellm releases"
    )
    args = parser.parse_args(argv)

    if args.diff:
        return diff_versions(*args.diff)

    if args.snapshot:
        payload = write_snapshot()
        print(
            f"Recorded {len(payload['signatures'])} signatures for "
            f"litellm {payload['litellm_version']} to {SNAPSHOT_PATH}"
        )
        return 0

    findings = check_manifest_complete() if args.check_complete else run_checks()

    if not findings:
        print("litellm API surface matches the manifest.")
        return 0

    print(f"{len(findings)} litellm API surface finding(s):\n")
    for finding in findings:
        print(f"  {finding}")
    print(
        "\nIf a change is deliberate, update the manifest in "
        "scripts/litellm_surface.py and re-run with --snapshot. "
        "See docs/litellm_upgrades.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
