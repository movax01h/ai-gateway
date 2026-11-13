# pylint: disable=file-naming-for-tests
"""Every console script in pyproject.toml must name a callable that imports.

No CI job starts the services through their scripts, so a renamed module or function would otherwise ship a broken
script.
"""

import importlib
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_SCRIPTS = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"]


@pytest.mark.parametrize(("script", "target"), sorted(_SCRIPTS.items()))
def test_console_script_target_resolves(script, target):
    module_name, _, attribute = target.partition(":")

    target_object = getattr(importlib.import_module(module_name), attribute)

    assert callable(target_object), f"{script} = {target!r} is not callable"
