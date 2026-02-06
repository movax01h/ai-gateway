"""Disable UNPICKLING process-wide (safe default).

This is a critical security measure to prevent remote arbitrary code execution.
"""

from __future__ import annotations

import os
import pickle as _pickle_mod
import sys
from typing import Any

import _pickle as _c_pickle

# This module intentionally reads the flag directly to avoid importing
# additional configuration layers, which could inadvertently cache
# unpickling in an enabled state.
# pylint: disable=direct-environment-variable-reference


class PickleDisabledError(RuntimeError):
    pass


def _blocked_unpickle(*_args: Any, **_kwargs: Any) -> None:
    raise PickleDisabledError(
        "Unpickling is disabled in this environment (blocked to prevent unsafe deserialization). Nice try though!"
    )


def disable_unpickling() -> None:
    # Block the dangerous entrypoints
    for name in ("load", "loads"):
        if hasattr(_pickle_mod, name):
            setattr(_pickle_mod, name, _blocked_unpickle)

    # Block class-based API for unpickling
    class _BlockedUnpickler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise PickleDisabledError("pickle.Unpickler is disabled.")

    _pickle_mod.Unpickler = _BlockedUnpickler  # type: ignore

    # Patch the accelerated C module too (some code imports _pickle directly)
    try:

        for name in ("load", "loads"):
            if hasattr(_c_pickle, name):
                setattr(_c_pickle, name, _blocked_unpickle)

        if hasattr(_c_pickle, "Unpickler"):
            _c_pickle.Unpickler = _BlockedUnpickler  # type: ignore
    except Exception as e:
        if os.environ.get("UNPICKLING_ENABLED", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(
                "Failed to patch the C pickle module (_pickle). This is a critical security measure "
                "to prevent remote arbitrary code execution. If you need unpickling in a trusted environment, "
                "set UNPICKLING_ENABLED=true."
            ) from e

    # Optional: prevent NEW imports of pickletools
    class _BlockPickleToolsFinder:
        def find_spec(  # pylint: disable=unused-argument
            self, fullname: str, path: Any = None, target: Any = None
        ) -> None:
            if fullname == "pickletools":
                raise PickleDisabledError("Import blocked: pickletools")

    sys.meta_path.insert(0, _BlockPickleToolsFinder())


disable_unpickling()
