"""The scoped-scan file list: repo-relative paths to review instead of discovering them.

On MR pipelines, these are the files the MR changed.
"""

import json
import re
from typing import Any, Optional

__all__ = ["resolve_target_files"]

# Whitespace separates paths, so paths must not contain it.
_TARGET_FILES_SPLIT = re.compile(r"[,;\s]+")


def resolve_target_files(value: Optional[Any]) -> list[str]:
    """Parse the scoped-scan file list into an ordered, de-duplicated path list.

    Accepts a list/tuple of paths, a JSON array of paths, or one string of paths separated by commas, semicolons or
    whitespace. Each path loses surrounding quotes, a leading ``./`` or ``/``, and a bracket left over from a string
    that only looks like a JSON array. A bracket that belongs to the path, as in ``[id].tsx``, is kept. Any miss
    returns ``[]`` ("no scoped scan"). Order is the caller's.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        text = str(value).strip()
        if text[:1] == "[":
            try:
                decoded = json.loads(text)
            except ValueError:
                decoded = None
            items = (
                [str(v) for v in decoded] if isinstance(decoded, list) else [str(value)]
            )
        else:
            items = [str(value)]

    out: list[str] = []
    seen = set()
    for item in items:
        for part in _TARGET_FILES_SPLIT.split(item):
            path = _strip_stray_brackets(part.strip().strip("'\"")).strip("'\"").strip()
            path = re.sub(r"^(?:\./|/)+", "", path)
            if not path or path in seen:
                continue
            seen.add(path)
            out.append(path)
    return out


def _strip_stray_brackets(part: str) -> str:
    """Drop a leading ``[`` with no ``]`` after it, and a trailing ``]`` with no ``[`` before it.

    Those are the edges of a malformed JSON array (``[a.py, b.py]`` splits into ``[a.py`` and ``b.py]``). Brackets
    that pair up belong to the path, as in Next.js routes such as ``[id].tsx``.
    """
    if part.startswith("[") and "]" not in part:
        part = part[1:]
    if part.endswith("]") and "[" not in part:
        part = part[:-1]
    return part
