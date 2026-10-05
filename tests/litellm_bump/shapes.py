"""Reduce an outgoing request to its schema: path, header names, and the structure of the body.

Values are ignored. The schema changes only when litellm adds, removes, renames or retypes something it sends.
"""

import json
from typing import Any

from tests.litellm_bump.conftest import CapturedRequest

# Headers that httpx adds on its own. They say nothing about what litellm builds.
HTTPX_HEADERS = {
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "user-agent",
}


def body_shape(value: Any) -> Any:
    """Return the structure of a JSON value: keys and nesting, with scalar values replaced by their type."""
    if isinstance(value, dict):
        return {key: body_shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        shapes = {json.dumps(body_shape(item), sort_keys=True) for item in value}
        return [json.loads(shape) for shape in sorted(shapes)]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "null"


def request_shape(request: CapturedRequest) -> dict[str, Any]:
    """Return the schema of a captured request."""
    return {
        "method": request.method,
        "path": request.url.path,
        "headers": sorted(set(request.headers.keys()) - HTTPX_HEADERS),
        "body": body_shape(request.body),
    }
