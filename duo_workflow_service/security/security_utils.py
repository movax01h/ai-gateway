"""Utility functions for security module."""

import hashlib
import json
from typing import Any, Dict, List, Union


def compute_response_hash(response: Union[str, Dict[str, Any], List[Any]]) -> str:
    """Compute a stable hash of a response for comparison.

    This function serializes the response to a stable JSON format and computes
    a SHA-256 hash. This is more efficient than string comparison for large responses
    and more accurate than naive string conversion.

    Args:
        response: The response data to hash (str, dict, or list)

    Returns:
        A SHA-256 hex digest string
    """
    try:
        # Serialize to stable JSON format with sorted keys
        json_str = json.dumps(response, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to string conversion if JSON serialization fails
        return hashlib.sha256(str(response).encode("utf-8")).hexdigest()


def compute_response_hash_with_length(
    response: Union[str, Dict[str, Any], List[Any]],
) -> tuple[str, int]:
    """Compute hash and length of a response in a single pass.

    This avoids double serialization when we need both hash and length for logging.
    Since json.dumps() already returns a string, we compute the hash and get the
    length from the same serialized string to avoid redundant conversions.

    Args:
        response: The response data to hash (str, dict, or list)

    Returns:
        Tuple of (hash, length) where hash is SHA-256 hex digest and length is string length
    """
    try:
        # Serialize to stable JSON format with sorted keys (only once!)
        json_str = json.dumps(response, sort_keys=True, default=str, ensure_ascii=False)
        hash_value = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        return (hash_value, len(json_str))
    except Exception:
        # Fallback to string conversion if JSON serialization fails
        str_value = str(response)
        hash_value = hashlib.sha256(str_value.encode("utf-8")).hexdigest()
        return (hash_value, len(str_value))
