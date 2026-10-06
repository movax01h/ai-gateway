from collections.abc import Mapping
from typing import Any

FEATURE_ENABLEMENT_TYPE_CLAIM = "feature_enablement_type"

MAX_FEATURE_ENABLEMENT_TYPE_LENGTH = 64


def parse_feature_enablement_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    if len(value) > MAX_FEATURE_ENABLEMENT_TYPE_LENGTH:
        return None

    return value


def resolve_feature_enablement_type(
    extra: Mapping[str, Any] | None,
    header_value: str | None,
) -> str | None:
    if extra is not None and FEATURE_ENABLEMENT_TYPE_CLAIM in extra:
        return parse_feature_enablement_type(extra[FEATURE_ENABLEMENT_TYPE_CLAIM])

    return parse_feature_enablement_type(header_value)
