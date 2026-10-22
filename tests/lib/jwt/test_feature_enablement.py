import pytest

from lib.jwt import (
    MAX_FEATURE_ENABLEMENT_TYPE_LENGTH,
    parse_feature_enablement_type,
    resolve_feature_enablement_type,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("duo_pro", "duo_pro"),
        ("duo_core", "duo_core"),
        ("duo_enterprise", "duo_enterprise"),
        ("self_hosted_dap", "self_hosted_dap"),
        ("self_hosted_usage_billing", "self_hosted_usage_billing"),
        ("gitlab_credits", "gitlab_credits"),
        ("", ""),
        ("some_future_add_on", "some_future_add_on"),
        (
            "x" * MAX_FEATURE_ENABLEMENT_TYPE_LENGTH,
            "x" * MAX_FEATURE_ENABLEMENT_TYPE_LENGTH,
        ),
        ("x" * (MAX_FEATURE_ENABLEMENT_TYPE_LENGTH + 1), None),
        (None, None),
        (42, None),
        (True, None),
        (["duo_pro"], None),
        ({"type": "duo_pro"}, None),
    ],
)
def test_parse_feature_enablement_type(value, expected):
    assert parse_feature_enablement_type(value) == expected


@pytest.mark.parametrize(
    "extra,header,expected",
    [
        ({"feature_enablement_type": "duo_pro"}, "duo_core", "duo_pro"),
        ({"feature_enablement_type": "duo_core"}, "duo_pro", "duo_core"),
        ({"feature_enablement_type": ""}, "duo_pro", ""),
        ({"feature_enablement_type": 42}, "duo_pro", None),
        ({}, "duo_pro", "duo_pro"),
        ({"gitlab_root_namespace_id": 1}, "duo_core", "duo_core"),
        (None, "duo_pro", "duo_pro"),
        (None, None, None),
        ({}, "", ""),
    ],
)
def test_resolve_feature_enablement_type(extra, header, expected):
    assert resolve_feature_enablement_type(extra, header) == expected
