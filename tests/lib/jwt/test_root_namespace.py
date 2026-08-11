import pytest

from lib.jwt import (
    parse_root_namespace_id,
    root_namespace_id_from_claims_extra,
    root_namespace_id_from_header,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        ("77", 77),
        (1, 1),
        ("1", 1),
        (True, None),
        (False, None),
        (42.9, None),
        ("42.9", None),
        (0, None),
        ("0", None),
        (-1, None),
        ("-1", None),
        (None, None),
        ("", None),
        ("not-a-number", None),
        ([42], None),
        ({"id": 42}, None),
        (" 42", None),
        ("42 ", None),
    ],
)
def test_parse_root_namespace_id(value, expected):
    assert parse_root_namespace_id(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("42", 42),
        (None, None),
        ("", None),
        ("null", None),
        ("undefined", None),
        ("0", None),
        ("-1", None),
        ("not-a-number", None),
    ],
)
def test_root_namespace_id_from_header(value, expected):
    assert root_namespace_id_from_header(value) == expected


class TestRootNamespaceIdFromClaimsExtra:
    @pytest.mark.parametrize(
        "extra,expected",
        [
            ({"gitlab_root_namespace_id": 42}, 42),
            ({"gitlab_root_namespace_id": "77"}, 77),
            ({}, None),
            (None, None),
            ({"gitlab_root_namespace_id": None}, None),
        ],
    )
    def test_returns_validated_claim(self, extra, expected):
        assert root_namespace_id_from_claims_extra(extra) == expected
