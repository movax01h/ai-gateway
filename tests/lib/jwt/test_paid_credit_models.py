import pytest

from lib.jwt import paid_credit_models_allowed_from_claims_extra


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"gitlab_paid_credit_models_allowed": True}, True),
        ({"gitlab_paid_credit_models_allowed": False}, False),
        ({"gitlab_paid_credit_models_allowed": "true"}, False),
        ({"gitlab_paid_credit_models_allowed": None}, False),
        ({"other_claim": True}, None),
        ({}, None),
        (None, None),
    ],
)
def test_paid_credit_models_allowed_from_claims_extra(extra, expected):
    assert paid_credit_models_allowed_from_claims_extra(extra) is expected
