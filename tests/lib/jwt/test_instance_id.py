from types import SimpleNamespace

import pytest

from lib.jwt import instance_uid_claim, trusted_instance_id

GATEWAY_ISSUER = "gitlab-ai-gateway"


def _claims(**overrides):
    values = {
        "issuer": "https://gitlab.example.com",
        "subject": "instance-uuid",
        "gitlab_instance_uid": "",
        "gitlab_instance_id": "reported-by-the-instance",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_no_claims_has_no_instance_id():
    assert trusted_instance_id(None) is None


def test_gitlab_instance_uid_claim_is_preferred():
    claims = _claims(gitlab_instance_uid="cdot-uid", subject="other")

    assert trusted_instance_id(claims) == "cdot-uid"


def test_subject_of_an_instance_signed_token_is_the_instance_id():
    assert trusted_instance_id(_claims()) == "instance-uuid"


def test_subject_of_a_gateway_issued_token_is_a_user_not_the_instance():
    claims = _claims(issuer=GATEWAY_ISSUER, subject="hashed-user-id")

    assert trusted_instance_id(claims) is None


def test_gateway_issued_token_with_instance_uid_claim_is_trusted():
    claims = _claims(
        issuer=GATEWAY_ISSUER, subject="hashed-user-id", gitlab_instance_uid="uid"
    )

    assert trusted_instance_id(claims) == "uid"


@pytest.mark.parametrize("subject", ["", None])
def test_empty_subject_has_no_instance_id(subject):
    assert trusted_instance_id(_claims(subject=subject)) is None


def test_instance_reported_id_is_never_used():
    claims = _claims(subject="", gitlab_instance_uid="")

    assert trusted_instance_id(claims) is None


def test_instance_uid_claim_carries_the_trusted_instance_id():
    assert instance_uid_claim(_claims()) == {"gitlab_instance_uid": "instance-uuid"}


def test_instance_uid_claim_is_none_for_a_gateway_issued_token():
    claims = _claims(issuer=GATEWAY_ISSUER, subject="hashed-user-id")

    assert instance_uid_claim(claims) == {"gitlab_instance_uid": None}
