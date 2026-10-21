from types import SimpleNamespace

import pytest

from lib.jwt import instance_uid_claim, trusted_instance_id

GATEWAY_ISSUER = "gitlab-ai-gateway"
CUSTOMERS_DOT = "https://customers.gitlab.com"


@pytest.fixture(name="gateway_service_name", autouse=True)
def gateway_service_name_fixture(monkeypatch):
    # The Cloud Connector service name is process-global; other tests (for example the
    # Duo Workflow Service `run()` tests) can leave it set to another service's name.
    monkeypatch.setenv("CLOUD_CONNECTOR_SERVICE_NAME", GATEWAY_ISSUER)
    monkeypatch.delenv("DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL", raising=False)
    monkeypatch.delenv("AIGW_CUSTOMER_PORTAL_URL", raising=False)


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


@pytest.mark.parametrize("issuer", [CUSTOMERS_DOT, CUSTOMERS_DOT + "/"])
def test_subject_of_a_customers_dot_token_is_not_the_instance_id(issuer):
    claims = _claims(issuer=issuer, subject="instance-uuid")

    assert trusted_instance_id(claims) is None
    assert instance_uid_claim(claims) == {"gitlab_instance_uid": None}


@pytest.mark.parametrize("issuer", [CUSTOMERS_DOT, CUSTOMERS_DOT + "/"])
def test_customers_dot_token_with_instance_uid_claim_is_trusted(issuer):
    claims = _claims(issuer=issuer, subject="instance-uuid", gitlab_instance_uid="uid")

    assert trusted_instance_id(claims) == "uid"


@pytest.mark.parametrize(
    ("env", "configured"),
    [
        (
            "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL",
            "https://customers.example.com/",
        ),
        ("AIGW_CUSTOMER_PORTAL_URL", "https://customers.example.com"),
    ],
)
@pytest.mark.parametrize(
    "issuer", ["https://customers.example.com", "https://customers.example.com/"]
)
def test_configured_customers_dot_url_is_the_customers_dot_issuer(
    monkeypatch, env, configured, issuer
):
    monkeypatch.setenv(env, configured)

    assert trusted_instance_id(_claims(issuer=issuer)) is None


@pytest.mark.parametrize(
    ("issuer_env", "other_env"),
    [
        ("AIGW_CUSTOMER_PORTAL_URL", "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL"),
        ("DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL", "AIGW_CUSTOMER_PORTAL_URL"),
    ],
)
def test_customers_dot_issuer_is_recognised_when_the_urls_disagree(
    monkeypatch, issuer_env, other_env
):
    monkeypatch.setenv(issuer_env, "https://customers.example.com")
    monkeypatch.setenv(other_env, "https://customers.staging.example.com")

    claims = _claims(issuer="https://customers.example.com")

    assert trusted_instance_id(claims) is None


def test_default_customers_dot_url_is_recognised_when_both_are_configured(
    monkeypatch,
):
    monkeypatch.setenv(
        "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL", "https://customers.example.com"
    )
    monkeypatch.setenv("AIGW_CUSTOMER_PORTAL_URL", "https://customers.example.com")

    assert trusted_instance_id(_claims(issuer=CUSTOMERS_DOT)) is None


@pytest.mark.parametrize(
    "issuer", ["HTTPS://Customers.GitLab.com", "https://CUSTOMERS.gitlab.com//"]
)
def test_customers_dot_issuer_scheme_and_host_are_case_insensitive(issuer):
    assert trusted_instance_id(_claims(issuer=issuer)) is None


@pytest.mark.parametrize("env_value", ["", "   "])
def test_empty_customers_dot_url_is_skipped(monkeypatch, env_value):
    monkeypatch.setenv("DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL", env_value)
    monkeypatch.setenv("AIGW_CUSTOMER_PORTAL_URL", env_value)

    # An empty configured URL must not make an empty issuer count as CustomersDot.
    assert trusted_instance_id(_claims(issuer="")) == "instance-uuid"
    assert trusted_instance_id(_claims()) == "instance-uuid"
    assert trusted_instance_id(_claims(issuer=CUSTOMERS_DOT)) is None


@pytest.mark.parametrize(
    "issuer", ["https://gitlab.example.com", "https://gitlab.example.com/"]
)
def test_subject_of_a_self_signed_token_is_still_the_instance_id(issuer):
    assert trusted_instance_id(_claims(issuer=issuer)) == "instance-uuid"
