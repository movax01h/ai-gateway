# pylint: disable=direct-environment-variable-reference
import json
import os
from unittest.mock import patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from structlog.testing import capture_logs

from lib.flow_enablement import (
    enabled_gated_flows,
    is_flow_enabled,
    is_gitlab_com_signed,
    refused_while_not_enforced,
)

ALLOWED = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
GITLAB = "https://gitlab.com"
ENV = "DUO_WORKFLOW_AUTH__ENABLED_FLOW_INSTANCE_UIDS"
ENFORCE_ENV = "DUO_WORKFLOW_AUTH__ENFORCE_FLOW_ENABLEMENT"
SERVICE = "gitlab-duo-workflow-service"


@pytest.fixture(autouse=True)
def _service_name():
    env = {
        "CLOUD_CONNECTOR_SERVICE_NAME": SERVICE,
        ENFORCE_ENV: "true",
        "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL": "https://customers.gitlab.com",
    }
    with patch.dict(os.environ, env):
        yield


def _user(
    instance_uid: str = OTHER,
    issuer: str = "https://customers.gitlab.com",
    realm: str = "self-managed",
    extra: dict | None = None,
    is_debug: bool = False,
) -> CloudConnectorUser:
    return CloudConnectorUser(
        authenticated=True,
        is_debug=is_debug,
        claims=UserClaims(
            issuer=issuer,
            gitlab_realm=realm,
            gitlab_instance_uid=instance_uid,
            extra=extra or {},
        ),
    )


def _subject_only_user(issuer: str = "https://customers.gitlab.com"):
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            issuer=issuer, subject=ALLOWED, gitlab_realm="saas", gitlab_instance_uid=""
        ),
    )


def _enabled(user: CloudConnectorUser, allowlist: str, flow="bl_security") -> bool:
    with patch.dict(os.environ, {ENV: allowlist}):
        return is_flow_enabled(user, flow, gitlab_issuer=GITLAB)


@pytest.mark.parametrize(
    ("uid", "expected"),
    [
        (ALLOWED, True),
        (ALLOWED.upper(), True),
        (ALLOWED.replace("-", ""), True),
        (OTHER, False),
        ("", False),
        ("not-a-uuid", False),
    ],
)
def test_allowlisted_instance(uid, expected):
    assert _enabled(_user(uid), json.dumps({"bl_security": [ALLOWED]})) is expected


@pytest.mark.parametrize(
    "allowlist",
    [
        "",
        "   ",
        "{not json",
        "[]",
        '"bl_security"',
        json.dumps({"bl_security": ALLOWED}),
    ],
)
def test_empty_or_malformed_allowlist_enables_nothing(allowlist):
    assert _enabled(_user(ALLOWED), allowlist) is False


def test_malformed_uids_are_dropped_and_logged():
    allowlist = json.dumps({"bl_security": ["bad", 42, ALLOWED.upper()]})

    with capture_logs() as logs:
        assert _enabled(_user(ALLOWED), allowlist) is True

    assert [log["dropped"] for log in logs] == [2]


@pytest.mark.parametrize(
    "issuer", ["https://customers.gitlab.com", "https://customers.gitlab.com/"]
)
def test_customers_dot_subject_is_not_allowlisted(issuer):
    allowlist = json.dumps({"bl_security": [ALLOWED]})

    assert _enabled(_subject_only_user(issuer), allowlist) is False


def test_entries_for_flows_not_gated_are_logged_without_their_uids():
    allowlist = json.dumps({"bl_security": [ALLOWED], "bl_securty": [OTHER]})

    with capture_logs() as logs:
        assert _enabled(_user(ALLOWED), allowlist) is True

    assert logs == [
        {
            "event": "Flow allowlist entry is not a gated flow",
            "flow": "bl_securty",
            "log_level": "warning",
        }
    ]


def test_allowlist_is_per_flow():
    allowlist = json.dumps({"other_flow": [ALLOWED]})

    assert _enabled(_user(ALLOWED), allowlist) is False


def test_flows_not_gated_are_enabled():
    assert _enabled(_user(), "", flow="developer") is True


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        pytest.param(_user(issuer=GITLAB, realm="saas"), True, id="gitlab-com-signed"),
        pytest.param(_user(issuer=GITLAB + "/", realm="saas"), True, id="slash"),
        pytest.param(_user(issuer=GITLAB), False, id="gitlab-com-signer-not-saas"),
        pytest.param(_user(realm="saas"), False, id="customers-dot-saas"),
        pytest.param(
            _user(issuer=SERVICE, extra={"gitlab_com_verified": True}),
            True,
            id="minted-for-gitlab-com",
        ),
        pytest.param(_user(issuer=SERVICE, realm="saas"), False, id="minted-saas"),
        pytest.param(
            _user(issuer=GITLAB, realm="saas", extra={"gitlab_com_verified": "true"}),
            True,
            id="claim-ignored-off-minted-tokens",
        ),
        pytest.param(
            _user(extra={"gitlab_com_verified": True}), False, id="claim-not-minted"
        ),
        pytest.param(_user(is_debug=True), True, id="debug"),
    ],
)
def test_gitlab_com_and_debug_callers(user, expected):
    assert _enabled(user, "") is expected


@pytest.mark.parametrize(
    ("configured", "issuer"),
    [(GITLAB + "/", GITLAB), (GITLAB, GITLAB + "/"), (GITLAB + "/", GITLAB + "/")],
)
def test_gitlab_com_issuer_ignores_trailing_slash(configured, issuer):
    user = _user(issuer=issuer, realm="saas")

    with patch.dict(os.environ, {ENV: ""}):
        assert is_flow_enabled(user, "bl_security", gitlab_issuer=configured) is True


@pytest.mark.parametrize("enforce", ["", "false", "0"])
@pytest.mark.parametrize(
    ("user", "expected"),
    [
        pytest.param(_user(ALLOWED), False, id="allowlisted"),
        pytest.param(_user(issuer=GITLAB, realm="saas"), True, id="gitlab-com"),
    ],
)
def test_allowlist_not_read_without_enforcement(enforce, user, expected):
    with patch.dict(os.environ, {ENFORCE_ENV: enforce}):
        assert _enabled(user, json.dumps({"bl_security": [ALLOWED]})) is expected


@pytest.mark.parametrize("enforce", ["true\n", " 1 ", "TRUE"])
def test_enforcement_switch_ignores_surrounding_whitespace(enforce):
    with patch.dict(os.environ, {ENFORCE_ENV: enforce}):
        assert _enabled(_user(ALLOWED), json.dumps({"bl_security": [ALLOWED]})) is True


def test_enabled_gated_flows():
    allowlist = json.dumps({"bl_security": [ALLOWED]})

    with patch.dict(os.environ, {ENV: allowlist}):
        assert enabled_gated_flows(_user(ALLOWED), gitlab_issuer=GITLAB) == [
            "bl_security"
        ]
        assert enabled_gated_flows(_user(OTHER), gitlab_issuer=GITLAB) == []


@pytest.mark.parametrize(
    ("user", "flow", "refused"),
    [
        pytest.param(_user(), "bl_security", True, id="own-token-off-gitlab-com"),
        pytest.param(_user(ALLOWED), "bl_security", True, id="own-token-allowlisted"),
        pytest.param(
            _user(issuer=GITLAB, realm="saas"), "bl_security", False, id="gitlab-com"
        ),
        pytest.param(_user(issuer=SERVICE), "bl_security", False, id="minted-token"),
        pytest.param(_user(is_debug=True), "bl_security", False, id="debug"),
        pytest.param(_user(), "developer", False, id="not-gated"),
    ],
)
def test_refused_while_not_enforced(user, flow, refused):
    with patch.dict(os.environ, {ENFORCE_ENV: ""}):
        assert refused_while_not_enforced(user, flow, gitlab_issuer=GITLAB) is refused


@pytest.mark.parametrize(
    ("configured", "issuer", "realm", "expected"),
    [
        (GITLAB, GITLAB, "saas", True),
        (GITLAB, f"{GITLAB}/", "saas", True),
        (f"{GITLAB}/", GITLAB, "saas", True),
        (GITLAB, "https://gitlab.example.com", "saas", False),
        (GITLAB, "", "saas", False),
        ("", GITLAB, "saas", False),
        (GITLAB, GITLAB, "self-managed", False),
        (GITLAB, GITLAB, None, False),
    ],
)
def test_is_gitlab_com_signed(configured, issuer, realm, expected):
    claims = UserClaims(issuer=issuer, gitlab_realm=realm)
    assert is_gitlab_com_signed(claims, configured) is expected


def test_is_gitlab_com_signed_without_claims():
    assert is_gitlab_com_signed(None, GITLAB) is False
