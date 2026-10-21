# pylint: disable=direct-environment-variable-reference
import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from ai_gateway.api.v1 import api_router

GITLAB = "https://gitlab.com"
ALLOWED = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {"gitlab_url": GITLAB}


@pytest.fixture(autouse=True)
def _pinned_config(mock_config):
    with patch("ai_gateway.api.v1.flows.enabled.get_config", return_value=mock_config):
        yield


@pytest.fixture(name="instance_uid")
def instance_uid_fixture():
    return ALLOWED


@pytest.fixture(name="issuer")
def issuer_fixture():
    return "https://customers.gitlab.com"


@pytest.fixture(name="realm")
def realm_fixture():
    return "self-managed"


@pytest.fixture(name="auth_user")
def auth_user_fixture(instance_uid, issuer, realm):
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_agent_platform"],
            issuer=issuer,
            gitlab_realm=realm,
            gitlab_instance_uid=instance_uid,
        ),
    )


@pytest.fixture(name="enforce")
def enforce_fixture():
    return "true"


@pytest.fixture(autouse=True)
def _allowlist(enforce):
    allowlist = json.dumps({"bl_security": [ALLOWED], "other_flow": [OTHER]})
    with patch.dict(
        os.environ,
        {
            "DUO_WORKFLOW_AUTH__ENABLED_FLOW_INSTANCE_UIDS": allowlist,
            "DUO_WORKFLOW_AUTH__ENFORCE_FLOW_ENABLEMENT": enforce,
        },
    ):
        yield


def _get(client: TestClient, realm: str = "self-managed"):
    return client.get(
        "/flows/enabled",
        headers={
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Realm": realm,
        },
    )


@pytest.mark.parametrize(
    ("instance_uid", "expected"), [(ALLOWED, ["bl_security"]), (OTHER, [])]
)
def test_enabled_flows_for_caller_instance(mock_client: TestClient, expected):
    response = _get(mock_client)

    assert response.status_code == 200
    assert response.json() == {"enabled_flows": expected}
    assert OTHER not in response.text
    assert ALLOWED not in response.text


@pytest.mark.parametrize("enforce", [""])
def test_no_gated_flows_off_gitlab_com_without_enforcement(mock_client: TestClient):
    response = _get(mock_client)

    assert response.status_code == 200
    assert response.json() == {"enabled_flows": []}


@pytest.mark.parametrize("enforce", ["true", ""])
@pytest.mark.parametrize(("issuer", "realm", "instance_uid"), [(GITLAB, "saas", "")])
def test_all_gated_flows_enabled_for_gitlab_com(mock_client: TestClient):
    response = _get(mock_client, realm="saas")

    assert response.status_code == 200
    assert response.json() == {"enabled_flows": ["bl_security"]}


@pytest.mark.parametrize(
    "auth_user",
    [
        CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["other"],
                gitlab_realm="self-managed",
                gitlab_instance_uid=ALLOWED,
            ),
        )
    ],
)
def test_requires_duo_agent_platform(mock_client: TestClient):
    response = _get(mock_client)

    assert response.status_code == 403
    assert response.json() == {"detail": "Unauthorized to list enabled flows"}
