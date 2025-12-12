from datetime import datetime, timedelta, timezone

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from gitlab_cloud_connector import CloudConnectorUser, CompositeProvider, UserClaims
from jose import jwt

from ai_gateway.api.v1 import api_router

# JSON Web Key can be generated via https://mkjwk.org/
# Private key: X.509 PEM format
# Public key: JWK format
TEST_PRIVATE_KEY = """
-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDOSYeAVOeCI7uY
/b32rEmj2uh39z0pLLJ3UbkX+Ukxe2hnnpFqzcgCcpA+XXu8Nepfw/pOcXavb5Jm
1sI3Cm7mjLe+s2+lvTp5PpUh4GMP26J2HRU9NBQdzRVzEYM5DzC9H75GCjF3POSO
sf4ouzCnsNgpDHuKm3yT0Ck1SvR/87iBYOF2xnbfegJO+O4USKZnlYythkDgkR7r
9Ris+JzbIKKOHQf/w+GPYwvL6eM8LyTb5fVZGilCzm4H9onErhJc8fQijNJSwPUn
+Md/+Uu1pmE9raHHTe9cD+QXEupJn0NWU+bXFVJochYvm8ULP4AlGpfvf3CxGg08
pyAfq+BhAgMBAAECggEATxjgjPOBRWRAJVx9/1x2bA6e/ojdebE6yQeb6jZau09v
a/PgHEzFOTMGXfNoY3Vk5c12Z6eX85LbVvVXyNUGSv5/4e5Zi/pvtlepxTCNq2Hy
/EkQgMQ8RmUBqXp4j2Nks8+9HIwCBY9ir9hN9P45nMLxT2QK5s3RybeSZW3VLE3u
IGV+S8N4K4WhSzyNdZ7hvHTzvIi8s05KAGPAAGqZyF7WNiQhIlFSI8jGj7aNx9+5
zQ5Dify1c1jRORvxdMPE5FumumLghf869Nhu8h/7NlJVokgIdVhcCcMIQlW0VZdC
uFpGfCXmEf7iSaRFjRs4De1JLCEb6BLKwxnU2BVstQKBgQDnF9lvn8SzrCFEXZhc
vhsm//xoAvLSw00ObfEUyDnnjmu5tyBncJiPe6J+aO+StI7yXzELRr+yiCky5/MJ
QqRSgKEzrl6OAiTvZqMOG5ee5gW62GQp81549ZUCyYMVsfP+DDyMQaHYWi4T8wq0
hV4uuvTQNmqzGs45Ell0fyAuAwKBgQDkhUDB2W8TB4W4qspBV9hRIUHVv998s8kJ
sLiX+fL1H8tORrqzlbcRMTxGb7X+G+3jz9NG6tqrDRmLlg3D4WPhlGgxMqDtZaDu
7KXZuKm0+nCTcNUuYgBB9uA+rWAA+eFIANLA2eLnmi4p+4t1oR+p2ap7DFHVkQMS
YYia83/MywKBgF+YCAQazR2d6K0FIo/KvCSn49uKzLPOwkNjy0RTh1B4I6vRSwA/
HXzNIey0r9W6Bx/PrNQDUi0iEhjSxkBgZuUR/J0KVmbcEDdP98dQNqoucNRXyydn
Wv8iZ5+diDIjSNEgcrN6Ot7qfwEVmqoOOWWPRNIUkJLCVehZ5NNB+yfNAoGAB04X
LtsziMkxxiB3jLUxLg7BGwMiMstQfuXOUNVlpd5ZUmxCZaFAk+UeByZlC/V6mlC3
cUnqqZMmoOawE/XtinWDCyeSK2SXS2v3NUmI60ciOCRgPDZXycQJkRdbvUw/nlyg
YBfXAA5WsXLgF2eKKpTRtVNEfm4/SeQiSMnF6RcCgYBq1nGimrNcjKOs0dxuzG6F
PO/rnSfjDo0kziQWSZa1VSX169+QRlyuonKsZlJDh9uvzsGliTmF7Q1rYGz7LdKn
8iTBxQ+vGwmSXnWU1lYxeGAzzdd3jdobmXBlXafpfNALSFMfX0AoXaZmrERrSCUd
OI8NZeUpzWYJEt7fPfKP2g==
-----END PRIVATE KEY-----
"""

TEST_PUBLIC_KEY = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzkmHgFTngiO7mP299qxJ
o9rod/c9KSyyd1G5F/lJMXtoZ56Ras3IAnKQPl17vDXqX8P6TnF2r2+SZtbCNwpu
5oy3vrNvpb06eT6VIeBjD9uidh0VPTQUHc0VcxGDOQ8wvR++RgoxdzzkjrH+KLsw
p7DYKQx7ipt8k9ApNUr0f/O4gWDhdsZ233oCTvjuFEimZ5WMrYZA4JEe6/UYrPic
2yCijh0H/8Phj2MLy+njPC8k2+X1WRopQs5uB/aJxK4SXPH0IozSUsD1J/jHf/lL
taZhPa2hx03vXA/kFxLqSZ9DVlPm1xVSaHIWL5vFCz+AJRqX739wsRoNPKcgH6vg
YQIDAQAB
-----END PUBLIC KEY-----
"""

GLOBAL_USER_ID = "777"


@pytest.fixture(name="fast_api_router", scope="class")
def fast_api_router_fixture():
    return api_router


@pytest.fixture(name="auth_user")
def auth_user_fixture(request):
    claims = UserClaims(
        scopes=["complete_code", "ai_gateway_model_provider_proxy"],
        gitlab_realm="self-managed",
    )
    return CloudConnectorUser(authenticated=True, claims=claims)


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {
        "self_signed_jwt": {
            "signing_key": TEST_PRIVATE_KEY,
        }
    }


class TestUserAccessTokenSuccessSelfManaged:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["complete_code", "ai_gateway_model_provider_proxy"],
            gitlab_realm="self-managed",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_success(
        self, mock_client: TestClient, mock_track_internal_event
    ):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "self-managed",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)

        assert response.status_code == status.HTTP_200_OK

        parsed_response = response.json()
        decoded_token = jwt.decode(
            parsed_response["token"],
            TEST_PUBLIC_KEY,
            audience="gitlab-ai-gateway",
            algorithms=CompositeProvider.SUPPORTED_ALGORITHMS,
        )

        current_time = datetime.now(timezone.utc)
        current_time_posix = int(current_time.timestamp())

        assert decoded_token["iss"] == "gitlab-ai-gateway"
        assert decoded_token["sub"] == GLOBAL_USER_ID
        assert decoded_token["aud"] == "gitlab-ai-gateway"
        assert decoded_token["exp"] > current_time_posix
        assert (decoded_token["exp"]) <= int(
            (current_time + timedelta(hours=1)).timestamp()
        )
        assert decoded_token["nbf"] <= current_time_posix
        assert decoded_token["iat"] <= current_time_posix
        assert decoded_token["jti"]
        assert decoded_token["gitlab_realm"] == "self-managed"
        assert decoded_token["scopes"] == [
            "complete_code",
            "ai_gateway_model_provider_proxy",
        ]
        assert decoded_token["gitlab_instance_id"] == "1234"
        assert parsed_response["expires_at"] == decoded_token["exp"]

        mock_track_internal_event.assert_called_once_with(
            "request_complete_code",
            category="ai_gateway.api.v1.code.user_access_token",
        )


class TestUserAccessTokenSuccessSaas:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["complete_code", "ai_gateway_model_provider_proxy"],
            gitlab_realm="saas",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_success(
        self, mock_client: TestClient, mock_track_internal_event
    ):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "saas",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)

        assert response.status_code == status.HTTP_200_OK

        parsed_response = response.json()
        decoded_token = jwt.decode(
            parsed_response["token"],
            TEST_PUBLIC_KEY,
            audience="gitlab-ai-gateway",
            algorithms=CompositeProvider.SUPPORTED_ALGORITHMS,
        )

        current_time = datetime.now(timezone.utc)
        current_time_posix = int(current_time.timestamp())

        assert decoded_token["iss"] == "gitlab-ai-gateway"
        assert decoded_token["sub"] == GLOBAL_USER_ID
        assert decoded_token["aud"] == "gitlab-ai-gateway"
        assert decoded_token["exp"] > current_time_posix
        assert (decoded_token["exp"]) <= int(
            (current_time + timedelta(hours=1)).timestamp()
        )
        assert decoded_token["nbf"] <= current_time_posix
        assert decoded_token["iat"] <= current_time_posix
        assert decoded_token["jti"]
        assert decoded_token["gitlab_realm"] == "saas"
        assert decoded_token["scopes"] == [
            "complete_code",
            "ai_gateway_model_provider_proxy",
        ]
        assert decoded_token["gitlab_instance_id"] == "1234"
        assert parsed_response["expires_at"] == decoded_token["exp"]

        mock_track_internal_event.assert_called_once_with(
            "request_complete_code",
            category="ai_gateway.api.v1.code.user_access_token",
        )


class TestUnauthorizedIssuer:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        return CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(
                scopes=["complete_code", "ai_gateway_model_provider_proxy"],
                issuer="gitlab-ai-gateway",
                gitlab_realm="self-managed",
                gitlab_instance_id="1234",
            ),
        )

    def test_failed_authorization_issuer(self, mock_client: TestClient):
        response = mock_client.post(
            "/code/user_access_token",
            headers={
                "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Instance-Id": "1234",
                "X-Gitlab-Realm": "self-managed",
            },
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Unauthorized to create user access token"


class TestCompleteCodePermission:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["complete_code"],
            gitlab_realm="self-managed",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_with_complete_code(self, mock_client: TestClient):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "self-managed",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)
        assert response.status_code == status.HTTP_200_OK


class TestDuoAgentPlatformPermission:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["ai_gateway_model_provider_proxy"],
            gitlab_realm="self-managed",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_with_model_provider_proxy(self, mock_client: TestClient):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "self-managed",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)
        assert response.status_code == status.HTTP_200_OK


class TestBothPermissions:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["complete_code", "ai_gateway_model_provider_proxy"],
            gitlab_realm="self-managed",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_with_both_permissions(
        self, mock_client: TestClient, mock_track_internal_event
    ):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "self-managed",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        parsed_response = response.json()
        decoded_token = jwt.decode(
            parsed_response["token"],
            TEST_PUBLIC_KEY,
            audience="gitlab-ai-gateway",
            algorithms=CompositeProvider.SUPPORTED_ALGORITHMS,
        )

        # Order might vary, so we use set comparison
        assert set(decoded_token["scopes"]) == {
            "complete_code",
            "ai_gateway_model_provider_proxy",
        }


class TestNoPermissions:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["random"],
            gitlab_realm="self-managed",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_with_no_permissions(self, mock_client: TestClient):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "self-managed",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Unauthorized to create user access token"


class TestUserAccessTokenJwtGenerationFailed:
    @pytest.fixture(name="config_values")
    def config_values_fixture(self):
        yield {
            "self_signed_jwt": {
                "signing_key": "BROKEN_KEY",
            }
        }

    def test_user_access_token_jwt_generation_failed(self, mock_client: TestClient):
        response = mock_client.post(
            "/code/user_access_token",
            headers={
                "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
                "Authorization": "Bearer 12345",
                "X-Gitlab-Authentication-Type": "oidc",
                "X-Gitlab-Instance-Id": "1234",
                "X-Gitlab-Realm": "self-managed",
            },
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()["detail"] == "Failed to generate JWT"


class TestSaasExtraClaims:
    @pytest.fixture(name="auth_user")
    def auth_user_fixture(self):
        claims = UserClaims(
            scopes=["complete_code", "ai_gateway_model_provider_proxy"],
            gitlab_realm="saas",
            gitlab_instance_id="1234",
        )
        return CloudConnectorUser(authenticated=True, claims=claims)

    def test_user_access_token_saas_with_extra_claims(self, mock_client: TestClient):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "saas",
            "X-Gitlab-Project-Id": "5678",
            "X-Gitlab-Namespace-Id": "9012",
            "X-Gitlab-Root-Namespace-Id": "3456",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)

        assert response.status_code == status.HTTP_200_OK

        parsed_response = response.json()
        decoded_token = jwt.decode(
            parsed_response["token"],
            TEST_PUBLIC_KEY,
            audience="gitlab-ai-gateway",
            algorithms=CompositeProvider.SUPPORTED_ALGORITHMS,
        )

        assert decoded_token["gitlab_realm"] == "saas"
        assert decoded_token["gitlab_project_id"] == "5678"
        assert decoded_token["gitlab_namespace_id"] == "9012"
        assert decoded_token["gitlab_root_namespace_id"] == "3456"

    def test_user_access_token_saas_without_extra_claims(self, mock_client: TestClient):
        headers = {
            "X-Gitlab-Global-User-Id": GLOBAL_USER_ID,
            "Authorization": "Bearer 12345",
            "X-Gitlab-Authentication-Type": "oidc",
            "X-Gitlab-Instance-Id": "1234",
            "X-Gitlab-Realm": "saas",
        }

        response = mock_client.post("/code/user_access_token", headers=headers)

        assert response.status_code == status.HTTP_200_OK

        parsed_response = response.json()
        decoded_token = jwt.decode(
            parsed_response["token"],
            TEST_PUBLIC_KEY,
            audience="gitlab-ai-gateway",
            algorithms=CompositeProvider.SUPPORTED_ALGORITHMS,
        )

        assert decoded_token["gitlab_realm"] == "saas"
        # Extra claims should be None if not provided
        assert decoded_token["gitlab_project_id"] is None
        assert decoded_token["gitlab_namespace_id"] is None
        assert decoded_token["gitlab_root_namespace_id"] is None
