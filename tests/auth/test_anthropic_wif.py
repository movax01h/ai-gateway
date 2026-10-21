from unittest.mock import patch

import pytest
from anthropic.lib.credentials import AccessToken, WorkloadIdentityCredentials

from ai_gateway.auth import anthropic_wif
from ai_gateway.model_selection.models import AnthropicWorkloadIdentityParams


@pytest.fixture(autouse=True)
def clear_caches():
    anthropic_wif.credentials.cache_clear()
    anthropic_wif._token_cache.cache_clear()  # pylint: disable=protected-access
    yield
    anthropic_wif.credentials.cache_clear()
    anthropic_wif._token_cache.cache_clear()  # pylint: disable=protected-access


@pytest.fixture(name="params")
def params_fixture():
    return AnthropicWorkloadIdentityParams(
        federation_rule_id="fdrl_test",
        organization_id="00000000-0000-0000-0000-000000000000",
        service_account_id="svac_test",
        workspace_id="wrkspc_test",
    )


class TestCredentials:
    def test_builds_sdk_credentials_from_params(self, params):
        creds = anthropic_wif.credentials(params)

        assert isinstance(creds, WorkloadIdentityCredentials)
        # pylint: disable=protected-access
        assert creds._federation_rule_id == "fdrl_test"
        assert creds._organization_id == "00000000-0000-0000-0000-000000000000"
        assert creds._service_account_id == "svac_test"
        assert creds._workspace_id == "wrkspc_test"

    def test_identity_token_comes_from_gcp_adc(self, params):
        creds = anthropic_wif.credentials(params)

        with patch("ai_gateway.auth.gcp.id_token", return_value="jwt") as mock_id:
            assert creds._identity_token_provider() == "jwt"  # pylint: disable=protected-access

        mock_id.assert_called_once_with("https://api.anthropic.com")

    def test_cached_per_equal_params(self, params):
        same = params.model_copy()
        other = params.model_copy(update={"workspace_id": "wrkspc_other"})

        assert anthropic_wif.credentials(params) is anthropic_wif.credentials(same)
        assert anthropic_wif.credentials(params) is not anthropic_wif.credentials(other)


class TestAccessToken:
    def test_exchanges_once_and_serves_cached_token(self, params):
        exchanges = []

        def provider(*, force_refresh: bool = False):
            exchanges.append(force_refresh)
            return AccessToken(token="sk-ant-oat01-test", expires_at=2**40)

        with patch.object(anthropic_wif, "credentials", return_value=provider):
            assert anthropic_wif.access_token(params) == "sk-ant-oat01-test"
            assert anthropic_wif.access_token(params) == "sk-ant-oat01-test"

        assert exchanges == [False]


class TestMergeOauthBetaHeader:
    @pytest.mark.parametrize(
        ("existing", "expected"),
        [
            (None, "oauth-2025-04-20"),
            ("", "oauth-2025-04-20"),
            (
                "interleaved-thinking-2025-05-14",
                "interleaved-thinking-2025-05-14,oauth-2025-04-20",
            ),
            (
                "oauth-2025-04-20, interleaved-thinking-2025-05-14",
                "oauth-2025-04-20,interleaved-thinking-2025-05-14",
            ),
        ],
    )
    def test_merge(self, existing, expected):
        assert anthropic_wif.merge_oauth_beta_header(existing) == expected


def test_oauth_beta_header_matches_sdk():
    from anthropic.lib.credentials import _constants

    assert anthropic_wif.OAUTH_API_BETA_HEADER == "oauth-2025-04-20"
    assert anthropic_wif.OAUTH_API_BETA_HEADER == _constants.OAUTH_API_BETA_HEADER
