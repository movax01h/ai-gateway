from datetime import datetime, timedelta, timezone
from unittest import mock

import botocore.session
import pytest
from botocore.exceptions import EndpointConnectionError
from botocore.stub import Stubber
from structlog.testing import capture_logs

from lib import aws_assume_role
from lib.aws_assume_role import (
    AssumeRoleError,
    clear_credentials_cache,
    get_assumed_role_credentials,
    validate_external_id,
    validate_iam_role_arn,
)

ROLE_ARN = "arn:aws:iam::123456789012:role/gitlab-bedrock"
EXTERNAL_ID = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_credentials_cache()
    yield
    clear_credentials_cache()


def _assume_role_response(
    suffix: str = "1", expires_in: timedelta = timedelta(hours=1)
) -> dict:
    return {
        "Credentials": {
            "AccessKeyId": f"ASIAEXAMPLEKEY00{suffix}",
            "SecretAccessKey": f"secret{suffix}",
            "SessionToken": f"token{suffix}",
            "Expiration": datetime.now(timezone.utc) + expires_in,
        },
        "AssumedRoleUser": {
            "AssumedRoleId": "AROAEXAMPLE:session",
            "Arn": "arn:aws:sts::123456789012:assumed-role/gitlab-bedrock/session",
        },
    }


@pytest.fixture(name="sts")
def sts_fixture(monkeypatch):
    """Real botocore session whose STS client is stubbed; source creds come from env."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIASOURCE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sourcesecret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")

    session = botocore.session.Session()
    client = session.create_client("sts")
    stubber = Stubber(client)
    stubber.activate()

    with (
        mock.patch.object(
            aws_assume_role.botocore.session, "get_session", return_value=session
        ),
        mock.patch.object(session, "create_client", return_value=client),
    ):
        yield stubber

    stubber.deactivate()


class TestValidateIamRoleArn:
    @pytest.mark.parametrize(
        "arn",
        [
            "arn:aws:iam::123456789012:role/gitlab-bedrock",
            "arn:aws:iam::123456789012:role/path/to/role+name=1,a.b@c_d-e",
            "arn:aws-us-gov:iam::123456789012:role/gov",
            "arn:aws-cn:iam::123456789012:role/cn",
        ],
    )
    def test_valid(self, arn):
        assert validate_iam_role_arn(arn) == arn

    @pytest.mark.parametrize(
        "arn",
        [
            "",
            "gitlab-bedrock",
            "arn:aws:iam::123:role/short-account",
            "arn:aws:iam::123456789012:user/not-a-role",
            "arn:aws:s3:::bucket",
            "arn:aws:iam::123456789012:role/",
            "arn:aws:iam::123456789012:role/has space",
            "arn:aws:iam::123456789012:role/" + "a" * 513,
        ],
    )
    def test_invalid(self, arn):
        with pytest.raises(ValueError, match="Invalid IAM role ARN"):
            validate_iam_role_arn(arn)


class TestValidateExternalId:
    @pytest.mark.parametrize(
        "value",
        [EXTERNAL_ID, "ab", "a" * 1224, "a+b=c,d.e@f:g/h-i_j"],
    )
    def test_valid(self, value):
        assert validate_external_id(value) == value

    @pytest.mark.parametrize(
        "value", ["", "a", "a" * 1225, "has space", "semi;colon", "star*", "na\u00efve"]
    )
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="Invalid external ID"):
            validate_external_id(value)


class TestGetAssumedRoleCredentials:
    def test_assumes_role_with_session_name(self, sts):
        sts.add_response(
            "assume_role",
            _assume_role_response(),
            {"RoleArn": ROLE_ARN, "RoleSessionName": "my-session"},
        )

        creds = get_assumed_role_credentials(ROLE_ARN, session_name="my-session")

        assert creds.to_litellm_params() == {
            "aws_access_key_id": "ASIAEXAMPLEKEY001",
            "aws_secret_access_key": "secret1",
            "aws_session_token": "token1",
        }
        sts.assert_no_pending_responses()

    def test_sends_external_id(self, sts):
        sts.add_response(
            "assume_role",
            _assume_role_response(),
            {
                "RoleArn": ROLE_ARN,
                "RoleSessionName": "gitlab-ai-gateway",
                "ExternalId": EXTERNAL_ID,
            },
        )

        get_assumed_role_credentials(ROLE_ARN, external_id=EXTERNAL_ID)

        sts.assert_no_pending_responses()

    def test_omits_external_id_when_absent(self, sts):
        sts.add_response(
            "assume_role",
            _assume_role_response(),
            {"RoleArn": ROLE_ARN, "RoleSessionName": "gitlab-ai-gateway"},
        )

        get_assumed_role_credentials(ROLE_ARN, external_id="")

        sts.assert_no_pending_responses()

    def test_separate_cache_entry_per_external_id(self, sts):
        sts.add_response("assume_role", _assume_role_response("A"))
        sts.add_response("assume_role", _assume_role_response("B"))
        sts.add_response("assume_role", _assume_role_response("C"))

        a = get_assumed_role_credentials(ROLE_ARN, external_id="id-a")
        b = get_assumed_role_credentials(ROLE_ARN, external_id="id-b")
        none = get_assumed_role_credentials(ROLE_ARN)
        again = get_assumed_role_credentials(ROLE_ARN, external_id="id-a")

        assert a.access_key == "ASIAEXAMPLEKEY00A"
        assert b.access_key == "ASIAEXAMPLEKEY00B"
        assert none.access_key == "ASIAEXAMPLEKEY00C"
        assert again == a
        sts.assert_no_pending_responses()

    def test_rejects_invalid_external_id_before_calling_sts(self, sts):
        with pytest.raises(ValueError, match="Invalid external ID"):
            get_assumed_role_credentials(ROLE_ARN, external_id="bad id")

    def test_sts_access_denied_raises_assume_role_error(self, sts):
        sts.add_client_error(
            "assume_role",
            service_error_code="AccessDenied",
            service_message="not authorized: secret-detail",
            http_status_code=403,
        )

        with capture_logs() as logs:
            with pytest.raises(AssumeRoleError) as exc_info:
                get_assumed_role_credentials(ROLE_ARN)

        assert exc_info.value.error_code == "AccessDenied"
        assert exc_info.value.role_arn == ROLE_ARN
        assert (
            str(exc_info.value) == f"Failed to assume IAM role {ROLE_ARN}: AccessDenied"
        )
        assert {
            "event": "sts:AssumeRole failed",
            "log_level": "warning",
            "role_arn": ROLE_ARN,
            "sts_error_code": "AccessDenied",
        } in logs
        sts.assert_no_pending_responses()

    def test_botocore_error_raises_assume_role_error(self):
        credentials = mock.Mock()
        credentials.get_frozen_credentials.side_effect = EndpointConnectionError(
            endpoint_url="https://sts.example.invalid"
        )

        with mock.patch.object(
            aws_assume_role, "_build_credentials", return_value=credentials
        ):
            with pytest.raises(AssumeRoleError) as exc_info:
                get_assumed_role_credentials(ROLE_ARN)

        assert exc_info.value.error_code == "EndpointConnectionError"
        assert exc_info.value.role_arn == ROLE_ARN

    def test_evicts_oldest_role_when_cache_is_full(self, sts, monkeypatch):
        monkeypatch.setattr(aws_assume_role, "_MAX_CACHED_ROLES", 1)
        other_arn = "arn:aws:iam::123456789012:role/other"
        sts.add_response("assume_role", _assume_role_response("1"))
        sts.add_response("assume_role", _assume_role_response("2"))
        sts.add_response("assume_role", _assume_role_response("3"))

        get_assumed_role_credentials(ROLE_ARN)
        get_assumed_role_credentials(other_arn)
        again = get_assumed_role_credentials(ROLE_ARN)

        assert again.access_key == "ASIAEXAMPLEKEY003"
        sts.assert_no_pending_responses()

    def test_retries_sts_after_failure(self, sts):
        sts.add_client_error("assume_role", service_error_code="Throttling")
        sts.add_response("assume_role", _assume_role_response())

        with pytest.raises(AssumeRoleError):
            get_assumed_role_credentials(ROLE_ARN)

        assert get_assumed_role_credentials(ROLE_ARN).access_key == "ASIAEXAMPLEKEY001"

    def test_default_session_name(self, sts):
        sts.add_response(
            "assume_role",
            _assume_role_response(),
            {"RoleArn": ROLE_ARN, "RoleSessionName": "gitlab-ai-gateway"},
        )

        get_assumed_role_credentials(ROLE_ARN)

        sts.assert_no_pending_responses()

    def test_cached_per_role_without_second_sts_call(self, sts):
        sts.add_response("assume_role", _assume_role_response())

        first = get_assumed_role_credentials(ROLE_ARN)
        second = get_assumed_role_credentials(ROLE_ARN)

        assert first == second
        sts.assert_no_pending_responses()

    def test_separate_cache_entry_per_role(self, sts):
        other = "arn:aws:iam::123456789012:role/other"
        sts.add_response("assume_role", _assume_role_response("A"))
        sts.add_response("assume_role", _assume_role_response("B"))

        a = get_assumed_role_credentials(ROLE_ARN)
        b = get_assumed_role_credentials(other)

        assert a.access_key == "ASIAEXAMPLEKEY00A"
        assert b.access_key == "ASIAEXAMPLEKEY00B"

    def test_refreshes_when_credentials_near_expiry(self, sts):
        sts.add_response(
            "assume_role", _assume_role_response("old", timedelta(seconds=30))
        )
        sts.add_response("assume_role", _assume_role_response("new"))

        old = get_assumed_role_credentials(ROLE_ARN)
        new = get_assumed_role_credentials(ROLE_ARN)

        assert old.access_key == "ASIAEXAMPLEKEY00old"
        assert new.access_key == "ASIAEXAMPLEKEY00new"
        sts.assert_no_pending_responses()

    def test_rejects_invalid_arn_before_calling_sts(self, sts):
        with pytest.raises(ValueError, match="Invalid IAM role ARN"):
            get_assumed_role_credentials("nope")

    def test_rejects_invalid_session_name(self, sts):
        with pytest.raises(ValueError, match="role session name"):
            get_assumed_role_credentials(ROLE_ARN, session_name="x")

    def test_no_source_credentials(self, monkeypatch):
        session = mock.Mock()
        session.get_credentials.return_value = None
        monkeypatch.setattr(
            aws_assume_role.botocore.session, "get_session", lambda: session
        )

        with pytest.raises(AssumeRoleError, match="no AWS credentials"):
            get_assumed_role_credentials(ROLE_ARN)


def test_to_litellm_params_omits_missing_token():
    params = aws_assume_role.AssumedRoleCredentials("a", "b", None).to_litellm_params()

    assert params == {"aws_access_key_id": "a", "aws_secret_access_key": "b"}


def test_assumed_role_credentials_repr_hides_secrets():
    credentials = aws_assume_role.AssumedRoleCredentials(
        "AKIAEXAMPLE", "super-secret-key", "super-secret-token"
    )

    assert "super-secret-key" not in repr(credentials)
    assert "super-secret-token" not in repr(credentials)
    assert "super-secret-key" not in str(credentials)
    assert "AKIAEXAMPLE" in repr(credentials)
