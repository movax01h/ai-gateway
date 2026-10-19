import pytest
import requests
from pydantic import SecretStr

from lib.snowplow import set_bearer_token


@pytest.mark.parametrize(
    "api_key,expected_header",
    [
        pytest.param(None, None, id="no_api_key"),
        pytest.param(SecretStr(""), None, id="empty_api_key"),
        pytest.param(SecretStr("glsa-key"), "Bearer glsa-key", id="api_key"),
    ],
)
def test_set_bearer_token(api_key, expected_header):
    session = requests.Session()

    set_bearer_token(session, api_key)

    assert session.headers.get("Authorization") == expected_header
